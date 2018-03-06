import json
import os
import time
import argparse

import numpy as np
import tensorflow as tf
from tensorflow import variable_scope as vs

from recipes import Recipes


class Embed():
    """This class' purpose is to create a pre-trained embedding"""
    def __init__(self, config="config.json"):
        """Load files and setup recipe helper"""
        recipes = Recipes()
        self.recipes = recipes
        vocab_size = 20000
        ing_size, unit_size, vocab_size = recipes.create_dictionaries(vocab_size)
        self.ing_size = ing_size
        self.unit_size = unit_size
        self.vocab_size = vocab_size

        with open(config, "r", encoding="utf-8") as cfg:
            settings = json.load(cfg)["network"]
        self.settings = settings

        # these were the control id's
        ing_ids = recipes.ings2ids(['kartoffel(n)','knoblauch','schokolade','nudeln'])
        
        self.dfg(ing_ids)

    def dfg(self, ing_ids):
        """Create the graph!"""
        settings = self.settings

        # load and set hyperparameters
        hyperparams = settings["hyperparameters"]

        embedding_size = hyperparams["embedding_size"]
        noise_samples = 64
        ing_size = self.ing_size

        # just in case
        tf.reset_default_graph()

        # each input is a batch of ingredient ids and a batch of context's
        # the learning rate is a placeholder to be changeable by parameter
        with vs("input"):
            input_ings = tf.placeholder(dtype=tf.int32, shape=[None], name="ingredients")
            input_recipes = tf.placeholder(dtype=tf.int64, shape=[None,1], name="recipes")
            learningrate = tf.placeholder(dtype=tf.float32, shape=[], name="learning_rate")
        
        # setup the embedding and create the lookup tensor
        with vs("embeddings"):
            init = tf.random_uniform_initializer(-1.0, 1.0)
            ing_embedding_w = tf.get_variable("ing_embedding", [ing_size, embedding_size], initializer=init)

            ing_embedding = tf.nn.embedding_lookup(ing_embedding_w, input_ings)

        # We use nce loss with pretty much the same setup as in class / tensorflow tutorial
        with vs("output_and_loss"):
            init = tf.truncated_normal_initializer(stddev=1.0 / np.sqrt(embedding_size))
            context_w = tf.get_variable("context_weight", [ing_size, embedding_size], initializer=init)
            context_b = tf.get_variable("context_biases", [ing_size], initializer=tf.zeros_initializer())

            loss = tf.reduce_mean(tf.nn.nce_loss(
                                    weights=context_w,
                                    biases=context_b,
                                    labels=input_recipes,
                                    inputs=ing_embedding,
                                    num_sampled=noise_samples,
                                    num_classes=ing_size
                                ))

            tf.summary.scalar("nce_loss", loss)

        # Simple gradient descent algorithm and minimise nce loss
        with vs("lets_backpropagate"):
            global_step = tf.get_variable("global_step", [], dtype=tf.int32, initializer=tf.zeros_initializer(), trainable=False)
            optimizer = tf.train.GradientDescentOptimizer(learningrate)
            step = optimizer.minimize(loss, global_step=global_step, name="train_step")
        
        validate = self.nearest_neighbours(context_w, ing_ids)
        summaries = tf.summary.merge_all()

        # need to reference those from the training function
        self.input_ings = input_ings
        self.input_recipes = input_recipes
        self.input_lr = learningrate
        self.context = context_w
        self.loss = loss
        self.step = step
        self.summaries = summaries
        self.validate = validate

    def train(self,
              batch_size=128,
              epochs=2,
              learning_rate=1
              ):
              """Train the network with given batch_size and learning rate for given epochs"""
        settings = self.settings
        directories = settings["directories"]

        max_ingredients = self.ing_size
        embedding_weights = directories["embedding_weights"]
        
        train_writer = tf.summary.FileWriter("./embed_summaries/train", tf.get_default_graph())
        # in the actual graph, we will only need the embedding matrix.
        saver = tf.train.Saver(
            var_list={'ingredient_embedding': self.context}) 

        with tf.Session() as session:
            # load last weights if possible
            ckpt = tf.train.latest_checkpoint(embedding_weights)
            if ckpt != None:
                saver.restore(session, ckpt)
            else:
                session.run(tf.global_variables_initializer())

            valid = session.run(self.validate)
            print("Nearest neighbours before training:")
            for i in valid:
                print(self.recipes.ids2ings(i))

            for epoch in range(1,epochs+1):
                start_time = time.time()

                for recipes in self.recipes.get_ingredient_batch(batch_size, max_ingredients):
                    ing, context = self.get_ingredient_and_context(recipes)
                    summaries, _ = session.run(
                            [self.summaries, self.step],
                            feed_dict={
                                self.input_ings: ing,
                                self.input_recipes: context,
                                self.input_lr: learning_rate
                                }
                            )
                    step = tf.train.get_global_step()
                    train_writer.add_summary(summaries, step)
                
                time_took = time.time() - start_time
                print("Done with epoch {0}, took {1: .3f}s. Estimated remaining time: {2: .3f}"\
                      .format(epoch, time_took, time_took * (epochs-epoch)))
                
                saver.save(session, embedding_weights+"model.ckpt", global_step=step)

                valid = session.run(self.validate)
                print("Nearest neighbours after epoch %d:" %epoch)
                for i in valid:
                    print(self.recipes.ids2ings(i))

        train_writer.close()

    def get_ingredient_and_context(self, recipes):
        """
        From a list of recipes, build tuples of ingredients.
        
        For each ingredient there will be a tuple with each other ingredient from the recipe.
        So if the recipe is [12, 37, 843] there will be tuples: (12, 37), (12, 843) and (37, 843)

        The tuples are then shuffled and returned are two lists: a list of ingredient id's and a list of contexts,
        corresponding to each other.
        """
        pairs = []
        for recipe in recipes:
            recipe = recipe[np.where(recipe!=0)]
            for i in range(len(recipe)):
                for j in range(len(recipe)):
                    if j == i:
                        continue
                    else:
                        pairs.append((recipe[i],recipe[j]))

        np.random.shuffle(pairs)
        pairs = np.array(pairs)
        return pairs[:,0], np.reshape(pairs[:,1],[-1,1])


    def nearest_neighbours(self, embeddings, words, k=5):
        """get the `k` nearest neighbours to each word in words"""
        k += 1

        normed_embedding = tf.nn.l2_normalize(embeddings, axis=1)
        if (type(words) != list):
            array = words
        else:
            words = np.array(words)
            array = tf.nn.embedding_lookup(embeddings, words)

        normed_array = tf.nn.l2_normalize(array, axis=1)

        cosine_similarity = tf.matmul(
            normed_array, tf.transpose(normed_embedding, [1, 0]))
        values, sorted_cosine_similarity = tf.nn.top_k(cosine_similarity, k=k)
        #closest_words = tf.argmax(cosine_similarity, 1)  # shape [batch_size], type int64
        return sorted_cosine_similarity

    def info(self):
        """Debugging function to check for shape consistency"""
        settings = self.settings
        directories = settings["directories"]

        embedding_weights = directories["embedding_weights"]
        saver = tf.train.Saver(
            var_list={'ingredient_embedding': self.context})
        with tf.Session() as session:
            ckpt = tf.train.latest_checkpoint(embedding_weights)
            if ckpt != None:
                saver.restore(session, ckpt)
            context = session.run(self.context)
            print(context.shape)

if __name__ == '__main__':
    """Make this script easily runnable from commandline by handling cli arguments and then start training."""
    parser = argparse.ArgumentParser(description="Pretrain an embedding for later use")
    parser.add_argument('--batch_size','--bs','-B', default=128, type=int, dest='batch_size', help="The batch size to use while training")
    parser.add_argument('--epochs', "-E", default=2, type=int, dest='epochs', help="Amount of epochs to train the network")
    parser.add_argument('--learning_rate', '--lr', '-L', default=1, type=float, dest='learning_rate', help="The learning rate to use")

    args = parser.parse_args()
    batchsize = args.batch_size
    epochs = args.epochs
    lr = args.learning_rate

    net = Embed()
    # net.train(batchsize, epochs, lr)
    net.info()
