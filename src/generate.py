import json
import tensorflow as tf
import numpy as np
from recipes import Recipes
from projecttest import *

cfg = json.loads(open("config.json").read())
settings = cfg['network']
weights = settings['directories']['weight_dir']

recipes = Recipes()

recipes.create_dictionaries()
batch_size = 50

# Not resetting causes conflicts in jupyter as the previous executed graph is still active in memory
tf.reset_default_graph()


###### The generator network. ######
with tf.variable_scope("generator"):

    ##### placeholders  #####
    # for input vector, and lstm state
    with tf.variable_scope("inputs"):
        input_vec = tf.placeholder(
            tf.float32, [batch_size, z_size], name="input_vec")

        cell_state = tf.placeholder(
            tf.float32, [batch_size, lstm_memory_size], name="cell_state")
        hidden_state = tf.placeholder(
            tf.float32, [batch_size, lstm_memory_size], name="hidden_state")

    ##### embedding #####
    # it is not trainable as it was pre trained to a sufficient degree
    with tf.variable_scope("embedding"):
        init = tf.random_uniform_initializer(-1.0, 1.0)
        embeddings = tf.get_variable("embed",
                                     [ingredient_size, embedding_size],
                                     initializer=init,
                                     trainable=False
                                     )

    ##### RNN #####
    # nothing special about it. We repeatedly feed the input vector for each ingredient to be produced
    with tf.variable_scope("rnn"):
        lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_memory_size)
        cell = tf.nn.rnn_cell.DropoutWrapper(
            lstm, output_keep_prob=dropout_rate)
        zero_state = cell.zero_state(batch_size, tf.float32)

        init_state = tf.nn.rnn_cell.LSTMStateTuple(
            c=cell_state, h=hidden_state)
        inputs = [input_vec for i in range(max_ingredients)]

        lstm_out, lstm_state = tf.nn.static_rnn(
            cell, inputs, initial_state=init_state)

        # has shape [max_ingredients, batch_size, embedding], but we need batch_size major
        outputs = tf.transpose(lstm_out, [1, 0, 2])

    # get trainable variables for generator
    train_gen = tf.get_variable_scope().get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES)


##### The discriminator network #####
with tf.variable_scope("discriminator"):

    ##### placeholder for originals & batch creation #####
    with tf.variable_scope("input"):
        orig_recipes_placeholder = tf.placeholder(tf.int64,
                                                  [batch_size, max_ingredients],
                                                  name="orig_recipes_placeholder"
                                                  )
        orig_recipes = tf.nn.embedding_lookup(
            embeddings, orig_recipes_placeholder)

        # create input batch for discriminator
        batch = tf.concat([outputs, orig_recipes], axis=0)

    ##### CNN #####
    with tf.variable_scope("convolutions"):
        convs = []
        for i in range(2, max_ingredients+1):
            with tf.variable_scope("conv"+str(i)):
                conv = discriminator_convolution(batch, i)
                convs.append(conv)

        conv_outputs = tf.stack(convs, axis=1)

    ##### Single Node reduction #####
    with tf.variable_scope("readout"):
        logits = feed_forward_layer(conv_outputs, 1, None)

    # get trainable variables for discriminator
    train_dis = tf.get_variable_scope().get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES)

##### Evaluation and training of the network #####
with tf.variable_scope("evaluation"):

    # to look at what glorious ingredient compilations it produced
    readable_outputs = get_ings(embeddings, outputs)

    # to compute cross entropy for generator.
    # consists of only "1" because this is the discriminator label for "real"
    gen_labels = tf.ones((batch_size, 1))

    # only the first half of the output from the discriminator concerns the pictures
    # produced by the generator
    # so only get first half of logits (which equals the batch_size)
    gen_logits = logits[:batch_size]

    # for discriminator cross entropy.
    # first half of input are fake images ("0"), second half real ones ("1")
    dis_labels = tf.concat(
        (tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))), axis=0)
    dis_logits = logits

    # calculating the loss for generator and discriminator
    gen_loss_mult_all = generator_similarity_loss(outputs)
    gen_loss_multing = tf.reduce_mean(gen_loss_mult_all) * 1  # lambda_ing
    gen_loss_ce = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=gen_labels, logits=gen_logits)
    gen_loss = gen_loss_ce + gen_loss_multing

    dis_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=dis_labels, logits=dis_logits)

    # initialize optimizer
    gen_optimizer = tf.train.AdamOptimizer(learning_rate=gen_lr, beta1=beta1)
    dis_optimizer = tf.train.AdamOptimizer(learning_rate=dis_lr, beta1=beta1)

    # define training steps with respective variable lists
    global_step = tf.get_variable("global_step", [], tf.int32, trainable=False)
    gen_step = gen_optimizer.minimize(
        gen_loss, var_list=train_gen, global_step=global_step)
    dis_step = dis_optimizer.minimize(
        dis_loss, var_list=train_dis, global_step=global_step)

    # tensorboard things
    tf.summary.scalar("generator_loss", tf.reduce_mean(gen_loss))
    tf.summary.scalar("discriminator_loss", tf.reduce_mean(dis_loss))
    summaries = tf.summary.merge_all()
