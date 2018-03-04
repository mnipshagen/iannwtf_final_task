import json
import tensorflow as tf
import numpy as np
from random import shuffle
import firebase_admin
from firebase_admin import firestore, credentials
# from pymongo import MongoClient as MC
import urllib.parse

from recipes import Recipes
from projecttest import get_ings


cred = credentials.Certificate("firebase_servie_account_key.json")
app = firebase_admin.initialize_app(cred)
db = firestore.client(app)
coll = db.collection('recipes')
# db_cfg = json.loads(open('db_cfg.json').read())
# user = db_cfg['user']
# pwd = db_cfg['pwd']
# client = MC('mongodb://anon:anon@ds255768.mlab.com:55768/ainnwtf')
# db = client['ainnwtf']
# coll = db['test']

cfg = json.loads(open("config.json").read())

settings = cfg['network']
directories = settings['directories']
hyperparams = settings['hyperparameters']

weight_dir = directories['weight_dir']

recipes = Recipes()
ingredient_size,_,_ = recipes.create_dictionaries()
ingredient_size = 3601
max_ingredients = hyperparams["max_ingredients"]
z_size = hyperparams["z_size"]
embedding_size = hyperparams["embedding_size"]
lstm_memory_size = embedding_size
dropout_rate = hyperparams["dropout_rate"]
batch_size = 50

def dfg():
    # Not resetting causes conflicts in jupyter as the previous executed graph is still active in memory
    tf.reset_default_graph()
    global input_vec, cell_state, hidden_state, zero_state, readable_outputs

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

    ##### Evaluation and training of the network #####
    with tf.variable_scope("evaluation"):
        # to look at what glorious ingredient compilations it produced
        readable_outputs = get_ings(embeddings, outputs)

def run(output_size=50000):
    dfg()
    # global input_vec, cell_state, hidden_state, zero_state, readable_outputs

    real_recipe_ids = []
    fake_recipe_ids = []

    with tf.Session() as session:
        ckpt = tf.train.latest_checkpoint(weight_dir)
        if ckpt is None:
            raise Exception("Could not find weights to load.")
        else:
            saver = tf.train.Saver()
            saver.restore(session, ckpt)

        iterations = output_size // batch_size + ( 0 if (output_size % batch_size == 0) else 1)
        counter = 0
        while counter < iterations:
            
            for real_rec in recipes.get_ingredient_batch(batch_size, max_ingredients):
                _state = session.run(zero_state)
                z = np.random.uniform(-1, 1, (batch_size, z_size))

                feed_dict = {cell_state: _state.c,
                             hidden_state: _state.h,
                             input_vec: z
                             }

                fake_rec = session.run([readable_outputs], feed_dict)[0]

                for i in range(len(real_rec)):
                    real_recipe_ids.append(real_rec[i])
                    fake_recipe_ids.append(fake_rec[i])
                
                counter += 1
                if counter >= iterations:
                    break

    if len(real_recipe_ids) != len(fake_recipe_ids):
        print("Length of recipe arrays are unequal with fake: {} and real: {}".format(len(fake_recipe_ids), len(real_recipe_ids)))

    real_recipe_ids = np.array(real_recipe_ids[:output_size])
    fake_recipe_ids = np.array(fake_recipe_ids[:output_size])

    real_recipes = recipes.ids2ings(real_recipe_ids)
    fake_recipes = recipes.ids2ings(fake_recipe_ids)

    real_obj = to_json(real_recipes, True)
    fake_obj = to_json(fake_recipes, False)

    objects = real_obj + fake_obj
    shuffle(objects)

    to_db(objects)
    # to_files(objects)

    # to_file(real_obj, True)
    # to_file(fake_obj, False)

def run_fake_only(output_size=50):
    dfg()

    recipes = []

    with tf.Session() as session:
        ckpt = tf.train.latest_checkpoint(weight_dir)
        if ckpt is None:
            raise Exception("Could not find weights to load.")
        else:
            saver = tf.train.Saver()
            saver.restore(session, ckpt)

        iterations = output_size // batch_size + ( 0 if (output_size % batch_size == 0) else 1)

        for i in range(iterations):
            _state = session.run(zero_state)
            z = np.random.uniform(-1, 1, (batch_size, z_size))

            feed_dict = {cell_state: _state.c,
                        hidden_state: _state.h,
                        input_vec: z
                        }

            fake_rec = session.run([readable_outputs], feed_dict)[0]
            for rec in fake_rec:
                recipes.append(rec)

    return recipes

def to_json(recipes, real):
    objects = []
    _id = 0
    for rec in recipes:
        lst = [str(x) for x in rec]
        o = {}
        o['ingredients'] = lst
        o['real'] = real
        o['correct'] = 0
        o['incorrect'] = 0
        o['random'] = int(np.random.rand() * 250000)
        objects.append(o)
        _id += 1

    return objects

def to_file(objects, real):
    file_name = "real.json" if real else "fake.json"
    open(file_name,"w+",encoding="utf-8").write(json.dumps(objects, ensure_ascii=False))

def to_files(objects, batch = 10):
    l = len(objects)
    files = l // batch + (0 if l%batch==0 else 1)
    for f in range(files):
        obj = objects[(f*batch):(f*batch+1)]
        to_str = json.dumps(obj, ensure_ascii=False)
        with open("./data/"+str(f)+".json","w+",encoding="utf-8") as fh:
            fh.write(to_str)

def to_db(objects):
    perbatch = 500
    l = len(objects)
    batches = l//perbatch + (1 if l%perbatch!=0 else 0)

    for b in range(batches):
        batch = db.batch()
        for doc in objects[(perbatch*b):(perbatch*(b+1))]:
            newRef = coll.document()
            batch.set(newRef, doc)
        batch.commit()