# %%
import os, pymongo
from bson import json_util

# %%

config = "config.json"

with open(config) as conf:
    data = json.load(conf)
    storage = data['storage']
    CRAWL_FOLDER = storage['folder']
    RECIPE_ID_FILE = CRAWL_FOLDER + storage['id_file']
    RECIPE_JSON_DUMP = CRAWL_FOLDER + storage['json_file']
    RECIPE_COLLECTION = storage['collection']

# %%
client = pymongo.MongoClient('mongodb://localhost:27017')
db = client['iannwtf']
coll = db[RECIPE_COLLECTION]

count = 0
q = queue.Queue()
try:
    for doc in coll.find({}, {"title": 1, "subtitle": 1, "instructions": 1,
                              "miscellaneousText": 1, "ingredientsText": 1, "tags": 1, "_id": 0}):
        s = " ".join([doc['title'], doc['subtitle'],
                      doc['instructions'], doc['miscellaneousText'],
                      doc['ingredientsText']] + doc['tags']).replace("\n", " ")
        q.put(s)
        count += 1
        if count % 500 == 0:
            print(count)


finally:
    with open("storage/all_words.txt", "w+", encoding="utf-8") as f:
        while q.qsize():
            f.write(q.get())
            f.write(" ")
