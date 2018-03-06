# %%
import os, pymongo, json, queue
from bson import json_util

# %%

config = "config.json"
# load the database config
with open(config) as conf:
    data = json.load(conf)
    storage = data['storage']
    CRAWL_FOLDER = storage['folder']
    RECIPE_ID_FILE = CRAWL_FOLDER + storage['id_file']
    RECIPE_JSON_DUMP = CRAWL_FOLDER + storage['json_file']
    RECIPE_COLLECTION = storage['collection']

client = pymongo.MongoClient('mongodb://localhost:27017')
db = client['iannwtf']
coll = db[RECIPE_COLLECTION]

# %%
def extract_text():
    """Extracts text from text fields of the recipes and puts them all into a file for further processing."""
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

def create_mapping():
    """creates a mapping from foodids into the interval [1, #ingredients]"""
    agg = coll.aggregate([
        {"$project": {"_id": 0, "ingredients.foodId": 1}},
        {"$unwind": "$ingredients"},
        {"$group": {
            "_id": "null",
            "ids": {"$addToSet": "$ingredients.foodId"}
        }}
    ], allowDiskUse=True)

    out = next(agg)['ids']
    ids = [int(x) for x in out]
    ids.sort()

    return {_id: (i+1) for i, _id in enumerate(ids)}

def remap_food_ids():
    """remaps the foodids in the database into the interval [1, #ingredients]"""
    remap = create_mapping()
    # why use .update() when you can update yourself? fml
    for doc in coll.find():
        coll.delete_one(doc)
        for i in range(len(doc['ingredients'])):
            fId = int(doc['ingredients'][i]['foodId'])
            doc['ingredients'][i]['foodId'] = remap[fId]
        
        coll.insert_one(doc)
