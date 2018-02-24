import pymongo, json, numpy as np
from collections import Counter
from nltk.tokenize import RegexpTokenizer


class Recipes:

  def __init__(
      self,
      useMongo=True,
      config="config.json"
    ):

    with open(config) as cfg:
      self.data = json.load(cfg)
      storage = self.data['storage']
      CRAWL_FOLDER = storage['folder']
      RECIPE_COLLECTION = storage['collection']

    client = pymongo.MongoClient('mongodb://localhost:27017')
    self.db = client['iannwtf']
    self.coll = self.db[RECIPE_COLLECTION]

  def create_dictionaries(self):
    agg = self.coll.aggregate([
        {"$project": {"_id": 0, "ingredients.foodId": 1, "ingredients.name": 1}},
        {"$unwind": "$ingredients"},
        {
            "$group": {
                "_id": "$ingredients.foodId",
                "name": {"$first": "$ingredients.name"}
            }
        }
      ], allowDiskUse=True)

    self._ing2id = {item['name']: item['_id'] for item in agg}
    self._id2ing = dict(zip(self._ing2id.values(), self._ing2id.keys()))
    self._word2id = {}
    self._id2word = {} 
    self._unit2id = {}
    self._id2unit = {}

  def ings2ids(self, ings):
    return self._2(ings, self._ing2id)
  
  def units2ids(self, units):
    return self._2(units, self._unit2id)

  def words2ids(self, words):
    return self._2(words, self._word2id)

  def ids2ings(self, ids):
    return self._2(ids, self._id2ing)
  
  def ids2units(self, ids):
    return self._2(ids, self._id2unit)

  def ids2words(self, ids):
    return self._2(ids, self._id2word)

  def _2(self, names, d):
    if type(names) in [list, range, np.ndarray]:
      return [d.get(n, 0) for n in names]
    else:
      return d.get(names, 0)

  def get_ingredient_batch(self, batchsize=25, max_ingredients=15):
    count = 0
    batch = np.zeros((batchsize, max_ingredients), dtype=np.float32)
    for doc in self.coll.sample({},
                              {"ingredients.foodId":1, "_id":0},
                              no_cursor_timeout=True
                              ):
      _ids = [int(x['foodId']) for x in doc['ingredients']]
      if len(_ids) > max_ingredients:
          _ids = _ids[:max_ingredients]
      batch[count][:len(_ids)] = _ids[:]
      count += 1

      if count == batchsize:
          yield batch
          count = 0
          batch = np.zeros((batchsize, max_ingredients), dtype=np.float32)
