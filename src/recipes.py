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

  def create_dictionaries(self,vocabulary_size=20000):
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
    
    self._ing2id = {item['name'].lower(): int(item['_id']) for item in agg}
    ids = list(self._ing2id.values())
    ids.sort()
    self._ing2id['NO_INGREDIENT'] = 0
    self._ing_id2nid = {}
    for i in range(len(ids)):
      nid = i+1
      self._ing_id2nid[ids[i]] = nid
    self._ing_id2nid[0] = 0
    self._ing_nid2id = dict(zip(self._ing_id2nid.values(), self._ing_id2nid.keys()))
    self._id2ing = dict(zip(self._ing2id.values(), self._ing2id.keys()))

    self._word2id = {}
    self._id2word = {} 
    self._unit2id = {}
    self._id2unit = {}

    ingredient_size = len(self._ing2id)
    unit_size = len(self._unit2id)

    return ingredient_size, unit_size, vocabulary_size

  def ings2ids(self, ings):
    tmp =  self._2(ings, self._ing2id, 0)
    return self._2(tmp, self._ing_id2nid, 0)
  
  def units2ids(self, units):
    return self._2(units, self._unit2id, 0)

  def words2ids(self, words):
    return self._2(words, self._word2id, 0)

  def ids2ings(self, ids):
    tmp = self._2(ids, self._ing_nid2id, 0)
    return self._2(tmp, self._id2ing, "NO_INGREDIENT")
  
  def ids2units(self, ids):
    return self._2(ids, self._id2unit, "NO_UNIT")

  def ids2words(self, ids):
    return self._2(ids, self._id2word, "UNKNOWN")

  def _2(self, names, d, undef):
    if type(names) in [list, range]:
      return [d.get(n, undef) for n in names]
    elif type(names) == np.ndarray:
      res = np.full_like(names, "", dtype=object)
      for i in range(res.shape[0]):
        res[i] = self._2(names[i], d, undef)
      return res
    else:
      return d.get(names, undef)

  def get_ingredient_batch(self, batchsize=25, max_ingredients=15):
    count = 0
    batch = np.zeros((batchsize, max_ingredients), dtype=np.float32)
    for doc in self.coll.find({},
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
