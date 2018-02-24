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
    pass

  def ing2ids(self):
    pass
  
  def unit2ids(self):
    pass

  def words2ids(self):
    pass

  def ids2ing(self):
    pass
  
  def ids2unit(self):
    pass

  def ids2words(self):
    pass

  def get_ingredient_batch(self, batchsize=25, max_ingredients=15):
    count = 0
    batch = np.zeros((batchsize, max_ingredients), dtype=np.int16)
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
          batch = np.zeros((batchsize, max_ingredients), dtype=np.int16)
