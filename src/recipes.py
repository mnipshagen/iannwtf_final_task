import pymongo
import json
import numpy as np
from collections import Counter
from nltk.tokenize import RegexpTokenizer

class Recipes:
  """A helper class to interface with the dataset"""

  def __init__(
    self,
    useMongo=True,
    config="config.json"
  ):
    """set up connection to db"""

    with open(config) as cfg:
      self.data = json.load(cfg)
      storage = self.data['storage']
      CRAWL_FOLDER = storage['folder']
      RECIPE_COLLECTION = storage['collection']

    client = pymongo.MongoClient('mongodb://localhost:27017')
    self.db = client['iannwtf']
    self.coll = self.db[RECIPE_COLLECTION]

  def create_dictionaries(self, vocabulary_size=20000):
    """
    Counts all unique ingredients and creates a dictionary mapping.
    
    We aggregate the data in the database to get one object per ingredient,
    each containing its foodId and name. The ingredients should be already remapped
    into the interval [1, #ingredients].
    We then build two dictionary: ingredient name to id and vice versa
    There are also empty dictionaries for word2id and unit2id, since it was created
    with expansion in mind.
    
    Returns:
    --------
    lengths: tuple
      the length of each dictionary mapping
    """
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

    # mapping names to ids. Dictionary comprehensions be praised.
    self._ing2id = {item['name'].lower(): int(item['_id']) for item in agg}
    # For whatever hecking reason, the chefkoch database has two entries for 'miracel whip'
    # they only differ in the capitalisation of ONE letter... but since the dictionary uses
    # lower case names, this broke the code
    # Since this was noticed fairly late into development, a more or less well suited fix was implemented
    self._ing2id['miracel whipmyass'] = 1362
    self._ing2id['NO_INGREDIENT'] = 0
    # and in reverse, for that super speed performance
    self._id2ing = dict(zip(self._ing2id.values(), self._ing2id.keys()))

    self._word2id = {}
    self._id2word = {}
    self._unit2id = {}
    self._id2unit = {}

    ingredient_size = len(self._ing2id)
    unit_size = len(self._unit2id)

    return ingredient_size, unit_size, vocabulary_size

  def ings2ids(self, ings):
    """returns ids for a list of ingredient names"""
    return self._2(ings, self._ing2id, 0)

  def units2ids(self, units):
    """returns ids for a list of unit names"""
    return self._2(units, self._unit2id, 0)

  def words2ids(self, words):
    """returns ids for a list of words"""
    return self._2(words, self._word2id, 0)

  def ids2ings(self, ids):
    """returns ingredient names for a list of ids"""
    return self._2(ids, self._id2ing, "NO_INGREDIENT")

  def ids2units(self, ids):
    """returns unit names for a list of ids"""
    return self._2(ids, self._id2unit, "NO_UNIT")

  def ids2words(self, ids):
    """returns words for a list of ids"""
    return self._2(ids, self._id2word, "UNKNOWN")

  def _2(self, names, d, undef):
    """Extracts the values for the keys `names` out of dictionary `d` and uses `undef` if not found"""
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
    """Generate a batch of original recipes and pad or cut to length `max_ingredients`"""
    count = 0
    batch = np.zeros((batchsize, max_ingredients), dtype=np.float32)
    cursor = self.coll.find(
      {}, {"ingredients.foodId": 1, "_id": 0}, no_cursor_timeout=True)

    recipes = np.array(list(cursor))
    np.random.shuffle(recipes)

    for doc in recipes:
      # ids could be strings. not cool.
      _ids = [int(x['foodId']) for x in doc['ingredients']]
      # cut off if necessary
      if len(_ids) > max_ingredients:
        _ids = _ids[:max_ingredients]
      batch[count][:len(_ids)] = _ids[:]
      count += 1

      if count == batchsize:
        # the batch is ready to be served
        yield batch
        count = 0
        batch = np.zeros((batchsize, max_ingredients),
                 dtype=np.float32)
