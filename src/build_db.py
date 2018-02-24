# %% imports
import re, pymongo, os, time, sys, logging
from functools import partial
from bson import json_util
from urllib.request import urlopen, HTTPError
from bs4 import BeautifulSoup
from multiprocessing import Pool as ThreadPool, Manager
from bson.son import SON
from bson.raw_bson import RawBSONDocument
from collections import MutableMapping

# %% All the constants & config we need
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', filename='log.log', level=logging.DEBUG)
debug = False
verbose = False
# 0 is the page we are on, going up in steps of 30
# 1 is the category we are searching, 92 are cakes
url_base = "https://www.chefkoch.de/rs/s{0}t{1}/"
# we need to extract the number sequence of the id tag
# which is in the form "recipe-{idsequence}"
re_get_id = re.compile(r'\d+')
re_recipes = re.compile(r'recipe-\d+')
re_get_rating = re.compile(r'(P?\d+)\sBewertung.*Ø (P?\d(?:\.\d+)*)')

# this is the url we are getting the recipe information in json
# formatter need the recipe id
# fields holds the entries in the json object we are interested in
url_api = "http://api.chefkoch.de/v2/recipes/{}"
fields = [
    'id',
    'type',
    'title',
    'subtitle',
    'rating',
    'difficulty',
    'preparationTime',
    'servings',
    'instructions',
    'miscellaneousText',
    'ingredientsText',
    'tags',
    'cookingTime',
    'restingTime',
    'totalTime',
    'ingredientGroups',
    'categoryIds'
]

# %% Define everything. Run is the entry point.
def run():
    """
    This is the entrypoint for the crawler to run. It aint pretty but it works.
    It makes the variables threads, min_stars and min_ratings globally available,
    because pool.map only takes one iterable

    run first queries through the categories 10 to 169 and collects the recipes
    Afterwards it collects all the json objects for all recipes and pushes them into the db

    TODO:
        create currying functions to avoid globlas to pass variables
            - somewhat done using partial?
        Make args set-able by cmd parameters
    """

    config = "config.json"

    with open(config) as conf:
        data = json_util.loads(conf)
        storage = data['storage']
        CRAWL_FOLDER = storage['folder']
        RECIPE_ID_FILE = CRAWL_FOLDER + storage['id_file']
        RECIPE_JSON_DUMP = CRAWL_FOLDER + storage['json_file']
        RECIPE_COLLECTION = storage['collection']

    if not os.path.isdir(CRAWL_FOLDER):
        logging.info("Found no storage folder in working directory, creating one.")
        os.makedirs(CRAWL_FOLDER)

    threads = 16
    min_stars = 0 #3.5
    min_ratings = 0 #5
    renew = True

    # all the categories we want to search in
    categories = range(1,200)

    # get all recipes! Wohoo! GONNA CATCH 'EM ALL
    logging.info("Here we go. Collecting Recipes...")
    
    # If we do not want to recrawl everything, check the file and load up the ids from there
    if (not renew) and os.path.isfile(RECIPE_ID_FILE):
        logging.info("Found existing csv file. Loading it.")
        with open(RECIPE_ID_FILE,'r') as f:
            lines = f.readlines()[0].split(",")
            
        logging.info("Loading {} recipe ids".format(len(lines)))
        recipes = [int(x) for x in lines]
    else:
        # we renew the id list and crawl the web once again
        # we store each id list in a synced queue to ensure we write all found ids even when we crash
        try:
            writeQueue = Manager().Queue()
            recipes = collect(categories, writeQueue, min_stars, min_ratings, threads)
        except Exception as e:
            logging.error(e)
            logging.error("This went hella wrong. We got %d recipes anyway. Continuing as planned." %len(recipes))
        finally:
            with open(RECIPE_ID_FILE, 'w+') as f:
                while writeQueue.qsize():
                    f.write(writeQueue.get())
                    f.flush()

    logging.info("Got all the recipes. Time to collect...")

    # setting up the mongoDB connection and checking if the database is non-empty
    # this is not ideal, but works for now, until I figured out how to cleanly update
    # the database when the format changes slightly without having inconsistencies.
    # versioning, what is it good for.
    client = pymongo.MongoClient('mongodb://localhost:27017')
    db = client['iannwtf']
    coll = db[RECIPE_COLLECTION]
    if coll.count() != 0:
        coll.drop()
        logging.info("Non-empty collection. Dropping and recreating.")

    coll = db[RECIPE_COLLECTION]

    logging.info("Database connection opened. Query the api.")
    
    # collecting all objects in their json format and saving them to a file as well. Just to be sure,
    # or in case we do not want to rely on mongodb
    # emptying the queue for reuse. Just to be sure. It should be emtpy already anyway.
    while not writeQueue.empty():
        writeQueue.get()
    try:
        objects = fill(recipes, coll, writeQueue, threads)
    finally:
       # save the objects we have as json strings to a file
        with open(RECIPE_JSON_DUMP,"w+") as f:
            f.write("[")
            while writeQueue.qsize():
                    f.write(writeQueue.get())
                    f.flush()
            f.seek(-1, os.SEEK_END)
            f.truncate()
            f.write("]")

    logging.info("Got all the objects and pushed all {} in.".format(len(objects)))


def collect(categories, writeQueue, min_stars, min_ratings, threads):
    """
    Collects recipe ids from the url given above and for all categories in the list.
    Only recipes with a rating better than min_stars and more ratings than min_ratings are considered.
    The collection is multi threaded.

    Params:
    -------
    categories: list
        the category ids to visit
    writeQueue: synchornised queue
        the queue each thread puts its result in
    min_stars: float
        the minimum rating a recipe must have
    min_ratings: int
        the minimum amount of ratings a recipe must have
    threads: int
        number of threads to use
    
    Returns:
    --------
    recipes: list
        a list of all unique crawled ids
    """

    # since the function which collects the ids per category has more than 1 parameter, but the 
    # multithreaded map implementation does not like that, we are using partial to fix the other three
    func = partial(collect_recipe_ids, writeQueue=writeQueue, min_stars=min_stars, min_ratings=min_ratings)

    # This is were we keep our threads
    pool = ThreadPool(threads)

    ## multi threaded collection of several categories
    # count holds how many categories were crawled, r_count how many ids we extracted
    # recipes_sublists will be a list of list of all returned recipe lists
    count = 0
    r_count = 0
    recipes_sublists = []
    
    # here we go. Using imap for neat console output.
    try:
        for recipe_list in pool.imap(func, categories):
            recipes_sublists.append(recipe_list)
            r_count += len(recipe_list)
            count += 1
            print("{:.2f}% ({}/{}) categories were visited and {} recipes collected.".format((count*100/len(categories)), count, len(categories), r_count), end='\r', flush=True)
        print("{:.2f}% ({}/{}) categories were visited and {} recipes collected.".format((count*100/len(categories)), count, len(categories), r_count), flush=True)
    finally:
        # time to end the party. Even though all threads should be done by this point we are making sure
        pool.close()
        pool.join()
    
    ## transform into unique id list and return
    recipes = [item for sublist in recipes_sublists for item in sublist]
    recipes = list(set(recipes))

    return recipes

def fill(recipes, collection, queue, threads):
    """

    """

    pool = ThreadPool(threads)
    count = 0
    recipe_obj = []
    for obj in pool.imap(collect_recipes, recipes):
        if type(obj) in [dict, SON, RawBSONDocument, MutableMapping]:
            recipe_obj.append(obj)
            # one object per line
            queue.put((json_util.dumps(obj) + ","))
            collection.insert_one(obj)
        count += 1
        print("{:.2f}% ({}/{})".format((count*100/len(recipes)), count, len(recipes)), end='\r', flush=True)
    print("{:.2f}% ({}/{})".format((count*100/len(recipes)), count, len(recipes)), flush=True)
    pool.close()
    pool.join()

    return recipe_obj

def collect_recipe_ids(category, writeQueue, min_stars, min_ratings):
    # global url_base, min_ratings, min_stars, debug

    logging.debug(">> Collecting id's of category {}.".format(category))

    recipe_ids = []
    stop = False
    reason = ""
    page_nr = 0  # increase by 30 each step
    total_recipes = 0

    while(not stop):
        stop = True
        for attempt in range(10):
            try:
                url = url_base.format(page_nr, category)
                page = urlopen(url).read()
            except HTTPError as e:
                reason = "Encountered HTTP Error"
                logging.warning("HTTPError: ", e.reason, file=sys.stderr)
                if e.code == 404:
                    break
                time.sleep(3)
                continue
            except ConnectionError as e:
                logging.error("ConnectionError. Ouch.", file=sys.stderr)
                time.sleep(3)
                continue
            else:
                stop = False
                break
        
        if stop:
            break

        if (not page):
            continue
        tree = BeautifulSoup(page, 'html.parser')
        recipes = tree.find_all("li", id=re_recipes, class_="search-list-item")
        if not recipes:
            continue
        total_recipes += len(recipes)

        for recipe in recipes:
            match = re_get_rating.findall(str(recipe))
            if not match:
                logging.warning("No rating in recipe: ", str(recipe))
                continue
            try:
                ratings = match[0]
                no_of_ratings = ratings[0]
                rating = ratings[1]
            except KeyError:
                logging.error("Wasn't able to extract rating")
                continue

            try:
                ratingf = float(rating)
            except ValueError:
                logging.error("Could not convert rating %s to float" %rating)
                continue

            if ratingf < min_stars:
                stop = True
                reason = "Recipe found below star value."
                break

            try:
                no_of_ratingsi = int(no_of_ratings)
            except ValueError:
                logging.error("Could not convert number of ratings %s into int." %no_of_ratings)
            
            if no_of_ratingsi >= min_ratings:
                _id = re_get_id.findall(recipe['id'])
                recipe_ids.append(_id[0])

        page_nr += 30

    logging.debug(">> Collected {} of {} recipes from category {}. Last rating was {}.".format(len(recipe_ids), total_recipes, category, rating))

    s = str(category) + ",".join([str(r) for r in recipe_ids]) + "\n"
    writeQueue.put(s)
        
    return recipe_ids

def collect_recipes(recipe):
    def flatten_ingredients(ingredientgroups):
        tmp = [ing for group in ingredientgroups for ing in group['ingredients']]
        res = []
        for ing in tmp:
            insert = True
            foodId = ing['foodId']
            unitId = ing['unitId']
            for x in res:
                if x['foodId'] == foodId:
                    if x['unitId'] == unitId:
                        x['amount'] += ing['amount']
                        insert = False
                        break
            if insert: res.append(ing)
        
        return res

    try:
        obj = None
        for attempt in range(10):
            try:
                url = url_api.format(recipe)
                page = urlopen(url).read()
            except (HTTPError) as e:
                if e.code == 404:
                    break
                time.sleep(3)
                continue
            except (ConnectionError) as e:
                time.sleep(3)
                continue
            success = True
            break

    except Exception as e:
        logging.error("Unexpected error! ", sys.exc_info()[0], file=sys.stderr)
    else:
        if success:
            obj = json_util.loads(page)
            keys = list(obj.keys())
            for k in keys:
                if k not in fields:
                    del obj[k]
                if k == 'id':
                    obj['_id'] = obj[k]
                    del obj[k]

            obj['ingredients'] = flatten_ingredients(obj['ingredientGroups'])
            del obj['ingredientGroups']

    finally:
        if obj == None:
            obj = json_util.loads('{"_id": 0}')
        
    return obj


#%%
if __name__ == '__main__':
    run()
