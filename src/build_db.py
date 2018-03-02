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
def run(config      = "config.json",
        threads     = 16,
        min_stars   = 0,
        min_ratings = 0,
        renew       = True,
        categories  = range(1,200)
        ):
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
    Collects recipe ids from the url given above and for all categories in the list, and returns a list of ids.

    Only recipes with a rating better than min_stars and more ratings than min_ratings are considered.
    The collection is multi threaded, and will use the amount of threads given by `threads`.
    To ensure a hickup-less, and fail-save writing process the id's are given to a queue, which will write
    all ids into a csv file even in case of exceptions.

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
    Collects the json files for each id and writes them into the database.

    It iterates over the ids in recipes and pushes them into `collection`, while
    also adding it to queue to write them to a file.

    Params:
    -------
    recipes: list
        the list holding the recipe ids to fetch
    collection: mongoDB collection
        A collection object in which the recipes should be pushed
    queue: Synced Queue
        The queue to push the json objects in to save to file
    threads: int
        How many threads to use concurrent

    Returns:
    --------
    recipe_objects: list
        a list of json objects. Returned for convenience
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
    """
    Crawls the url above for recipe ids and returns them as a list.

    It also adds them to `writeQueue` to be written into a file as a safety measure.

    Params:
    -------
    category: int
        the category id to crawl for recipes
    writeQueue: synced queue
        the queue to store the ids into as strings
    min_stars: float
        Only recipes with at least this rating are considered
    min_ratings: int
        Only recipes with at least this many ratings are considered

    Returns:
    --------
    recipe_ids: list
        a list holding all recipe ids
    """
    # global url_base, min_ratings, min_stars, debug

    logging.debug(">> Collecting id's of category {}.".format(category))

    recipe_ids = []
    stop = False
    reason = ""
    page_nr = 0  # increased by 30 each step
    total_recipes = 0

    # as long as there is there are recipes to crawl we continue on
    while(not stop):
        stop = True
        # try to connect to the url ten times. in the worst case this takes 30 seconds.abs
        # but it helps with gateway errors and other temporary connection problems
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

        # if the page was not properly loaded for whatever reason, go to the next
        # This should technically not happen, but it occured exactly once?
        if (not page):
            page_nr += 30
            continue

        # if there is no recipe segment in the html page, continue to the next one
        tree = BeautifulSoup(page, 'html.parser')
        recipes = tree.find_all("li", id=re_recipes, class_="search-list-item")
        if not recipes:
            page_nr += 30
            continue
        total_recipes += len(recipes)


        for recipe in recipes:
            match = re_get_rating.findall(str(recipe))
            if not match: # regex to no result. weird, but move on
                logging.warning("No rating in recipe: ", str(recipe))
                continue
            try:
                ratings = match[0]
                no_of_ratings = ratings[0]
                rating = ratings[1]
            except KeyError: # This should literally never be happening, but it did. Why?
                logging.error("Wasn't able to extract rating")
                continue

            try:
                ratingf = float(rating)
            except ValueError: # Ratings are floats. They should be. Always.
                logging.error("Could not convert rating %s to float" %rating)
                continue

            if ratingf < min_stars: # Since recipes are ordered by rating by default we can stop here
                stop = True
                reason = "Recipe found below star value."
                break

            try:
                no_of_ratingsi = int(no_of_ratings)
            except ValueError: # This has no reason to fail. But hey.
                logging.error("Could not convert number of ratings %s into int." %no_of_ratings)
                continue
            
            if no_of_ratingsi >= min_ratings: # only add recipe if it has enough ratings.
                _id = re_get_id.findall(recipe['id'])
                recipe_ids.append(_id[0])

        page_nr += 30

    logging.debug(">> Collected {} of {} recipes from category {}. Last rating was {} and reason for stopping was {}.".format(len(recipe_ids), total_recipes, category, rating, reason))

    # add the recipes to the queue as a list seperated by commas.
    s = str(category) + ",".join([str(r) for r in recipe_ids]) + "\n"
    writeQueue.put(s)
        
    return recipe_ids

def collect_recipes(recipe):
    """
    Collects the json data from the ip for the given recipe and returns it.
    """
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
    """If started from the command line, offer commandline options to run it for convenience."""
    import argparse
    parser = argparse.ArgumentParser(description="Crawling the chefkoch api for recipes. It's fun!")
    parser.add_argument('--config', '--cfg', '-C', default="config.json", type=str, dest='config', help="the json config file to load the values for the database from. Defaults to 'config.json'.")
    parser.add_argument('--threads', '--cores', '-T', default=16, type=int, dest='threads', help="How many threads to use for the multi threaded parts. Defaults to 4")
    parser.add_argument('--min_stars', '--stars', '-S', default=0, type=float, dest='min_stars', help="Only collect recipes that have at least this good a rating. Defaults to 0.")
    parser.add_argument('--min_ratings', '--ratings', '-R', default=0, type=int, dest='min_ratings', help="Only collect recipes with at least that many ratings. Defaults to 0")
    parser.add_argument('--renew', '-R', dest='renew', action='store_true', help="If renew is `True`, the recipe id's are crawled from the web. When `False` it tries to read from a csv file. Defaults to True.")
    parser.add_argument('--no-renew', '-!R', dest='renew', action='stroe_false')
    parser.set_defaults(renew=True)
    parser.add_argument('--categories', '--cat', '--cat_range', '-CR', default=[1, 200], type=int, nargs=2, dest='categories', help="The lower and upper bound of category ids to try out. Defaults to 1-200.")

    config = parser.config
    threads = parser.threads
    min_stars = parser.min_stars
    min_ratings = parser.min_ratings
    renew = parser.renew
    cat = parser.categories
    categories = range(cat[0], cat[1])

    run(config=config,
        threads=threads,
        min_stars=min_stars,
        min_ratings=min_ratings,
        renew=renew,
        categories=categories
        )
