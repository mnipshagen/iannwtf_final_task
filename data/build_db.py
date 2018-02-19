# %% imports
import re, pymongo, os, sys, numpy as np
from functools import partial
from bson import json_util
from urllib.request import urlopen, HTTPError
from bs4 import BeautifulSoup
from multiprocessing import Pool as ThreadPool

# %% All the constants we need
debug = True
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

    CRAWL_FOLDER = "storage/"
    RECIPE_ID_FILE = CRAWL_FOLDER + "recipes.csv"

    if not os.path.isdir(CRAWL_FOLDER):
        print("Found no storage folder in working directory, creating one.")
        os.makedirs(CRAWL_FOLDER)

    threads = 12
    min_stars = 0 #3.5
    min_ratings = 0 #5
    renew = True

    # all the categories we want to search in
    categories = range(1,200)

    # get all recipes! Wohoo! GONNA CATCH 'EM ALL
    if debug: print("Here we go. Collecting Recipes...")
    if (not renew) and os.path.isfile(RECIPE_ID_FILE):
        print("Found existing csv file. Loading it.")
        with open(RECIPE_ID_FILE,'r') as f:
            lines = f.readlines().split(",")
        print("Loading {} recipe ids".format(len(lines)))
        recipes = [int(x) for x in lines]
    else:
        recipes = collect(categories, min_stars, min_ratings, threads)
        
        with open(RECIPE_ID_FILE, 'w+') as f:
            s = ",".join([str(r) for r in recipes])
            f.write(s)

    if debug: print("Got all the recipes. Time to collect...")

    client = pymongo.MongoClient('mongodb://localhost:27017')
    db = client['iannwtf']
    coll = db['recipes']
    if coll.count() != 0:
        coll.drop()
        print("Non-empty collection. Dropping and recreating.")

    coll = db['recipes']

    if debug: print("Database connection opened. Query the api.")
    objects = fill(recipes, threads)
    if debug: print("Got all the objects. Now push all {} in.".format(len(objects)))

    coll.insert_many(objects)

def collect(categories, min_stars, min_ratings, threads):
    # global debug

    func = partial(collect_recipe_ids, min_stars = min_stars, min_ratings = min_ratings)

    pool = ThreadPool(threads)
    ## multi threaded collection of several categories
    recipes_sublists = pool.map(func, categories)
    pool.close()
    pool.join()
    ## transform into unique id list
    recipes = [item for sublist in recipes_sublists for item in sublist]
    recipes = list(set(recipes))

    return recipes

def fill(recipes, threads):
    # global debug

    pool = ThreadPool(threads)
    json = pool.map(collect_recipes, recipes)
    pool.close()
    pool.join()

    return json

def collect_recipe_ids(category, min_stars, min_ratings):
    # global url_base, min_ratings, min_stars, debug

    if debug: print(">> Collecting id's of category {}...".format(category))

    recipe_ids = []
    stop = False
    reason = ""
    page_nr = 0  # increase by 30 each step
    total_recipes = 0

    while(not stop):
        if debug and verbose: print(">> Starting with page {0} in category {1}.".format((page_nr//30),category))
        try:
            url = url_base.format(page_nr, category)
            page = urlopen(url).read()
        except HTTPError as e:
            stop = True
            reason = "Encountered HTTP Error"
            print("HTTPError: ", e.reason)
            break
        tree = BeautifulSoup(page, 'html.parser')
        recipes = tree.find_all("li", id=re_recipes, class_="search-list-item")
        total_recipes += len(recipes)

        for recipe in recipes:
            match = re_get_rating.findall(str(recipe))
            if not match:
                if debug: print("No rating in recipe: ", str(recipe))
                continue
            rating = match[0]
            no_of_ratings = rating[0]
            rating = rating[1]
            if debug and verbose: print(">> Recipe with rating of: {0: .2f} and {1: d} ratings".format(float(rating), int(no_of_ratings)))
            if float(rating) < min_stars:
                stop = True
                reason = "Recipe found below star value."
                break
            if int(no_of_ratings) >= min_ratings:
                _id = re_get_id.findall(recipe['id'])
                recipe_ids.append(_id[0])
                if debug and verbose: print(">>> was inserted")

        page_nr += 30

    if debug: print(">> Collected {} of {} found recipes".format(len(recipe_ids), total_recipes))
    if debug: print(">> Indexed all recipes. Last rating was {}".format(rating))

    if debug: print(">> Collected {} from category {}.".format(len(recipe_ids), category))
        
    return recipe_ids

def collect_recipes(recipe):
    # global url_api, fields

    try:
        url = url_api.format(recipe)
        page = urlopen(url).read()
    except HTTPError as e:
        print("HTTPError: ", e.reason)
        return

    obj = json_util.loads(page)
    keys = list(obj.keys())
    for k in keys:
        if k not in fields:
            del obj[k]

    return obj

#%%
if __name__ == '__main__':
    run()
