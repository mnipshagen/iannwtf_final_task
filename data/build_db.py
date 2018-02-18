# %% imports
import re, pymongo
from bson import json_util
from urllib.request import urlopen, HTTPError
from bs4 import BeautifulSoup

# %% some constant stuff we need

# 0 is the page we are on, going up in steps of 30
# 1 is the category we are searching, 92 are cakes
url_base = "https://www.chefkoch.de/rs/s{0}t{1}/"
# we need to extract the number sequence of the id tag
# which is in the form "recipe-{idsequence}"
re_get_id = re.compile(r'\d+')
re_recipes = re.compile(r'recipe-\d+')
re_get_rating = re.compile(r'Ø (P?\d(?:\.\d+)*)')

# %% and now get all the things!

recipe_ids = []
low_rating = False
no_results = False
page_nr = 0 # increase by 30 each step
category = 92 # caaaaaaake

while(not (low_rating or no_results)):
    # print(">>> Starting with page {0} in category {1}.".format((page_nr//30),category))
    try:
        url = url_base.format(page_nr, category)
        page = urlopen(url).read()
    except HTTPError as e:
        no_results = True
        print(e.message)
        break
    soup = BeautifulSoup(page, 'html.parser')
    recipes = soup.find_all("li", id=re_recipes)
    # print(">>> Found {} recipes...".format(len(recipes)))
    for recipe in recipes:
        rating = re_get_rating.findall(str(recipe))
        if float(rating[0]) < 4.5:
            low_rating = True
            break
        _id = re_get_id.findall(recipe['id'])
        recipe_ids.append(_id[0])
    # print(">>> Indexed all recipes. Last rating was {}".format(rating))
    page_nr += 30


recipe_ids = list(set(recipe_ids))
# %% We now have a set of recipe_ids. Sweet. Query the api
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


client = pymongo.MongoClient('mongodb://localhost:27017')
db = client['iannwtf']
coll = db['recipes']


# %%
for recipe_id in recipe_ids:
    try:
        url = url_api.format(recipe_id)
        page = urlopen(url).read()
    except HTTPError as e:
        print(e.message)
        break

    obj = json_util.loads(page)
    keys = list(obj.keys())
    for k in keys:
        if k not in fields:
            del obj[k]

    coll.insert_one(obj)
