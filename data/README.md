# Data collection and pre-processing
The `build_db.py` is the crawler script to fetch the data from the chefkoch api. It expects a local hosted mongo_db database running at port 27017 with a db named `iannwtf`. It will create a collection named `recipes` and remove it if it already exists.
