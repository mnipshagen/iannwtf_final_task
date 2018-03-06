# Implementing Artificial Neural Networks with Tensorflow - Final Task: A recipe generator
## A project by Antonia Hain, Anton Laukemper & Moritz Nipshagen

## Preface
This project was developed in the context of the course **Implementing Artificial Neural Networks with Tensorflow** taught at Osnabrück University by Lukas Braun. Its goal was to test and enhance our skillset and understanding of neural networks and the tensorflow framework.

The code was developed and tested with
* python 3.6
* tensorflow 1.5 & 1.3
* MongoDB 3.6
* Firebase 03/2018

## An overview
* MongoDB
    * Download & install [MongoDB](https://docs.mongodb.com/manual/installation/)
    * The application expects a local server running at `mongodb://localhost:27017`
    * The server either needs to host a database named `iannwtf` and a collection `recipes` or the `build_db.py` (see below) script can crawl and build the database from scratch. This takes a while.
    * A backup of the database can be downloaded from [here](https://1drv.ms/f/s!AvTNk9gMBgjaegEexxiqTOTx1gs) and restored with the `mongorestore` application bundled with the MongoDB server installation. The syntax is `mongorestore -d iannwtf <directory_backup>` or `mongorestore ./` if the iannwtf folder is in the working directory. Also refer to [the documentation](https://docs.mongodb.com/manual/reference/program/mongorestore/).
* The Jupyter-Notebook file holds the actual network code.
* The backup is already processed, but if the database is created from scratch, the `remap_food_ids` function from the `process.py` script must be called beforehand.
* 