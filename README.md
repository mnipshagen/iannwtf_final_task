# Implementing Artificial Neural Networks with Tensorflow - Final Task: A recipe generator
## A project by Antonia Hain, Anton Laukemper & Moritz Nipshagen

## Preface
This project was developed in the context of the course **Implementing Artificial Neural Networks with Tensorflow** taught at Osnabrück University by Lukas Braun. Its goal was to test and enhance our skillset and understanding of neural networks and the tensorflow framework.

The code was developed and tested with
* python 3.6
* tensorflow 1.5
* MongoDB 3.6

## Install & Use
* MongoDB
    * Download & install [MongoDB](https://docs.mongodb.com/manual/installation/)
    * The application expects a local server running at `mongodb://localhost:27017`
    * The server either needs to host a database named `iannwtf` and a collection `recipes` or the `build_db.py` (see below) script can crawl and build the database from scratch. This takes a while.
    * A backup of the database can be downloaded from [Mo's Onedrive](https://1drv.ms/f/s!Am3LtCW8Ozvuh7J19osxy3B-qIucfQ) and restored with the `mongorestore` application bundled with the MongoDB server installation. The syntax is `mongorestore -d iannwtf <directory_backup>`. Also refer to [the documentation](https://docs.mongodb.com/manual/reference/program/mongorestore/).

## What this is

### Data collection and pre-processing
The `build_db.py` is the crawler script to fetch the data from the chefkoch api. It expects a local hosted mongo_db database running at port 27017 with a db named `iannwtf`. It will create a collection named `recipes` and remove it if it already exists.
