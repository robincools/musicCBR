# Springboard capstone: Content based music recommender system.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/robincools/musiccbr/main/main.py) [![Readthedocs](https://readthedocs.org/projects/musiccbr/badge/)](https://musiccbr.readthedocs.io/en/latest/index.html)

## Summary

This repository contains the source code for a content based recommender system that recommends music based on a input song. It makes recommandations based on 2 factors:
1. [Audio features](https://developer.spotify.com/documentation/web-api/reference/#category-tracks) (acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, valence) gathered with the Spotify API. These audio features are normalized to values between 0 and 1. A cosine similarity will make recommendations based on songs with the closest audio feature vectors.
2. Song lyrics which are embedded using the [Doc2Vec](https://arxiv.org/pdf/1405.4053.pdf) method. This gives a 300-dimensional vector for each song-lyric. New lyrics can be inferred with a neural-network trained with the [gensim library](https://radimrehurek.com/gensim/models/doc2vec.html). Recommendations are made by finding the closest vectors with cosine similarity.

## How to run

### Deployed application

The recommender system is deployed via a web application on the streamlit share servica and can be accessed trough the '[Open in Streamlit](https://share.streamlit.io/robincools/musiccbr/main/main.py)' button at the top of the ReadMe.

### Local

If you want to run the app locally, this can be easily done with docker, or by manually installing the dependencies with the requirements file. To use the application you need acces to the [Spotify API](https://developer.spotify.com/documentation/general/guides/app-settings/#register-your-app) and a [google advanced search engine](https://pypi.org/project/lyrics-extractor/) and get the API keys of those services. Once you have the API keys, create following file in the root directory of the project:

`.streamlit/secrets.toml`

Add the API keys to this file as follows:

``` 
SPOTIPY_CLIENT_ID = <YOUR_SPOTIPY_CLIENT_ID>
SPOTIPY_CLIENT_SECRET = <YOUR_SPOTIPY_CLIENT_SECRET>
GCS_API_KEY = <YOUR_GCS_API_KEY>
GCS_ENGINE_ID = <YOUR_GCS_ENGINE_ID>
```

If you want to store the song data from a MySQL database, you have to create one and add the data to it using the create_sql_tables.ipynb notebook. In the secrets.toml file you need to add the following (for a local MySQL database):
```
[mysql]
host = "localhost"
port = 3306
database = "<database_name>"
user = "<user_name>"
password = "<user_password>"
```
#### Docker

To run the app with Docker we first have to build the Docker image. Assuming Docker is running on you machine, this can be done with following command:

`$ docker build -t <USERNAME>/<YOUR_IMAGE_NAME> .` in the root directory

The application can then be deployed using following command:

`$ docker run -p 8501:8501 <USERNAME>/<YOUR_IMAGE_NAME>` 

You can acces the application on following adress:

`http://localhost:8501/` 

#### Pip


Install the dependencies by with pip by running following command:

`pip install -r requirements.txt`

Run the streamlit application by running

`streamlit run main.py` in the root directory.

You can acces the application on following adress:

`http://localhost:8501/` 

## Documentation

The documentation can be found vie this [link](https://musiccbr.readthedocs.io/en/latest/index.html) or via the badge at the top of the ReadMe.

## Repository Content

* main.py: streamlit frontend
* recommender.py: class to make recommendations
* helper_functions.py: extra functions needed to make the application work
* train_model: function to train the model on new dataset and save the new model
* Jupyter notebooks:
    * create_sql_tables.ipynb: add csv-data-files to MySQL database
    * Capstone_CBF_lyrics_data.ipynb: notebook to do data wrangling
    * Loudness_linear.ipynb: notebook to research the influence of the logaritmic nature of dB on the loudness audio features
* data/: csv-files containing the wrangled data and the id-lookup table
* datasets_raw/: unwrangled datasets
* model/: model-files of the trained Doc2Vec model
* docs/: Documentation files to make/update readthedocs page
* Dockerfile: Docker-file to create docker image
* requirements.txt: list of required python-dependencies

