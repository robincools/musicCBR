import pandas as pd
import numpy as np
import spotipy
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, MiniBatchKMeans
from helper_functions import get_img_urls, clean_lyrics, authenticate_extract_lyrics, authenticate_spotify_api
import streamlit as st
from lyrics_extractor import SongLyrics
from gensim.models.doc2vec import Doc2Vec

class recommender:
  """
  A class to make song recommendations based on Spotify audio features and song lyrics.
  
  Parameters
  ----------
  track_id : str
    Spotify track ID of the track we want to make recommendations for.
  database : DataFrame
    Pandas dataframe with spotify ID's, audio features and lyrics of the tracks to make recommendations with.
  lookup_table : DataFrame
    Pandas dataframe with Spotify ID's and track names and artists.
  do_kmeans : boolean
    True if we want to do kmeans before recommendation.
  n_songs : int
    Number of songs to recommend.
  alpha: float
    Number between 0 and 1 to weigh the lyrics recommendation to the feature recommendation.
    
  Methods
  -------
  find_song()
    Returns the audio features from the Spotify-API and the lyrics scraped with the Lyrics-extractor package.
  kmeans_cluster(song)
    Kmeans-cluster audio features
  recommender_features(song, database)
    Calculate cosine similarities between the audio features of the song and the audio features of the songs in the database.
  recommender_lyrics(song, dataset)
    Calculate cosine similarities between the Doc2Vec vectors of the song and the songs in the database.
  content_based_recommender(self)
    Combine recommender_features and recommender_lyrics and get top recommendations.
  """
  
  def __init__(self, track_id, database, lookup_table, do_kmeans,  n_songs , alpha , n_clusters = 10):
    """
    Parameters
    ----------
    track_id : str
      Spotify track ID of the track we want to make recommendations for.
    database : DataFrame
      Pandas dataframe with spotify ID's, audio features and lyrics of the tracks to make recommendations with.
    lookup_table : DataFrame
      Pandas dataframe with Spotify ID's and track names and artists.
    do_kmeans : boolean
      True if we want to do kmeans before recommendation.
    n_songs : int
      Number of songs to recommend.
    alpha: float
      Number between 0 and 1 to weigh the lyrics recommendation to the feature recommendation.
    """
    self.track_id = track_id
    self.database = database
    self.n_songs = n_songs
    self.alpha = alpha
    self.n_clusters = n_clusters
    self.do_kmeans = do_kmeans
    self.lookup_table = lookup_table
    self.sp = authenticate_spotify_api()
    self.extract_lyrics = authenticate_extract_lyrics()

    
  @st.cache(allow_output_mutation=True)
  def find_song(self):

    """
    This function returns the audio features from the Spotify-API and the lyrics scraped with the Lyrics-extractor package.
    This functions inputs the song name, artist and a database to check if the song is already in the database.
    """
  
    # Initialize dictionary to story song data
    song_data = defaultdict()

    # Check if the song is already in the song database, so we don't need to search for the song again
    if self.database.loc[(self.database['id'] == self.track_id)].values in self.database.values:
      return self.database.loc[(self.database['id'] == self.track_id)]

    # Get the audio features of a song using the Spotify API and the track ID.
    audio_features = self.sp.audio_features(self.track_id)[0]

    # Store the audio features we need of the track in the dictionary
    song_data['id'] = [self.track_id]
    song_data['acousticness'] = audio_features['acousticness']
    song_data['danceability'] = audio_features['danceability']
    song_data['energy'] = audio_features['energy']
    song_data['instrumentalness'] = audio_features['instrumentalness']
    song_data['liveness'] = audio_features['liveness']
    song_data['loudness'] = audio_features['loudness']
    song_data['speechiness'] = audio_features['speechiness']
    song_data['tempo'] = audio_features['tempo']
    song_data['valence'] = audio_features['valence']

    # Get the name of the track
    track = self.sp.track(self.track_id)
    name = track['name']

    # Get the lyrics of the track using lyrics_extractor
    lyrics = self.extract_lyrics.get_lyrics(name)
    lyrics = lyrics['lyrics'].replace('\n',' ')
    
    # Store the lyrics in the dictionary
    song_data['Lyrics'] = lyrics

    # return dataframe of the song data dictionary
    return pd.DataFrame(song_data)
    
  def kmeans_cluster(self, song):
    """
    Do kmeans clustering on data. At this moment not used because not possible to use together with Doc2Vec
    """

    if song['id'].values not in self.database['id'].values:
      self.database = song.append(self.database) 
      i = 0
    else:
      i = self.database.index[(self.database['id'] == song['id'].values[0])][0]

    data_nolyrics = self.database[['danceability','energy', 'loudness','speechiness','acousticness','instrumentalness','liveness','valence','tempo']]
    data_nolyrics = (data_nolyrics - data_nolyrics.min())/(data_nolyrics.max() - data_nolyrics.min())

    kmeans = MiniBatchKMeans(self.n_clusters,batch_size = 200)  
    song_cluster_labels = kmeans.fit_predict(data_nolyrics)

    self.database['cluster_label'] = song_cluster_labels

    if i == 0:
      self.database.reset_index(inplace = True, drop = True)

    database_kmeans = self.database[self.database['cluster_label'] == self.database['cluster_label'][i]]
    database_kmeans = database_kmeans.drop('cluster_label', axis=1)
    database_kmeans.reset_index(inplace = True, drop = True)

    return database_kmeans

  def recommender_features(self, song, database):

    """
    This function calculates the cosine similarities based on the audio features. All features are normalized to values between zero and one.
    This function inputs the dataframe that the find_song function outputs and a database with songs.
    
    Parameters
    ----------
    song : DataFrame
      dataframe with id, audio features and lyrics of song
    database : DataFrame
      dataframe with id, audio features and lyrics of database of songs
    """

    # Check if song is in database, if not append database and song. If song is in database then find index of song.
    if song['id'].values not in database['id'].values:
      database = song.append(database) 
      i = 0
    else:
      i = database.index[(database['id'] == song['id'].values[0])][0]

    # Only keep audio features of database and normalize them
    database_norm = database[['danceability','energy', 'speechiness','acousticness','instrumentalness','liveness','valence','tempo', 'loudness']]
    database_norm = (database_norm - database_norm.min())/(database_norm.max() - database_norm.min())

    #  Get the cosine similarity between the song and the songs of the database
    cos_sim = cosine_similarity(database_norm.iloc[[i]], database_norm)
    cos_sim = np.transpose(cos_sim)

    # Add the cosine similarity to the database dataframe
    database['similarity_features'] = cos_sim

    # return dataframe with the id and the similarity scores of the features
    return database[['id','similarity_features']]

  def recommender_lyrics(self, song, dataset):
    """
    This function calculates the cosine similarity between the vector embedding of the song lyrics and the vector embeddings of the database of lyrics.
    This function inputs the dataframe that the find_song function outputs and a database with songs.
    
    Parameters
    ----------
    song : DataFrame
      dataframe with id, audio features and lyrics of song
    database : DataFrame
      dataframe with id, audio features and lyrics of database of songs
    """

    # Load the Doc2Vec model trained on the database of songs
    model = Doc2Vec.load('data/doc2vec_60epochs.model')

    # Extract the lyrics of the song dataframe
    lyrics_str = song['Lyrics'].values[0]

    # clean the lyrics using the clean_lyrics function
    lyrics_clean = clean_lyrics([lyrics_str])[0]
    
    # Infer the vector embedding of the lyrics using the Doc2Vec model.
    lyrics_vectorized = model.infer_vector(lyrics_clean)

    # Find the cosine similarities between the song and the songs in the database
    similar = model.dv.most_similar(positive = [lyrics_vectorized], topn= len(dataset))
    
    # Make a dataframe with the similarity scores and the song indices
    similar_df = pd.DataFrame(similar, columns = ['index', 'similarity_lyrics'])

    # Add the similarity scores to the song database
    data_sim = dataset.merge(similar_df, left_index = True, right_on = 'index')

    # return the song ID's and similarity scores
    return data_sim[['id','similarity_lyrics']]

  @st.cache(allow_output_mutation=True)
  def content_based_recommender(self):

    """
    This function returns song recommendations based on the audio features and lyrics of a song.
    This function inputs the name and the artist of the song, to get an dataframe with the audio features and lyrics of the song with the find_song function.
    This dataframe is then used to calculate the cosine-similarities between the song and the database with the recommender_features and recommender_lyrics functions.
    The recommendations are made by adding the similarity scores with a weighing constant alpha and sorting them. The songs with the n_songs highest similarities are given as recommendation.
    """

    # Get song dataframe to recommend on
    song = recommender.find_song(self)

    # 
    if self.do_kmeans:
      database = recommender.kmeans_cluster(self, song)
    else:
      database = self.database

    # get similarity scores for the features and lyrics
    features_sim = recommender.recommender_features(self, song, database)
    lyrics_sim = recommender.recommender_lyrics(self, song, database)

    # Drop the song we want to make recommendations on from the database
    features_sim = features_sim.drop(features_sim.index[features_sim['id'] == song['id'].values[0]])
    lyrics_sim = lyrics_sim.drop(lyrics_sim.index[lyrics_sim['id'] == song['id'].values[0]])

    # Merge both similarity scores to same dataframe
    features_sim = features_sim.merge(lyrics_sim, on = 'id')
    
    # Add the similarity scores weight with alpha 
    features_sim['similarity'] = (1 - self.alpha) * features_sim['similarity_features'] + self.alpha * features_sim['similarity_lyrics']

    # Sort on similarity score and keep the n_songs highest 
    recommendation = features_sim.sort_values(by='similarity',ascending=False).head(self.n_songs)
    
    # Add track name and artist to dataframe
    recommendation = recommendation.merge(self.lookup_table, on='id')

    # Add album image urls to dataframe
    recommendation_ids = list(recommendation['id'])
    img_urls = get_img_urls(recommendation_ids)
    recommendation['img_url'] = img_urls

    # Add play on Spotify link to dataframe
    recommendation['track_url'] = '[Play on Spotify](https://open.spotify.com/track/' + recommendation['id'] + ')'

    return song, recommendation
    


