import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import plotly.express as px
import streamlit as st
from lyrics_extractor import SongLyrics
import nltk
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from gensim.models.doc2vec import TaggedDocument
import mysql.connector as mysql

@st.cache(allow_output_mutation=True)
def authenticate_spotify_api(SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET):
  """
  Function to authenticate the Spotify API with client credentials flow manager.
  
  :type SPOTIPY_CLIENT_ID: str
  :param SPOTIPY_CLIENT_ID:  public Spotify API key
  
  :type SPOTIPY_CLIENT_SECRET: str
  :param SPOTIPY_CLIENT_SECRET: private Spotify API key
  
  :type SPOTIPY_REDIRECT_URI: link
  :param SPOTIPY_REDIRECT_URI: Link to which Spotify API is set in Spotify Dashboard
  
  :return: Spotify API client
  
  """
  auth_manager = SpotifyClientCredentials(client_id = SPOTIPY_CLIENT_ID, 
                                          client_secret=SPOTIPY_CLIENT_SECRET)
    
  return spotipy.Spotify(auth_manager=auth_manager)

@st.cache(allow_output_mutation=True)
def authenticate_extract_lyrics(GCS_API_KEY, GCS_ENGINE_ID):
  """
  Function to initialize the lyrics_extractor class and to authenticate the google custom search engine.
  
  :type GCS_API_KEY: str
  :param GCS_API_KEY: Google cloud service API key
  
  :type GCS_ENGINE_ID: str
  :param GCS_ENGINE_ID: Google custom search engine ID
  
  :return: Songlyrics object
  """
  
  # Initialize lyrics_extractor class
  return SongLyrics(GCS_API_KEY, GCS_ENGINE_ID)

def search_multiple_tracks(search_query, sp):
  """
  Function to return the top 10 Spotify track id's, track name and artist given a search querry.

  Parameters
  ----------
  :type search_query: str
  :param search_quear: search query / search term
  
  :type sp: object
  :param sp: spotipy.Spotify object initialized and authenticated with authenticate_spotify_api()
  
  :return: (dict) Dictionary with track names, artists and id's
  """
  
  # List to store the track ids
  track_ids = []
  # List to store the track names and artists
  tracks = []

  #Search for 10 results in the Spotify API given a search querry
  results = sp.search(q = search_query ,limit=10)
  results = results['tracks']['items']

  # Extract the track id's, names and artists for all the search results
  for i in range(len(results)):

      # Get track id, artist and name
      track_id = results[i]['id']
      artist = results[i]['artists'][0]['name']
      track_name = results[i]['name']

      # Get a string with the artist and track name
      track = artist + ' - ' + track_name

      # Append the track id's and track name/artist to the list
      track_ids.append(track_id)
      tracks.append(track)

  # Make a dictionary of the track id and track name/artist list.
  return dict(zip(tracks,track_ids))

def get_img_urls(track_ids, sp):
  """
  Function to get the album image urls from tracks with the Spotify API, given a list of track id's.

  :type track_ids: list
  :param track_ids: Spotify track id's
  
  :type sp: object
  :param sp: spotipy.Spotify object initialized and authenticated with authenticate_spotify_api()
  
  :return: (href) URL's to album images from Spotify API
  """
  
  # Get a list with track information using a list of track id's
  tracks = sp.tracks(track_ids)

  # Initialize list to append image urls to
  img_urls = []

  for i in range(len(tracks['tracks'])):
    images = tracks['tracks'][i]['album']['images']

    seq = [x['height'] for x in images]
    img = next(item for item in images if item['height'] == min(seq))
    img_url = img['url']

    img_urls.append(img_url)

  return img_urls

def radar_chart(song, dataset):
  """
  Function to make a radar chart of the audio features of a song.

  :type song: DataFrame
  :param song: Pandas dataframe with the audio features of a song

  :type dataset: DataFrame
  :param dataset: Pandas dataframe with the database of songs we use to make recommendations. Used for normalizing the audio features (tempo and loudness) of the song.
  
  :return: plotly-express radar chart figure
  """
  # Reset the index of the song dataframe
  song = song.reset_index(drop = True)
  
  # Normalize the audio features of the song using the audio features of the database.
  song['tempo_norm'] = (song['tempo'] - dataset['tempo'].min())/(dataset['tempo'].max()- dataset['tempo'].min())
  song ['loudness_norm'] = 10 ** (song['loudness']/20)
  
  # Only keep the audio features of the song
  song = song[['danceability','energy','speechiness','acousticness','instrumentalness','liveness','valence','tempo_norm', 'loudness_norm']]

  song = song.rename({'tempo':'tempo normalized', 'loudness': 'loudness normalized'},axis=1)
  song = song.T
  song = song.reset_index()

  # Create radar chart
  fig = px.line_polar(song, r = 0, theta = 'index', line_close = True)
  fig.update_traces(fill = 'toself')

  return fig

@st.cache(allow_output_mutation=True)
def load_data_csv():
  """
  Create dataframes for the song data and the id lookup table from csv files.
  
  :return: (Pandas DataFrame) Pandas DataFrame of song data and lookup table
  """
  
  # Load lookup table
  path = 'data/id_lookup.csv'
  lookup_table = pd.read_csv(path, index_col=0)

  # Load song data
  path2 = 'data/data_lyrics_features.csv'
  data = pd.read_csv(path2, index_col=0)

  return data, lookup_table


@st.cache(allow_output_mutation=True, hash_funcs={"_thread.RLock": lambda _: None})
def load_data_sql():
  """
  Create dataframes for the song data and the id lookup table from sql tables
  
  :type conn: MySQL connection
  :param conn: connection to MySQL server
  
  :return: (Pandas DataFrame) Pandas DataFrame of song data and lookup table
  """ 
  conn = mysql.connect(**st.secrets["mysql"])

  data = pd.read_sql('SELECT * FROM song_data', conn)
  lookup_table = pd.read_sql('SELECT * FROM lookup_table', conn)
  
  return data, lookup_table

def clean_lyrics(data):
  """
  Function to clean the lyrics. It lowercases the lyrics, tokenizes it and removes all stopwords.

  :type data: list
  :param data:  list of strings of song lyrics
  
  :return: (list) list of tokenized and clean lyrics
  """
  
  #Initialize list to store clean data, tokenizer and the set of stopwords
  cleaned_data = []
  tokenizer = RegexpTokenizer(r'\w+')
  stopword_set = set(stopwords.words('english'))

  # Clean data for all the lyrics in the list
  for doc in data:
    # Get lowercase of lyrics string
    new_str = doc.lower()

    # Tokenize lyrics strings
    dlist = tokenizer.tokenize(new_str)

    # Remove stopwords
    dlist = list(set(dlist).difference(stopword_set))

    # Append cleaned lyrics to list
    cleaned_data.append(dlist)

  return cleaned_data

@st.cache()
def download_nltk():
  """ 
  Function to download the nltk stopwords, necessary for downloading them in deployed streamlit.
  """
  nltk.download('stopwords')
  return

def tag_lyrics(data):
  """
  Function to tag every document. Needed as input for the Doc2Vec network.

  :type data: list
  :param data: list of cleaned lyrics (output of the clean_lyrics function)
  
  :return: (list) list of tagged lyrics
  """

  # Initialize list to store tagged lyrics
  tagged_documents = []

  # Tag lyrics for all the lyrics in the list
  for i, doc in enumerate(data):

    # Tag lyrics
    tagged = TaggedDocument(doc, [i])

    # Append tagged lyrics to
    tagged_documents.append(tagged)

  return tagged_documents

def create_playlist(user_id, sp, recommendations, name, description):
  """ 
  Function to create a playlist on the Spotify account of the authenticated user.

  :type sp: object
  :param sp: spotipy.Spotify object initialized and authenticated with authenticate_spotify_api()
  
  :type recommendations: DataFrame
  :param recommendations: DataFrame of recommendations, output of recommender.content_based_recommender().
  
  :type name: str
  :param name: Name of the playlist.
  
  :type description: str
  :param description: Description of the playlist.
  
  :return: (str) succes/error message
  """
  
  # Get current user ID
  current_user = sp.current_user()
  current_user_id = current_user['id']
  
  # Get list of track ID's
  track_id_list = list(recommendations['id'].values)
  
  # Create Empty playlist
  sp.user_playlist_create(user = user_id, 
                          name = name, 
                          description = description)
  
  # Get playlist ID
  playlists = sp.current_user_playlists(limit=1)
  playlist_name = playlists['items'][0]['name']
  playlist_id = playlists['items'][0]['id']
  
  # Add tracks to playlist
  sp.user_playlist_add_tracks(user = current_user_id, 
                              playlist_id = playlist_id, 
                              tracks = track_id_list)
  
  # Check if playlist is succesfully created.
  if name == playlist_name:
    return '**Playlist was succesfully created on your Spotify account.**'
  else:
    return '**Playlist was not succesfully created.**'