import streamlit as st
from helper_functions import search_multiple_tracks, radar_chart, load_data_csv, load_data_sql, authenticate_spotify_api, create_playlist, authenticate_extract_lyrics
from recommender import recommender
import mysql.connector as mysql

def main():
    
    # Set the streamlit page configuration
    st.set_page_config(layout="wide", initial_sidebar_state='collapsed')
    
    #Authenticate Spotify API
    sp = authenticate_spotify_api(SPOTIPY_CLIENT_ID = st.secrets["SPOTIPY_CLIENT_ID"], 
                                  SPOTIPY_CLIENT_SECRET = st.secrets["SPOTIPY_CLIENT_SECRET"],
                                  SPOTIPY_REDIRECT_URI = st.secrets["SPOTIPY_REDIRECT_URI"])
    
    extract_lyrics = authenticate_extract_lyrics(GCS_API_KEY = st.secrets["GCS_API_KEY"],
                                                 GCS_ENGINE_ID = st.secrets["GCS_ENGINE_ID"])

    # Load the song data and id lookup_table from sql server. If connection to sql-server can't be made, load from csv file.
    try:
        data, lookup_table = load_data_sql()
    except:
        data, lookup_table = load_data_csv()
    
    # Title
    st.title("Spotify Recommendation system")

    # Text input for search query
    search_input = st.text_input('Search Song/Artist')
    
    # Check if a search query is given, otherwise ask for search query.
    if not search_input:
        st.write('**Please enter a search term.**')
    else:
        # Search the track information
        track_dict = search_multiple_tracks(search_input, sp)
        
        # Check if search query gives search results, otherwise ask for different search query.
        if not track_dict:
            st.write('**No results found, please change your search term.**')
        else:
            # Select a track of the search results
            choose_track = st.selectbox('Select Track', options = track_dict)

            # Get the track id of the selected track
            track_id = track_dict[choose_track]
            
            # Define 2 columns for n_songs and alpha
            cols = st.beta_columns(2)
            
            # Number input for the number of songs that need to be recommended, limit number of recommendations between 1 and 50 (Otherwise Spotify API doesn't work to get img_urls).
            n_songs = int(cols[0].number_input('Number of recommendations (max 50)',value=20, max_value=50, min_value=1))
            
            # Slider to determine the percentage that the lyrics recommender needs to be part of the total recommendation
            alpha = cols[1].slider(label = 'How much do the lyrics need to part of the recommendation? (in percentage)', min_value=0, max_value=100, value = 10)
            
            # Create recommend object from recommender class.
            recommend = recommender(track_id = track_id, 
                                    database = data, 
                                    lookup_table = lookup_table, 
                                    n_songs = n_songs,
                                    alpha = alpha/100,
                                    sp = sp,
                                    extract_lyrics = extract_lyrics)

            # Call content_based_recommender method
            song, recommendations = recommend.content_based_recommender()

            # Make radar chart of the song we want to make recommendations for
            fig = radar_chart(song, data)

            # Create expander for the track analysis
            track_analysis = st.beta_expander(label = 'Show track audio features')
            with track_analysis:
                
                # Plot radar chart
                st.plotly_chart(fig)

                # horizontal line
                st.markdown("""---""")
                
                # Print the audio features of the song in 3 columns
                col = st.beta_columns(3)
                col[0].write('Acousticness: ' + str(song['acousticness'].values[0]))
                col[0].write('Danceability: ' + str(song['danceability'].values[0]))
                col[0].write('Energy: ' + str(song['energy'].values[0]))
                col[1].write('Instrumentalness: ' + str(song['instrumentalness'].values[0]))
                col[1].write('Livenss: ' + str(song['liveness'].values[0]))
                col[1].write('Loudness: ' + str(song['loudness'].values[0]) + 'dB')
                col[2].write('Speechiness: ' + str(song['speechiness'].values[0]))
                col[2].write('Tempo: ' + str(int(song['tempo'].values[0])) + 'BPM')
                col[2].write('Valence: ' + str(song['valence'].values[0]))
                
                # horizontal line
                st.markdown("""---""")

                # Explanation of the audio features
                st.markdown('**Acousticness**')
                st.write('A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.')
                st.markdown('**Danceability**')
                st.write('Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.')
                st.markdown('**Energy**')
                st.write('Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.')
                st.markdown('**Instrumentalness**')
                st.write('Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.')
                st.markdown('**Liveness**')
                st.write('Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.')
                st.markdown('**Loudness**')
                st.write('The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db. Here normalized to a value between 0 and 1.')
                st.markdown('**Speechiness**')
                st.write('Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.')
                st.markdown('**Tempo**')
                st.write('The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration. Here normalized to a value between 0 and 1.')
                st.markdown('**Valence**')
                st.write('A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).')

            # Create Expander for playlist creation.
            create_playlist_expander = st.beta_expander(label = 'Create playlist of recommendations', expanded=True)
            with create_playlist_expander:
                
                # Input playlist_name
                playlist_name = st.text_input('Playlist Name', value = choose_track + ' Recommendations')
                
                # Playlist description
                description = 'Recommendations based on the audio features and lyrics of ' + choose_track + '.'
                
                # Button that calls create_playlist to create playlist.
                if st.button('Create playlist'):
                    # Check if a playlist name is given, otherwise ask for playlist name.
                    if not playlist_name:
                        st.write('**Please input playlist name.**')
                    else:
                        message = create_playlist(sp = sp,
                                        recommendations = recommendations, 
                                        name = playlist_name, 
                                        description = description)
                        st.write(message)
            
            # Create header
            st.header('Recommendations')
            
            # Whitespace
            st.markdown(' ')
            
            # Print the column headers in the correct columns
            cols = st.beta_columns(6)
            cols[1].markdown('**Track name**')
            cols[2].markdown('**Artist**')
            cols[3].markdown('**Similarity score**')

            #Whitespace
            st.markdown(' ')

            # Initialize list for the show lyrics buttons of 
            lyrics_button = []

            # Print the recommendations
            for index, recommendation in recommendations.iterrows():
                cols = st.beta_columns(6)
                
                # Print album image, song name, artist name and similarty score
                cols[0].image(recommendation['img_url'])
                cols[1].markdown(recommendation['Song'])
                cols[2].markdown(recommendation['Band'])
                cols[3].markdown(round(recommendation['similarity'],4))
                
                # Make show lyrics button for each song
                button = cols[4].checkbox('Show Lyrics', key= str(index))
                
                # Append buttons to list
                lyrics_button.append(button)
                
                # Link to play song on Spotify
                cols[5].markdown(recommendation['track_url'], unsafe_allow_html = True)
                
                # Print lyrics if the show lyrics button is clicked (button value is True)
                if lyrics_button[index]:
                    st.markdown('**Lyrics:**')
                    st.write(data[data['id'] == recommendation['id']]['Lyrics'].values[0])
                    st.write(' ')

if __name__ == '__main__':
    main()