import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from helper_functions import clean_lyrics, tag_lyrics

def train_model(data, vector_size = 300, dm = 1, min_count = 5, epochs = 60, window = 5, alpha = 0.025, min_alpha = 0.001):
    """ 
    Function to train the model.

    :type data: Pandas DataFrame
    :param data: Pandas DataFrame with one of the columns the lyrics of the tracks in the database.
    
    :type vector_size: int
    :param vector_size: Dimensionality of the feature vectors.
    
    :type dm: {1,0}
    :param dm: Defines the training algorithm. If dm=1, ‘distributed memory’ (PV-DM) is used. Otherwise, distributed bag of words (PV-DBOW) is employed.
    
    :type min_count: int
    :param min_count: Ignores all words with total frequency lower than this.
    
    :type epochs: int
    :param epochs: Number of iterations (epochs) over the corpus. Defaults to 10 for Doc2Vec.
    
    :type window: int
    :param window: The maximum distance between the current and predicted word within a sentence.
    
    :type alpha: float
    :param alpha: The initial learning rate.
    
    :type min_alpha: float
    :param min_alpha: Learning rate will linearly drop to min_alpha as training progresses.
    """
    
    # Get lyrics in right format
    lyrics_list = list(data['Lyrics'].values)
    lyrics_clean = clean_lyrics(lyrics_list)
    lyrics_tagged = tag_lyrics(lyrics_clean)

    # Initialize model
    model = Doc2Vec(vector_size = vector_size, 
                    dm = dm, 
                    min_count = min_count, 
                    epochs = epochs, 
                    window = window, 
                    alpha = alpha, 
                    min_alpha = min_alpha)

    # Build lyrics-embeddings
    model.build_vocab(lyrics_tagged)

    # Train the neural network to infer new lyrics-embeddings
    model.train(lyrics_tagged, total_examples = model.corpus_count, epochs = model.epochs)

    #Save the model
    model.save('model/doc2vec.model')

    return