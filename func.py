#this file contains the function that preprocesses the data
#and returns the features and target variables
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    # Drop rows with missing values (where the whole row is missing)
    df.dropna(inplace=True)

    # Replace empty artist names with 'Unknown Artist'
    df.loc[df['artist_name'] == 'empty_field', 'artist_name'] = 'Unknown Artist'

    # Calculate the median duration from valid entries
    median_duration = df[df['duration_ms'] > 0]['duration_ms'].median()
    # Replace -1 values with the median duration
    df.loc[df['duration_ms'] == -1, 'duration_ms'] = median_duration

    
    # Replace missing tempo values with the mean tempo
    df['tempo'] = pd.to_numeric(df['tempo'].replace('?', np.nan), errors='coerce')
    mean_tempo = df['tempo'].mean()                                                 #normally  this should have been commented out-but in the sent code it was not commented out
    df.loc[df['tempo'].isnull(), 'tempo'] = mean_tempo                              #normally  this should have been commented out
    

    # Fill missing tempo values based on music genre
    if 'music_genre' in df.columns:
        for genre in df['music_genre'].unique():
            genre_mode = df[df['music_genre'] == genre]['tempo'].mode()[0]
            df.loc[(df['tempo'].isnull()) & (df['music_genre'] == genre), 'tempo'] = genre_mode
            
    
    # Encode categorical features
    encoder = LabelEncoder()
    df['key_encoded'] = encoder.fit_transform(df['key'])
    df['mode_encoded'] = encoder.fit_transform(df['mode'])
    df['music_genre_encoded'] = encoder.fit_transform(df['music_genre'])

    # Drop the original categorical columns
    df = df.drop(columns=['key', 'mode', 'music_genre'])

    # Drop irrelevant columns
    df = df.drop(columns=['instance_id', 'artist_name', 'track_name', 'obtained_date'])

    # Drop key_encoded and mode_encoded columns
    df = df.drop(columns=['mode_encoded', 'key_encoded'])

    y = df['music_genre_encoded']
    X = df.drop(columns=['music_genre_encoded'])

    return X, y
