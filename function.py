import mysql.connector
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from RecommendationModel import RecommenderNet


def get_DataRating():
    my_conn = mysql.connector.connect(host="127.0.0.1",
                                        port=3306,
                                        user='root',
                                        # password='f#Ur8J3N',
                                        database='tour_in')
    query_insert="""
    SELECT * FROM rating_fe;
    """
    raw_rating_fe = pd.read_sql_query(query_insert, my_conn)
    return raw_rating_fe



def get_DataListWisata():
    my_conn = mysql.connector.connect(host="127.0.0.1",
                                        port=3306,
                                        user='root',
                                        # password='f#Ur8J3N',
                                        database='tour_in')
    query_insert="""
    SELECT * FROM list_wisata;
    """
    raw_list_wisata = pd.read_sql_query(query_insert, my_conn)
    return raw_list_wisata


   
def training(df):  
    def dict_encoder(col, data=df):
        # Mengubah kolom suatu dataframe menjadi list tanpa nilai yang sama
        unique_val = data[col].unique().tolist()

        # Melakukan encoding value kolom suatu dataframe ke angka
        val_to_val_encoded = {x: i for i, x in enumerate(unique_val)}
    
        # Melakukan proses encoding angka ke value dari kolom suatu dataframe
        val_encoded_to_val = {i: x for i, x in enumerate(unique_val)}    
        return val_to_val_encoded, val_encoded_to_val
        #Encoding dan Mapping Kolom User
               
    # Encoding User_Id
    user_to_user_encoded, user_encoded_to_user = dict_encoder('id_user')

    # Mapping User_Id ke dataframe
    df['user'] = df['id_user'].map(user_to_user_encoded)

    # Encoding dan Mapping Kolom Place

    # Encoding Place_Id
    place_to_place_encoded, place_encoded_to_place = dict_encoder('id_wisata')

    # Mapping Place_Id ke dataframe place
    df['place'] = df['id_wisata'].map(place_to_place_encoded)

    #setting parameters for model
    num_users, num_place = len(user_to_user_encoded), len(place_to_place_encoded)
    
    # Mengubah rating menjadi nilai float
    df['rating'] = df['rating'].values.astype(np.float32)
    
    # Mendapatkan nilai minimum dan maksimum rating
    min_rating, max_rating = min(df['rating']), max(df['rating'])
    
    # Membuat variabel x untuk mencocokkan data user dan place menjadi satu value
    x = df[['user', 'place']].values
    
    # Membuat variabel y untuk membuat rating dari hasil 
    y = df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
    
    # Membagi menjadi 80% data train dan 20% data validasi
    train_indices = int(0.8 * df.shape[0])
    x_train, x_val, y_train, y_val = (
        x[:train_indices],
        x[train_indices:],
        y[:train_indices],
        y[train_indices:]
    )
    model = RecommenderNet(num_users, num_place) # inisialisasi model
    
    # model compile
    model.compile(
        loss = tf.keras.losses.BinaryCrossentropy(),
        optimizer = keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
            
    # Training Model
    history = model.fit(x = x_train,
                        y = y_train,
                        batch_size=16,
                        epochs = 50,
                        verbose=1,
                        validation_data = (x_val, y_val)
                       )

    #save model    
    return model.save_weights('model_checkpoint/model_weights', save_format='tf')

def testing(test):    
    #Encoding dan Mapping Kolom User

    # Encoding User_Id
    user_to_user_encoded, user_encoded_to_user = dict_encoder('id_user')

    # Mapping User_Id ke dataframe
    df['user'] = df['id_user'].map(user_to_user_encoded)

    # Encoding dan Mapping Kolom Place

    # Encoding Place_Id
    place_to_place_encoded, place_encoded_to_place = dict_encoder('id_wisata')

    # Mapping Place_Id ke dataframe place
    df['place'] = df['id_wisata'].map(place_to_place_encoded)

    #setting parameters for model
    num_users, num_place = len(user_to_user_encoded), len(place_to_place_encoded)



    #Loading model and saved weights
    loaded_model = RecommenderNet(num_users, num_place)
    loaded_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                         optimizer=keras.optimizers.Adam(learning_rate=0.001))
    loaded_model.load_weights('model_checkpoint/model_weights')


    # Inputan User yang akan dicari recomendasinya
    user_id = test.id_user.iloc[0]
    place_visited_by_user = df[df.id_user == user_id]

    # Membuat data lokasi yang belum dikunjungi user
    place_not_visited = place_df[~place_df['id'].isin(place_visited_by_user.id_wisata.values)]['id'] 
    place_not_visited = list(
        set(place_not_visited)
        .intersection(set(place_to_place_encoded.keys()))
    )
 
    place_not_visited = [[place_to_place_encoded.get(x)] for x in place_not_visited]
    user_encoder = user_to_user_encoded.get(user_id)
    user_place_array = np.hstack(
        ([[user_encoder]] * len(place_not_visited), place_not_visited)
    )

    # Mengambil top 7 recommendation
    ratings = loaded_model.predict(user_place_array).flatten()
    top_ratings_indices = ratings.argsort()[-5:][::-1]
    recommended_place_ids = [
        place_encoded_to_place.get(place_not_visited[x][0]) for x in top_ratings_indices
    ]

    #get the anime_id's, so we can make a request to anilist api for more info on show
    recommended_place = place_df[place_df['id'].isin(recommended_place_ids)]
    
    return recommended_place
    