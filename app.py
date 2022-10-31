from flask import Flask
import pandas as pd
import numpy as np
from function import *
from RecommendationModel import RecommenderNet
import warnings
warnings.filterwarnings('ignore')


app = Flask(__name__)
# @app.route("/")
# def home():
#     return "Hello, Flask!"

@app.route("/")
def recomendation(): 
    ranting_fe=get_DataRating().copy()
    rating=ranting_fe[['kode_user','kode_wisata','rate_value']]
    rating.columns = [['id_user','id_wisata', 'rating']]
    rating.to_csv('Dataset/rating_fe.csv', index=False)

    list_wisata=get_DataListWisata().copy()
    place=list_wisata[['id_wisata','wisata']]
    place.columns = ['id','place_name']
    place.to_csv('Dataset/list_wisata_db.csv', index=False)

    rating = pd.read_csv('Dataset/rating_fe.csv')
    place = pd.read_csv('Dataset/list_wisata_db.csv')
    df=rating.copy()

    training(df)
    
    return "Training Sukses"
