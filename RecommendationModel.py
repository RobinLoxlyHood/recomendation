import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Membuat Class fungsi Model
class RecommenderNet(tf.keras.Model):
    # Insialisasi fungsi
    def __init__(self, num_users, num_places, embedding_size=16, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        num_users += 100
        self.num_users = num_users
        self.num_places = num_places
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding( # layer embedding user
            num_users,
            embedding_size,
            embeddings_initializer = 'he_normal',
            embeddings_regularizer = keras.regularizers.l2(1e-6)
        )
        self.user_bias = layers.Embedding(num_users, 1) # layer embedding user bias
        self.places_embedding = layers.Embedding( # layer embeddings places
            num_places,
            embedding_size,
            embeddings_initializer = 'he_normal',
            embeddings_regularizer = keras.regularizers.l2(1e-6)
        )
        self.places_bias = layers.Embedding(num_places, 1) # layer embedding places bias
 
    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:,0]) # memanggil layer embedding 1
        user_bias = self.user_bias(inputs[:, 0]) # memanggil layer embedding 2
        places_vector = self.places_embedding(inputs[:, 1]) # memanggil layer embedding 3
        places_bias = self.places_bias(inputs[:, 1]) # memanggil layer embedding 4
 
        dot_user_places = tf.tensordot(user_vector, places_vector, 2) 
 
        x = dot_user_places + user_bias + places_bias
    
        return tf.nn.sigmoid(x) # activation sigmoid
    
    def get_config(self):
        return {
            "num_users": self.num_users,
            "num_places": self.num_places,
            "embedding_size": self.embedding_size,
        }
    @classmethod
    def from_config(cls, config):
        return cls(**config)