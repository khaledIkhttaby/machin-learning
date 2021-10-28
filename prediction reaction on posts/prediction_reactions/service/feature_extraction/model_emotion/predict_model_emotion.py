import pickle
from keras.models import load_model

import numpy as np

from keras.preprocessing.sequence import pad_sequences

import pandas as pd

from prediction_reactions.service.preprocessing_data.prepearing_data import preprocessing,resources


def len_sent(text):
    return len(text.split())





model = load_model(resources+"models/"+"model_emotion.h5")

# model.save(resources+"models/"+"model_emotion.pb")
tokenizer_model_emotion = pickle.load(open(resources+"models/"+"tokenizer_model_emotion.pickle", "rb"))

# In[5]:


encoder_model_emotion = pickle.load(open(resources+"models/"+"encoder_model_emotion.pickle", "rb"))

# In[6]:


def get_vector_from_model_emotion(sentence):
    #     sentence=preprocessing(sentence)
    sentence = pd.DataFrame(sentence)[0].apply(preprocessing).values
    X_test = tokenizer_model_emotion.texts_to_sequences(sentence)
    X_test = pad_sequences(X_test, padding='post', maxlen=100)
    return model.predict(X_test)


# In[7]:


def get_emotion(sentence):
    sentence = pd.DataFrame(sentence)[0].apply(preprocessing).values

    X_test = tokenizer_model_emotion.texts_to_sequences(sentence)
    X_test = pad_sequences(X_test, padding='post', maxlen=100)
    return encoder_model_emotion.inverse_transform([np.argmax(model.predict(X_test))])


print(get_vector_from_model_emotion([" بكون جبل يهرب علي السجن", " بكون جبل يهرب علي السجن"]))
