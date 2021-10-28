import os

from keras.models import load_model

from keras.preprocessing.sequence import pad_sequences
import pickle
from prediction_reactions.service.preprocessing_data.prepearing_data import preprocessing,resources
import numpy as np



import pandas as pd
# settings_dir = os.path.dirname(__file__)
# settings_dir = os.path.abspath(os.path.dirname(settings_dir))

# project_root = os.path.abspath(os.path.dirname(settings_dir))
# resources =  "/resources/"
print(":::::lllllllll:::::",resources)



def len_sent(text):
    return len(text.split())



# In[16]:
# model_cnn_regression=load_model(os.path.join(resources+"models\model_cnn_regression.h5"))

# C:\Users\khaled.15\Desktop\project2\prediction_reactions\resources\models\model_cnn_regression.h5
model_cnn_regression=load_model(resources+"models/"+"model_cnn_regression.h5")

# model_cnn_regression=load_model("model_cnn_regression.h5")
# model_cnn_regression.save(resources+"models/"+"model_cnn_regression.pb")
model_lstm_regression=load_model(resources+"models/"+"model_lstm_regression.h5")
# model_lstm_regression.save(resources+"models/"+"model_lstm_regression.pb")

# model_lstm_regression=load_model(resources+"models/"+"model_lstm_regression.pb")

# In[11]:


tokenizer_model_cnn_lstm_regression=pickle.load(open(resources+"models/"+"tokenizer_model_cnn_lstm_regression.pickle","rb"))


# In[76]:


def get_vector_from_model_cnn_regression(sentence):
    # sentence=pd.DataFrame(sentence)[0].apply(preprocessing).values
    X_test=tokenizer_model_cnn_lstm_regression.texts_to_sequences(sentence)
    X_test = pad_sequences(X_test, padding='post', maxlen=100)
    # output=model_cnn_regression.layers[0](X_test)
    # output=model_cnn_regression.layers[1](output)
    # output=model_cnn_regression.layers[2](output)
    # output=model_cnn_regression.layers[3](output)
    return model_cnn_regression.predict(X_test)


# In[77]:


# r2=get_vector_from_model_cnn_regression(['امطار خفيفه تهطل العاصمه دمشق عدسه'])


# In[78]:


def get_vector_from_model_lstm_regression(sentence):
    # sentence=pd.DataFrame(sentence)[0].apply(preprocessing).values

    X_test=tokenizer_model_cnn_lstm_regression.texts_to_sequences(sentence)
    X_test = pad_sequences(X_test, padding='post', maxlen=100)
    # output=model_lstm_regression.layers[0](X_test)
    # output=model_lstm_regression.layers[1](output)
    # output=model_lstm_regression.layers[2](output)
    # output=model_lstm_regression.layers[3](output)
    return model_lstm_regression.predict(X_test)


# In[79]:


# r1=get_vector_from_model_lstm_regression(['امطار خفيفه  العاصمه دمشق عدسه',"امطار خفيفه تهطل العاصمه دمشق"])
#


def get_mean_lstm_cnn(sentenc):
    cnn=get_vector_from_model_cnn_regression(sentenc)
    lstm=get_vector_from_model_lstm_regression(sentenc)
    return np.mean([cnn,lstm],axis=0)


# In[81]:


get_mean_lstm_cnn(['امطار خفيفه تهطل العاصمه دمشق عدسه',"امطار خفيفه تهطل العاصمه دمشق"])
#
