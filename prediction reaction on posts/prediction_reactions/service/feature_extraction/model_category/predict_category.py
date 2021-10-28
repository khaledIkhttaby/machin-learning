from keras.models import load_model

import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd

from prediction_reactions.service.preprocessing_data.prepearing_data import resources, preprocessing


def len_sent(text):
    return len(text.split())


dictionary = {"__label__Tourism": "سياحة", "__label__Accidents_Crimes": "حوادث وجرائم", "__label__Art_Culture": "فن"
    ,"__label__Economy": "اقتصاد", "__label__Education": "تعليم", "__label__Health": "صحة",
               "__label__Military": "عسكري",
               "__label__Politics": "سياسي", "__label__Religion": "ديني",
               "__label__Science_Technology": "علوم تكنلوجيا", "__label__Social": "اجتماعي",
               "__label__Sport": "رياضة"

               }
model_category = load_model(resources + "models/" + "model_category.h5")
# model_category.save(resources + "models/" + "model_category.pb")

tokenizer_model_category = pickle.load(open(resources + "models/" + "tokenizer_model_category.pickle", "rb"))

encoder_model_category = pickle.load(open(resources + "models/" + "encoder_model_category_pickle", "rb"))


def get_vector_from_model_category(sentence):
    sentence = pd.DataFrame(sentence)[0].apply(preprocessing).values
    X_test = tokenizer_model_category.texts_to_sequences(sentence)
    X_test = pad_sequences(X_test, padding='post', maxlen=100)
    return model_category.predict(X_test)


def get_category(sentence):
    def convert_to_arabic(result):
        r=[]
        for item in result:
            r.append(dictionary[item])
        return r
    sentence = pd.DataFrame(sentence)[0].apply(preprocessing).values
    X_test = tokenizer_model_category.texts_to_sequences(sentence)
    X_test = pad_sequences(X_test, padding='post', maxlen=100)
    print(np.argmax(model_category.predict(X_test),axis=1), ":::::::::::::::")
    result=encoder_model_category.inverse_transform(np.argmax(model_category.predict(X_test),axis=1))
    result=convert_to_arabic(result)
    return result


# print(np.argmax(get_vector_from_model_category(['اصابات في صفوف المدنيين','اصابات في صفوف المدنيين']),axis=1))
print(get_category(['اصابات في صفوف المدنيين','اصابات في صفوف المدنيين']))
