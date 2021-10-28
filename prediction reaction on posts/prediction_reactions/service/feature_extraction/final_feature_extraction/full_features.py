from prediction_reactions.service.feature_extraction.cnn_lstm_model.predict_cnn_lstm import get_mean_lstm_cnn
from prediction_reactions.service.feature_extraction.model_emotion.predict_model_emotion import \
    get_vector_from_model_emotion
from prediction_reactions.service.feature_extraction.model_category.lda_with_category_model import \
    get_vector_lda_category_mean
import numpy as np


def get_features_from_all_models(sentence):
    features1 = get_mean_lstm_cnn(sentence)
    features2 = get_vector_from_model_emotion(sentence)
    features3 = get_vector_lda_category_mean(sentence)
    return np.concatenate((features1, features2, features3), axis=1)

# In[33]:


print(get_features_from_all_models(["امطار خفيفه تهطل العاصمه دمشق عدس همطار خفيفه تهطل العاصمه دمشق عدسه"]),":::::done::::")


# featurs_full_data=data['contentclean'].apply(get_features_from_all_models)
