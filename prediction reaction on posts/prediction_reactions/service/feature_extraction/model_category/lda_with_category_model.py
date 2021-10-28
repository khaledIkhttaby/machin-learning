#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from prediction_reactions.service.feature_extraction.model_category.predict_category import get_vector_from_model_category


# In[2]:


from prediction_reactions.service.feature_extraction.model_category.predict_lda import get_vector_lda_model




def get_vector_lda_category_mean(sentence):
    r1=get_vector_lda_model(sentence)
    r2=get_vector_from_model_category(sentence)
    return np.mean([r1,r2],axis=0)





get_vector_lda_category_mean(['امطار خفيفه تهطل العاصمه دمشق عدسه'])





