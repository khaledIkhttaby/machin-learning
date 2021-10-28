
import pickle
from prediction_reactions.service.preprocessing_data.prepearing_data import resources




cv=pickle.load(open(resources+"models/"+"count_vectorize.pickle","rb"))





lda_model=pickle.load(open(resources+"models/"+"lda_model.pickle","rb"))





def get_vector_lda_model(sentence):
    X_test=cv.transform(sentence)
    return lda_model.transform(X_test)




#c=get_vector_lda_model(['امطار خفيفه تهطل العاصمه دمشق عدسه'])






