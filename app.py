# -*- coding: utf-8 -*-
"""
Created on Wed Sept 26 02:20:31 2022
@authors: Supriya, Akash, Tejaswini, Rukmananda
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sept 26 02:20:31 2022
@authors: Supriya, Akash, Tejaswini, Rukmananda
"""


import numpy as np
import pickle
import pandas as pd
import streamlit as st 


from sklearn.base import BaseEstimator, TransformerMixin

class InputTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        print("initalized InputTransformer")

    def fit(self, X, y=None):
        print('fit')
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        print('trasform')
        transformedX = self.vectorizer.transform(X_)
        return transformedX

pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

def predict(text):
    prediction=classifier.predict([text])
    print(prediction)
    return prediction

def main():
    st.title("Fake News Classifier")
    text = st.text_area("Text", height=200)
    
    result=""
    if st.button("Predict"):
        result=predict(text)
        if int(result) == 1:
            st.error('The news is Fake')
        else:
            st.success('The news is Real')

    
    

if __name__=='__main__':
    main()