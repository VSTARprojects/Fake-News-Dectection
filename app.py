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


from urllib import response
import numpy as np
import pickle
import pandas as pd
import streamlit as st 
import requests

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

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

pickle_in = open("classifier_final.pkl","rb")
classifier=pickle.load(pickle_in)

from urllib.request import urlopen
pickle_in_2 = urlopen("https://github.com/2711-bharath/test-upload-large-files-to-gitHub/blob/main/sentence_transformer_model.pkl?raw=true")
# pickle_in_2 = open("sentence_transformer_model.pkl","rb")
sentence_transformer = pickle.load(pickle_in_2)

def predict(text):
    prediction=classifier.predict([text])
    print(prediction)
    return prediction

def encode(sentences):
    prediction=sentence_transformer.encode(sentences)
    print(prediction)
    return prediction

def main():
    st.title("Fake News Classifier")
    text = st.text_area("Text", height=200)
    
    result=""
    if st.button("Predict"):
        result=predict(text)
        temp = requests.get('https://factchecktools.googleapis.com/v1alpha1/claims:search?key=AIzaSyBRuXbKLveIy-0H5LkCaoDLlm8BxFcs44Q&query=' + text)
        temp = temp.json()
        if len(temp) > 0:
            claims_list = []
            for x in temp['claims']:
                sentences = [
                    x['text'],
                    text
                ]
                sentence_embeddings = encode(sentences)
                t = cosine_similarity(
                    [sentence_embeddings[0]],
                    sentence_embeddings[1:]
                )
                print(t)
                if t > 0.44:
                    label_dict = defaultdict(lambda: -1)
                    label_dict['Half True'] = 0
                    label_dict['Mostly True'] = 0
                    label_dict['False'] = 1
                    label_dict['True'] = 0
                    label_dict['Barely True'] = 1
                    label_dict['Pants on Fire'] = 1
                    label_dict['Distorts the Facts'] = 1
                    label_dict['Partly False'] = 1
                    claims_list.append({'claimText': x['text'], 'rating': label_dict[x['claimReview'][0]['textualRating']], 'textualRating': x['claimReview'][0]['textualRating']})
            if len(claims_list) > 0:
                st.subheader('Based on Known Facts')
                for claim in claims_list:
                    result = claim['rating']
                    if int(result) == 1:
                        st.error('"' + claim['claimText'] + '" ' + ' - Fake')
                    elif int(result) == 0:
                        st.success('"' + claim['claimText'] + '" ' + ' ' + ' - Real')
                    else:
                        st.info(claim['claimText'] + '  - ' + claim['textualRating'])   
        st.subheader('Based on the words used')  
        if int(result) == 1:
            st.error('Our model predicts: The news could be Fake')
        else:
            st.success('Our model predicts: The news could be Real')




    

if __name__=='__main__':
    main()
