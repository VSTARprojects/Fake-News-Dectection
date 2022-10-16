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

pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

def predict(variance,skewness,curtosis,entropy):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
   
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return prediction

# change

def main():
    st.title("Fake News Classifier")
    # html_temp = """
    # <div style="background-color:tomato;padding:10px">
    # <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
    # </div>
    # """
    # st.markdown(html_temp,unsafe_allow_html=True)
    variance = st.text_input("Variance")
    skewness = st.text_input("skewness")
    curtosis = st.text_input("curtosis")
    entropy = st.text_input("entropy")
    result=""
    if st.button("Predict"):
        result=predict(variance,skewness,curtosis,entropy)
        if int(result) == 1:
            st.error('The news is Fake')
        else:
            st.success('The news is Real')

    
    

if __name__=='__main__':
    main()