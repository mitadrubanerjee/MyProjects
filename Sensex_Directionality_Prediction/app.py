import numpy as np
import pandas as pd
import streamlit as st
import pickle

# pickle_in = open("model.pkl", "rb")
pickle_in = open("D:/Jupyter/Diabetes_PredictionModel_Application_Streamlit_Code/model.pkl", "rb")
classifier = pickle.load(pickle_in)


def welcome():
    return "Welcome All"


# I am running the classification model using Open, High, Low, Close
# The idea is to enter these values which will give us an indication of how the market will move
def predict_note_authentication(open_price, high, low, close):
    prediction = classifier.predict([[open_price, high, low, close]])
    print(prediction)
    return prediction


# Factors in this model, Glucose, BloodPressure, Insulin, BMI, Age
def main():
    st.title("Sensex Prediction Model")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Sensex Prediction Machine Learning Model</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    open_price = st.text_input("Open_price", "Type the Open Price Here")
    high = st.text_input("High Price", "Type the High Price Here")
    low = st.text_input("Low Price", "Type the Low Price Here")
    close = st.text_input("Close Price", "Type the Close Price Here")

    result = ""
    if st.button("Predict"):
        res = predict_note_authentication(open_price, high, low, close)
        result = res[0]
    if result == 1:
        st.success('There is a chance that the Stock Market will go up')
    else:
        st.success('There is a chance that the Stock Market will go down')

    # st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")


if __name__ == '__main__':
    main()
