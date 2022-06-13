import numpy as np
import pandas as pd
import streamlit as st
import pickle

# pickle_in = open("model.pkl", "rb")
pickle_in = open("D:/Jupyter/Diabetes_PredictionModel_Application_Streamlit_Code/model.pkl", "rb")
classifier = pickle.load(pickle_in)


def welcome():
    return "Welcome All"


def predict_note_authentication(glucose, bloodpressure, insulin, bmi, age):
    prediction = classifier.predict([[glucose, bloodpressure, insulin, bmi, age]])
    print(prediction)
    return prediction


# Factors in this model, Glucose, BloodPressure, Insulin, BMI, Age
def main():
    st.title("Diabetes Prediction Model")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Diabetes Prediction Machine Learning Model</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    glucose = st.text_input("Glucose", "Type the Glucose Levels Here")
    bloodpressure = st.text_input("BloodPressure", "Type the Blood Pressure Here")
    insulin = st.text_input("Insulin", "Type the Insulin levels Here")
    bmi = st.text_input("BMI", "Type the BMI Here")
    age = st.text_input("Age", "Type the Age Here")
    result = ""
    if st.button("Predict"):
        res = predict_note_authentication(glucose, bloodpressure, insulin, bmi, age)
        result = res[0]
    if result == 1:
        st.success('There is a high chance of being diabetic')
    else:
        st.success('There is a low chance of being diabetic')

    # st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")


if __name__ == '__main__':
    main()
