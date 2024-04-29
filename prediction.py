import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the machine learning model and encode
model = joblib.load('best_model.pkl')
gender_encode= joblib.load('gender_encode.pkl')
label_encode_geo=joblib.load('label_encode_geo.pkl')

def main():
    st.title('Churn Model Deployment')

    #Add user input components
    #input one by one
    CreditScore=st.number_input("CreditScore", 100, 1000)
    Gender=st.radio("gender", ["Male","Female"])
    Geography=st.radio("geographical location", ["France","Spain", "Germany", "Others"])
    Age=st.number_input("age", 0, 100)
    Tenure=st.number_input("tenure", 0,10)
    Balance=st.number_input("balance", 0,1000000)
    NumOfProducts=st.number_input("number of products", 1,5)
    HasCrCard=st.number_input("have credit card? [0 for no, 1 for yes]", 0,1)
    IsActiveMember=st.number_input("an active member? [0 for no, 1 for yes]", 0,1)
    EstimatedSalary=st.number_input("salary", 0,1000000)
    
    data = {'CreditScore': float(CreditScore), 'Gender': Gender, 'Geography': Geography,
            'Age': int(Age), 'Tenure':int(Tenure), 'Balance': float(Balance), 
            'NumOfProducts': int(NumOfProducts), 'HasCrCard': int(HasCrCard),
            'IsActiveMember': int(IsActiveMember), 'EstimatedSalary': float(EstimatedSalary)}
    
    df=pd.DataFrame([list(data.values())], columns=['CreditScore', 'Gender', 'Geography',
                                                    'Age', 'Tenure', 'Balance', 'NumOfProducts',
                                                    'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])

    df=df.replace(gender_encode)
    df=df.replace(label_encode_geo)
    
    if st.button('Make Prediction'):
        features=df      
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()