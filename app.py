import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn. preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

model=tf.keras.models.load_model('model.h5')

with open('ohe.pkl','rb') as file:
    ohe=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

with open('le.pkl','rb') as file:
    le=pickle.load(file)

## Streamlit Code
st.title('Churn Prediction')
col1 , col2=st.columns(2)
with col1:
    geography=st.selectbox('Gepgraphy',ohe.categories_[0])
    gender=st.selectbox("Gender",le.classes_)
    estimated_salary=st.number_input('Estimated Salary')
    balence=st.number_input('Balence')
    is_active=st.selectbox('Is Active',[1,0])

with col2:
    credit_score=st.slider('Credit Score',0,700)
    tenure=st.slider("Tensure",0,10)
    age = st.slider('Age', 18, 92)
    num_of_products=st.slider('NO. Of Products',0,4)
    has_cr=st.selectbox("Hold Credit Card ",[1,0])

if st.button(label="Predict Churn", help="Click to analyze customer data", type="primary"):
    input_data=pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [le.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balence],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr],
        'IsActiveMember': [is_active],
        'EstimatedSalary': [estimated_salary]
    })


    geo_encoded = ohe.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded,columns=ohe.get_feature_names_out())

    input_data=pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    scaled_input=scaler.transform(input_data)

    with st.spinner("Analyzing data..."):
            pred=model.predict(scaled_input)
            prediction=pred[0][0]
        
    st.subheader("Prediction Result")
    if prediction > 0.5 :
            st.write("Customer Is Likely To Churn")
    else:
            st.write("Customer Not Likely To Churn")


