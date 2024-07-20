import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load('fraud_detection_model.pkl')

# Create a LabelEncoder object
le_type = LabelEncoder()
le_nameOrig = LabelEncoder()
le_nameDest = LabelEncoder()

# Load the dataset to fit the LabelEncoders
data = pd.read_csv('your_dataset.csv')
le_type.fit(data['type'])
le_nameOrig.fit(data['nameOrig'])
le_nameDest.fit(data['nameDest'])

# Title and description
st.title('Fraud Detection Prediction')
st.write('Enter the transaction details to predict if it is fraudulent.')

# Input fields
step = st.number_input('Step', min_value=1)
type_input = st.selectbox('Type', le_type.classes_)
amount = st.number_input('Amount', min_value=0.0)
nameOrig_input = st.selectbox('NameOrig', le_nameOrig.classes_)
oldbalanceOrg = st.number_input('Old Balance Org', min_value=0.0)
newbalanceOrig = st.number_input('New Balance Orig', min_value=0.0)
nameDest_input = st.selectbox('NameDest', le_nameDest.classes_)
oldbalanceDest = st.number_input('Old Balance Dest', min_value=0.0)
newbalanceDest = st.number_input('New Balance Dest', min_value=0.0)

# Button to make predictions
if st.button('Predict'):
    # Encode the categorical features
    type_encoded = le_type.transform([type_input])[0]
    nameOrig_encoded = le_nameOrig.transform([nameOrig_input])[0]
    nameDest_encoded = le_nameDest.transform([nameDest_input])[0]
    
    # Create a DataFrame for the input features
    input_data = pd.DataFrame(
        [[step, type_encoded, amount, nameOrig_encoded, oldbalanceOrg, newbalanceOrig, nameDest_encoded, oldbalanceDest, newbalanceDest]],
        columns=['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest']
    )
    
    # Make a prediction
    prediction = model.predict(input_data)[0]
    
    # Display the result
    if prediction == 1:
        st.error('The transaction is predicted to be fraudulent.')
    else:
        st.success('The transaction is predicted to be not fraudulent.')

