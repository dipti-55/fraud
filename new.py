import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load and preprocess the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('fraud.csv')
    data = data.drop_duplicates()

    le_type = LabelEncoder()
    le_nameOrig = LabelEncoder()
    le_nameDest = LabelEncoder()

    data['type'] = le_type.fit_transform(data['type'])
    data['nameOrig'] = le_nameOrig.fit_transform(data['nameOrig'])
    data['nameDest'] = le_nameDest.fit_transform(data['nameDest'])

    X = data.drop(['isFraud', 'isFlaggedFraud'], axis=1)
    y = data['isFraud']

    return train_test_split(X, y, test_size=0.3, random_state=42), le_type, le_nameOrig, le_nameDest

# Train the model
@st.cache_data
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'fraud_detection_model.pkl')
    return model

# Load data and model
(X_train, X_test, y_train, y_test), le_type, le_nameOrig, le_nameDest = load_data()
model = train_model(X_train, y_train)

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

