import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px

st.set_page_config(page_title="Loan Prediction App", layout="wide")

# Set background and sidebar styling
def set_background():
    st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1556741533-f6acd6477c5b");
        background-size: cover;
        background-repeat: no-repeat;
    }
    .css-1d391kg {background-color: rgba(255, 255, 255, 0.8);}
    .stSidebar {
        background-image: url("https://images.unsplash.com/photo-1581090700227-1e8d50d2960d");
        background-size: cover;
    }
    </style>
    """, unsafe_allow_html=True)

set_background()

# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv("loan_data.csv")

@st.cache_resource
def load_model():
    return joblib.load("loan_model.pkl")

data = load_data()
model = load_model()

# Sidebar Navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Select View", ["Data Analysis", "Model Performance", "Prediction"])

# ---------------- Page 1: Data Analysis ----------------
if selection == "Data Analysis":
    st.title("📊 Data Analysis")

    st.subheader("Dataset Overview")
    st.dataframe(data.head())

    st.subheader("Loan Status Distribution")
    fig1 = px.histogram(data, x='Loan_Status', color='Loan_Status')
    st.plotly_chart(fig1)

    st.subheader("Property Area vs Loan Status")
    fig2 = px.histogram(data, x='Property_Area', color='Loan_Status', barmode='group')
    st.plotly_chart(fig2)

    st.subheader("Correlation Heatmap")
    corr_data = data.select_dtypes(include=np.number)
    fig3 = plt.figure(figsize=(8, 5))
    sns.heatmap(corr_data.corr(), annot=True, cmap='coolwarm')
    st.pyplot(fig3)

# ---------------- Page 2: Model Performance ----------------
elif selection == "Model Performance":
    st.title("📈 Model Performance")

    st.subheader("Sample Prediction Accuracy")

    # Clean and encode for training evaluation
    df = data.dropna()
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    st.metric("Accuracy", f"{acc:.2f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    st.write(cm)

# ---------------- Page 3: Prediction ----------------
elif selection == "Prediction":
    st.title("🔮 Loan Approval Prediction")

    col1, col2 = st.columns(2)

    with col1:
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Married = st.selectbox("Married", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
        Property_Area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

    with col2:
        ApplicantIncome = st.slider("Applicant Income", 0, 100000, 5000)
        CoapplicantIncome = st.slider("Coapplicant Income", 0, 50000, 2000)
        LoanAmount = st.slider("Loan Amount", 0, 700, 150)
        Loan_Amount_Term = st.slider("Loan Term (in days)", 0, 480, 360)
        Credit_History = st.selectbox("Credit History", [1.0, 0.0])

    submit = st.button("Predict")

    if submit:
        input_data = pd.DataFrame({
            'Gender': [Gender],
            'Married': [Married],
            'Dependents': [Dependents],
            'Education': [Education],
            'Self_Employed': [Self_Employed],
            'ApplicantIncome': [ApplicantIncome],
            'CoapplicantIncome': [CoapplicantIncome],
            'LoanAmount': [LoanAmount],
            'Loan_Amount_Term': [Loan_Amount_Term],
            'Credit_History': [Credit_History],
            'Property_Area': [Property_Area]
        })

        for col in input_data.select_dtypes(include='object').columns:
            input_data[col] = LabelEncoder().fit(data[col]).transform(input_data[col])

        prediction = model.predict(input_data)[0]
        result = "Approved ✅" if prediction == 1 else "Rejected ❌"

        st.success(f"Loan Prediction Result: **{result}**")
