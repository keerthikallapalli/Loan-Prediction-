import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and scaler
model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
data = pd.read_csv("loan_data.csv")

# --- Page config ---
st.set_page_config(page_title="Loan Prediction App", layout="wide")

# --- Styling ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #e3f2fd, #fce4ec);
    }
    .block-container {
        padding: 2rem;
        background-color: white;
        border-radius: 12px;
        max-width: 1000px;
        margin: auto;
        box-shadow: 0px 0px 15px rgba(0,0,0,0.15);
    }
    h1, h2, h3 {
        color: #004d40;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3064/3064197.png", width=80)
st.sidebar.title("Navigation")
view = st.sidebar.radio("Go to", ["Data Analysis", "Model Performance", "Prediction"])

# --- Main Content ---
with st.container():
    if view == "Prediction":
        st.markdown("<div class='block-container'>", unsafe_allow_html=True)
        st.header("🏦 Loan Eligibility Prediction")

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            applicant_income = st.number_input("Applicant Income", min_value=0)

        with col2:
            property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])
            married = st.selectbox("Married", ["Yes", "No"])
            coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
            loan_amount = st.number_input("Loan Amount", min_value=0)
            loan_amount_term = st.number_input("Loan Term (in days)", min_value=0)

        credit_history = st.selectbox("Credit History", ["1.0", "0.0"])

        if st.button("🔍 Predict Loan Status"):
            # Mapping to match training
            gender = 1 if gender == "Male" else 0
            married = 1 if married == "Yes" else 0
            education = 1 if education == "Graduate" else 0
            self_employed = 1 if self_employed == "Yes" else 0
            property_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}
            property_area = property_map[property_area]
            dependents = 3 if dependents == "3+" else int(dependents)
            credit_history = float(credit_history)

            input_data = np.array([[gender, married, dependents, education,
                                    self_employed, applicant_income, coapplicant_income,
                                    loan_amount, loan_amount_term, credit_history, property_area]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]

            if prediction == 1:
                st.success("✅ Loan Approved")
            else:
                st.error("❌ Loan Not Approved")

        st.markdown("</div>", unsafe_allow_html=True)

    elif view == "Data Analysis":
        st.markdown("<div class='block-container'>", unsafe_allow_html=True)
        st.header("📊 Data Analysis")

        st.subheader("Loan Status Count")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=data, x="Loan_Status", palette="Set2", ax=ax1)
        st.pyplot(fig1)

        st.subheader("Property Area vs Loan Status")
        fig2, ax2 = plt.subplots()
        sns.countplot(data=data, x="Property_Area", hue="Loan_Status", palette="Set1", ax=ax2)
        st.pyplot(fig2)

        st.subheader("Applicant Income Distribution")
        fig3, ax3 = plt.subplots()
        sns.histplot(data['ApplicantIncome'], kde=True, color="purple", ax=ax3)
        st.pyplot(fig3)

        st.markdown("</div>", unsafe_allow_html=True)

    elif view == "Model Performance":
        st.markdown("<div class='block-container'>", unsafe_allow_html=True)
        st.header("📈 Model Performance")

        from sklearn.metrics import classification_report, accuracy_score

        # Preprocess data
        test_data = data.copy()
        test_data['Loan_Status'] = test_data['Loan_Status'].map({'Y': 1, 'N': 0})
        X = test_data.drop(['Loan_Status', 'Loan_ID'], axis=1, errors='ignore')
        y = test_data['Loan_Status']
        X = pd.get_dummies(X)
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)

        acc = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True)

        st.metric("🔢 Accuracy", f"{acc*100:.2f}%")
        st.write("### Classification Report")
        st.dataframe(pd.DataFrame(report).transpose().style.background_gradient(cmap="YlGn"))

        st.markdown("</div>", unsafe_allow_html=True)
