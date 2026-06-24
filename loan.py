import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

st.title("🏦 Loan Eligibility Approval Predictor")

file = st.file_uploader("Upload Loan Dataset", type=["csv"])

if file:

    df = pd.read_csv(file)

    le = LabelEncoder()

    for col in ['Gender','Married','Education','Loan_Status']:
        df[col] = le.fit_transform(df[col])

    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    model = DecisionTreeClassifier()
    model.fit(X, y)

    st.header("Customer Details")

    gender = st.selectbox("Gender", ["Male","Female"])
    married = st.selectbox("Married", ["Yes","No"])
    income = st.number_input("Income")
    loan = st.number_input("Loan Amount")
    credit = st.selectbox("Credit History",[0,1])
    education = st.selectbox("Education",
                             ["Graduate","Not Graduate"])

    gender = 1 if gender=="Male" else 0
    married = 1 if married=="Yes" else 0
    education = 0 if education=="Graduate" else 1

    data = [[gender,
             married,
             income,
             loan,
             credit,
             education]]

    if st.button("Predict Loan Status"):

        prediction = model.predict(data)

        if prediction[0] == 0:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")