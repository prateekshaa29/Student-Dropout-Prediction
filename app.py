import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Student Dropout Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

.main {
    background-color: #F4F7FC;
}

.block-container{
    padding-top:2rem;
}

h1{
    color:#0F4C81;
    text-align:center;
}

h3{
    color:#1565C0;
}

div[data-testid="stMetric"]{
    background:white;
    border-radius:15px;
    padding:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.15);
    text-align:center;
}

.stButton>button{
    background:#0F4C81;
    color:white;
    width:100%;
    height:55px;
    font-size:20px;
    border-radius:10px;
}

.stButton>button:hover{
    background:#1565C0;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #

model = joblib.load("models/student_dropout_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("📊 Project Information")

st.sidebar.markdown("---")

st.sidebar.subheader("Project Type")
st.sidebar.success("Data Science Project")

st.sidebar.subheader("Algorithm")
st.sidebar.info("Random Forest Classifier")

st.sidebar.subheader("Model Accuracy")
st.sidebar.metric("Accuracy", "74.01%")

st.sidebar.subheader("Dataset")
st.sidebar.write("4424 Student Records")

st.sidebar.subheader("Input Features")
st.sidebar.write("11 Selected Features")

st.sidebar.markdown("---")

st.sidebar.subheader("Technologies")

st.sidebar.write("""
• Python

• Pandas

• NumPy

• Scikit-learn

• Streamlit
""")

# ---------------- TITLE ---------------- #

st.title("🎓 Student Dropout Prediction System")

st.markdown("### A Data Science Project for Student Performance Analysis")

st.markdown("---")

# ---------------- DASHBOARD ---------------- #

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", "74.01%")

with col2:
    st.metric("Algorithm", "Random Forest")

with col3:
    st.metric("Features", "11")

with col4:
    st.metric("Dataset", "4424")

st.markdown("---")

# ---------------- INPUTS ---------------- #

col_left, col_right = st.columns(2)

with col_left:

    st.subheader("👤 Personal Information")

    age = st.number_input(
        "Age at Enrollment",
        min_value=15,
        max_value=70,
        value=20
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    st.subheader("💰 Financial Information")

    scholarship = st.selectbox(
        "Scholarship Holder",
        ["No", "Yes"]
    )

    debtor = st.selectbox(
        "Debtor",
        ["No", "Yes"]
    )

    fees = st.selectbox(
        "Tuition Fees Up To Date",
        ["Yes", "No"]
    )

with col_right:

    st.subheader("🎓 Academic Information")

    admission_grade = st.number_input(
        "Admission Grade",
        value=130.0
    )

    previous_grade = st.number_input(
        "Previous Qualification Grade",
        value=130.0
    )

    approved1 = st.number_input(
        "1st Semester Approved Units",
        min_value=0,
        value=6
    )

    approved2 = st.number_input(
        "2nd Semester Approved Units",
        min_value=0,
        value=6
    )

    grade1 = st.number_input(
        "1st Semester Grade",
        value=12.0
    )

    grade2 = st.number_input(
        "2nd Semester Grade",
        value=12.0
    )

# ---------------- CONVERT VALUES ---------------- #

gender = 1 if gender == "Male" else 0
scholarship = 1 if scholarship == "Yes" else 0
debtor = 1 if debtor == "Yes" else 0
fees = 1 if fees == "Yes" else 0

# ---------------- PREDICTION ---------------- #

st.markdown("---")

if st.button("🔍 Predict Student Outcome"):

    data = pd.DataFrame([{
        "Age at enrollment": age,
        "Admission grade": admission_grade,
        "Previous qualification (grade)": previous_grade,
        "Gender": gender,
        "Scholarship holder": scholarship,
        "Debtor": debtor,
        "Tuition fees up to date": fees,
        "Curricular units 1st sem (approved)": approved1,
        "Curricular units 2nd sem (approved)": approved2,
        "Curricular units 1st sem (grade)": grade1,
        "Curricular units 2nd sem (grade)": grade2
    }])

    prediction = model.predict(data)
    probabilities = model.predict_proba(data)[0]

    result = label_encoder.inverse_transform(prediction)[0]

    st.markdown("## 🎯 Prediction Result")

    if result == "Graduate":
        st.success("🎓 **Graduate**")
        st.write("The student is likely to successfully complete the course.")

    elif result == "Dropout":
        st.error("⚠ **Dropout**")
        st.write("The student may require additional academic support.")

    else:
        st.info("📘 **Enrolled**")
        st.write("The student is currently progressing through the course.")

    st.markdown("---")

    st.subheader("📈 Prediction Probability")

    probability_df = pd.DataFrame({
        "Class": label_encoder.classes_,
        "Probability": probabilities
    })

    st.bar_chart(
        probability_df.set_index("Class")
    )

# ---------------- FOOTER ---------------- #

st.markdown("---")

st.caption(
    "📊 Developed using Python • Pandas • NumPy • Scikit-learn • Streamlit"
)