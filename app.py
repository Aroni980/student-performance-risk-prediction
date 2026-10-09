

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Student Risk Predictor",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("models/student_risk_model.pkl")

model = load_model()

st.title("🎓 Student Performance Risk Predictor")
st.write(
    "Enter student information to estimate potential academic risk "
    "based on the UCI Student Performance dataset."
)

st.info(
    "Educational demonstration only. Predictions are estimates, "
    "not definitive judgments about a student."
)

st.caption(
    "Gender is not used by this model. The model was trained on "
    "historical student records and may not generalize to every student."
)

with st.form("student_form"):
    st.subheader("Student Information")

    col1, col2 = st.columns(2)

    with col1:
        school = st.selectbox(
            "School",
            ["GP", "MS"],
            format_func=lambda x: {
                "GP": "Gabriel Pereira",
                "MS": "Mousinho da Silveira"
            }[x]
        )

        gender = st.selectbox(
            "Gender (not used for prediction)",
            [
                "Female",
                "Male",
                "Non-binary",
                "Another identity",
                "Prefer not to say"
            ]
        )

        age = st.slider("Age", 15, 22, 17)

        address = st.selectbox(
            "Area of residence",
            ["U", "R"],
            format_func=lambda x: {
                "U": "Urban",
                "R": "Rural"
            }[x]
        )

        famsize = st.selectbox(
            "Family size",
            ["LE3", "GT3"],
            format_func=lambda x: {
                "LE3": "3 or fewer people",
                "GT3": "More than 3 people"
            }[x]
        )

        Pstatus = st.selectbox(
            "Parents' living arrangement",
            ["T", "A"],
            format_func=lambda x: {
                "T": "Living together",
                "A": "Living apart"
            }[x]
        )

        Medu = st.select_slider(
            "Mother's education level",
            options=[0, 1, 2, 3, 4],
            value=2,
            format_func=lambda x: {
                0: "0 - None",
                1: "1 - Primary",
                2: "2 - Lower secondary",
                3: "3 - Upper secondary",
                4: "4 - Higher education"
            }[x]
        )

        Fedu = st.select_slider(
            "Father's education level",
            options=[0, 1, 2, 3, 4],
            value=2,
            format_func=lambda x: {
                0: "0 - None",
                1: "1 - Primary",
                2: "2 - Lower secondary",
                3: "3 - Upper secondary",
                4: "4 - Higher education"
            }[x]
        )

        Mjob = st.selectbox(
            "Mother's job",
            ["teacher", "health", "services", "at_home", "other"],
            format_func=lambda x: {
                "teacher": "Teacher",
                "health": "Healthcare",
                "services": "Services",
                "at_home": "At home",
                "other": "Other"
            }[x]
        )

        Fjob = st.selectbox(
            "Father's job",
            ["teacher", "health", "services", "at_home", "other"],
            format_func=lambda x: {
                "teacher": "Teacher",
                "health": "Healthcare",
                "services": "Services",
                "at_home": "At home",
                "other": "Other"
            }[x]
        )

        reason = st.selectbox(
            "Reason for choosing this school",
            ["home", "reputation", "course", "other"],
            format_func=lambda x: {
                "home": "Close to home",
                "reputation": "School reputation",
                "course": "Course preference",
                "other": "Other reason"
            }[x]
        )

        guardian = st.selectbox(
            "Main guardian",
            ["mother", "father", "other"],
            format_func=lambda x: {
                "mother": "Mother",
                "father": "Father",
                "other": "Other"
            }[x]
        )

        traveltime = st.select_slider(
            "Travel time to school",
            options=[1, 2, 3, 4],
            value=1,
            format_func=lambda x: {
                1: "1 - Under 15 minutes",
                2: "2 - 15 to 30 minutes",
                3: "3 - 30 minutes to 1 hour",
                4: "4 - Over 1 hour"
            }[x]
        )

        studytime = st.select_slider(
            "Weekly study time",
            options=[1, 2, 3, 4],
            value=2,
            format_func=lambda x: {
                1: "1 - Under 2 hours",
                2: "2 - 2 to 5 hours",
                3: "3 - 5 to 10 hours",
                4: "4 - Over 10 hours"
            }[x]
        )

        failures = st.selectbox(
            "Past class failures",
            [0, 1, 2, 3, 4],
            format_func=lambda x: "4 or more" if x == 4 else str(x)
        )

        schoolsup = st.selectbox(
            "Extra educational support from school",
            ["no", "yes"],
            format_func=lambda x: "No" if x == "no" else "Yes"
        )

        famsup = st.selectbox(
            "Educational support from family",
            ["no", "yes"],
            format_func=lambda x: "No" if x == "no" else "Yes"
        )

        paid = st.selectbox(
            "Extra paid classes",
            ["no", "yes"],
            format_func=lambda x: "No" if x == "no" else "Yes"
        )

    with col2:
        activities = st.selectbox(
            "Extracurricular activities",
            ["yes", "no"],
            format_func=lambda x: "Yes" if x == "yes" else "No"
        )

        nursery = st.selectbox(
            "Attended nursery school",
            ["yes", "no"],
            format_func=lambda x: "Yes" if x == "yes" else "No"
        )

        higher = st.selectbox(
            "Wants to pursue higher education",
            ["yes", "no"],
            format_func=lambda x: "Yes" if x == "yes" else "No"
        )

        internet = st.selectbox(
            "Internet access at home",
            ["yes", "no"],
            format_func=lambda x: "Yes" if x == "yes" else "No"
        )

        romantic = st.selectbox(
            "Currently in a romantic relationship",
            ["yes", "no"],
            format_func=lambda x: "Yes" if x == "yes" else "No"
        )

        famrel = st.slider("Family relationship quality (1–5)", 1, 5, 4)
        freetime = st.slider("Free time after school (1–5)", 1, 5, 3)
        goout = st.slider("Going out with friends (1–5)", 1, 5, 3)
        Dalc = st.slider("Workday alcohol consumption scale (1–5)", 1, 5, 1)
        Walc = st.slider("Weekend alcohol consumption scale (1–5)", 1, 5, 1)
        health = st.slider("Self-reported health (1–5)", 1, 5, 3)

        absences = st.number_input(
            "School absences",
            min_value=0,
            max_value=100,
            value=0
        )

        G1 = st.slider("First-period grade (0–20)", 0, 20, 10)
        G2 = st.slider("Second-period grade (0–20)", 0, 20, 10)

    submitted = st.form_submit_button(
        "Predict Academic Risk",
        use_container_width=True
    )

if submitted:
    student = pd.DataFrame([{
        "school": school,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": Pstatus,
        "Medu": Medu,
        "Fedu": Fedu,
        "Mjob": Mjob,
        "Fjob": Fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": Dalc,
        "Walc": Walc,
        "health": health,
        "absences": absences,
        "G1": G1,
        "G2": G2
    }])

    try:
        prediction = int(model.predict(student)[0])

        st.subheader("Prediction Result")

        if prediction == 1:
            st.warning("Potential academic risk predicted")
        else:
            st.success("Lower predicted academic risk")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(student)[0]
            risk_index = list(model.classes_).index(1)
            risk_probability = probabilities[risk_index]

            st.metric(
                "Estimated risk probability",
                f"{risk_probability:.1%}"
            )

            st.caption(
                "This is a model-generated estimate, not a calibrated "
                "or guaranteed probability of future academic difficulty."
            )

    except Exception as error:
        st.error("The prediction could not be completed.")
        st.exception(error)