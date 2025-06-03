import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("titanic_model.pkl", "rb"))

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict survival")

# User inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 25)
fare = st.slider("Fare", 0, 500, 50)
sibsp = st.slider("Siblings/Spouses aboard", 0, 5, 0)
parch = st.slider("Parents/Children aboard", 0, 5, 0)

# Convert sex to numeric
sex = 1 if sex == "male" else 0

# Predict
input_features = np.array([[pclass, sex, age, fare, sibsp, parch]])
prediction = model.predict(input_features)[0]
prob = model.predict_proba(input_features)[0][1]

# Output
if st.button("Predict"):
    st.write(f"🧮 Survival Probability: {prob:.2f}")
    if prediction == 1:
        st.success("✅ Survived")
    else:
        st.error("❌ Did not survive")
