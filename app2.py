import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- 1. SETUP THE PAGE CONFIGURATION ---
st.set_page_config(page_title="BP Predictor AI", page_icon="🩺", layout="centered")

# --- 2. TRAIN THE MODEL (On the fly) ---
# We use the @st.cache_resource decorator so the model isn't retrained every time a user clicks a button
@st.cache_resource
def train_model():
    # Synthetic Data for Training (Age, Weight -> Systolic BP)
    data = {
        'Age': [25, 30, 45, 50, 60, 22, 35, 55, 65, 40, 48, 70, 28, 33, 58],
        'Weight': [60, 65, 80, 85, 90, 55, 70, 88, 95, 75, 82, 92, 62, 68, 86],
        'Systolic_BP': [115, 118, 130, 135, 145, 110, 122, 140, 150, 128, 133, 155, 116, 120, 142]
    }
    df = pd.DataFrame(data)
    
    X = df[['Age', 'Weight']]
    y = df['Systolic_BP']
    
    model = LinearRegression()
    model.fit(X, y)
    return model

model = train_model()

# --- 3. CREATE THE WEBSITE INTERFACE ---
st.title("🩺 Blood Pressure Predictor")
st.write("This application uses a **Linear Regression Algorithm** to estimate Systolic Blood Pressure based on age and weight.")
st.info("⚠️ **Disclaimer:** This is a demonstration project using synthetic data. Do not use for medical diagnosis.")

st.write("---")

# Create two columns for the input fields
col1, col2 = st.columns(2)

with col1:
    st.header("Patient Details")
    age = st.slider("Select Age", 18, 100, 30)
    weight = st.number_input("Enter Weight (kg)", min_value=40, max_value=150, value=70)

with col2:
    st.header("Live Analysis")
    # Real-time visual feedback
    st.metric(label="Current Age", value=f"{age} yrs")
    st.metric(label="Current Weight", value=f"{weight} kg")

# --- 4. PREDICTION LOGIC ---
if st.button("Predict Blood Pressure", type="primary"):
    # Reshape input to match model requirements
    input_data = np.array([[age, weight]])
    prediction = model.predict(input_data)[0]
    
    st.write("---")
    st.subheader("Prediction Result")
    
    # Display the result with color coding
    result_value = round(prediction, 2)
    
    if result_value < 120:
        st.success(f"Estimated Systolic BP: **{result_value} mmHg** (Normal Range)")
    elif 120 <= result_value < 130:
        st.warning(f"Estimated Systolic BP: **{result_value} mmHg** (Elevated)")
    else:
        st.error(f"Estimated Systolic BP: **{result_value} mmHg** (High)")

# --- 5. SHOW ALGORITHM INTERNALS (Optional Educational View) ---
with st.expander("See How It Works (Algorithm Details)"):
    st.write("The Linear Regression model calculated the following coefficients from the training data:")
    st.latex(r''' BP = \beta_0 + (\beta_1 \times Age) + (\beta_2 \times Weight) ''')
    st.write(f"**Intercept ($\beta_0$):** {model.intercept_:.2f}")
    st.write(f"**Age Coefficient ($\beta_1$):** {model.coef_[0]:.2f}")
    st.write(f"**Weight Coefficient ($\beta_2$):** {model.coef_[1]:.2f}")