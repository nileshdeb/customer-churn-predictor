import streamlit as st
import pandas as pd
import pickle

# Improved loader that handles different file structures
@st.cache_resource
def load_artifacts():
    try:
        # Try loading model (handles both direct model and dict cases)
        with open('customer_churn_model.pkl', 'rb') as f:
            model_content = pickle.load(f)
            model = model_content['model'] if isinstance(model_content, dict) else model_content
        
        # Verify it's a proper model
        if not hasattr(model, 'predict'):
            st.error("The model file doesn't contain a valid model with predict method")
            st.stop()
            
        # Load encoders
        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
            
        return model, encoders
        
    except Exception as e:
        st.error(f"Error loading files: {str(e)}")
        st.stop()

model, encoders = load_artifacts()

# Rest of your Streamlit app code remains the same...

# Verify the loaded model has predict method
if not hasattr(model, 'predict'):
    st.error("Loaded model object doesn't have predict method. Please check your model file.")
    st.stop()

# App UI
st.title("📊 Customer Churn Prediction")
st.markdown("Predict whether a customer is likely to churn based on their service details")

# Sidebar inputs
with st.sidebar:
    st.header("Customer Details")
    
    # Personal info
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.radio("Senior Citizen", ["No", "Yes"], horizontal=True)
    partner = st.radio("Has Partner", ["No", "Yes"], horizontal=True)
    dependents = st.radio("Has Dependents", ["No", "Yes"], horizontal=True)
    
    # Service info
    tenure = st.slider("Tenure (months)", 1, 72, 25)
    phone_service = st.radio("Phone Service", ["No", "Yes"], horizontal=True)
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    
    # Internet service
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    # Online services
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    
    # Contract and billing
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.radio("Paperless Billing", ["No", "Yes"], horizontal=True)
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", 
        "Mailed check", 
        "Bank transfer (automatic)", 
        "Credit card (automatic)"
    ])
    
    # Charges
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=75.25, step=1.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=488.55, step=1.0)

# Prediction function
def predict_churn():
    input_data = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical features
    for col in encoders:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str).str.title()
            unknown_mask = ~input_df[col].isin(encoders[col].classes_)
            if unknown_mask.any():
                default_value = encoders[col].classes_[0]
                input_df.loc[unknown_mask, col] = default_value
            input_df[col] = encoders[col].transform(input_df[col])
    
    # Ensure correct column order (adjust based on your model)
    feature_order = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ]
    
    input_df = input_df.reindex(columns=feature_order, fill_value=0)
    
    # Make prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1] if hasattr(model, 'predict_proba') else float(prediction[0])
    
    return prediction[0], probability

# Prediction button
if st.button("Predict Churn Probability", type="primary"):
    with st.spinner("Analyzing customer data..."):
        try:
            prediction, probability = predict_churn()
            
            st.divider()
            st.subheader("Prediction Result")
            
            if prediction == 1:
                st.error(f"🚨 High churn risk: {probability*100:.1f}% probability")
                st.markdown("**Recommended actions:**")
                st.markdown("- Offer retention discount")
                st.markdown("- Provide personalized service review")
                st.markdown("- Consider loyalty benefits")
            else:
                st.success(f"✅ Low churn risk: {probability*100:.1f}% probability")
                st.markdown("**Customer appears satisfied with service**")
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# Add some app info
st.divider()
st.markdown("""
This interactive tool predicts customer churn risk using machine learning:

1. Input customer service details  
2. Receive instant risk probability  
3. View tailored retention recommendations  
""")