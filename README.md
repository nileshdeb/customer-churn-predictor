#  Customer Churn Prediction Web App

##  What is Customer Churn?

**Customer churn** occurs when a customer discontinues a service or subscription.  
For subscription-based industries like **telecom, banking, SaaS**, or **e-commerce**, churn is a critical business metric.

>  Reducing churn improves profitability, customer lifetime value, and reduces acquisition cost.

---

##  About the Project

This project uses **machine learning** to predict whether a customer is likely to churn based on service usage patterns and personal details.

 Built with:
- **Scikit-learn**, **SMOTE** for data balancing
- **Random Forest Classifier** (best model)
- **Streamlit** for interactive UI

 Business goal: Help organizations proactively retain at-risk customers.

---

##  Model Details

- **Dataset:** IBM Telco Customer Churn Dataset
- **Features Used:** Gender, SeniorCitizen, Tenure, PhoneService, InternetService, Charges, etc.
- **Preprocessing:** Label encoding + missing value handling
- **Balancing:** Applied **SMOTE** to tackle class imbalance
- **Trained Models:**
  -  Random Forest (Selected)
  - Decision Tree
  - XGBoost
- **Validation Metric:** 5-Fold Cross-Validation  
- **Test Accuracy:** ~80%

---

##  Streamlit Web App Features

Interactive features of the deployed Streamlit app:
-  Input customer details via sidebar
-  Get churn prediction and probability instantly
-  View actionable recommendations for retention

###  Screenshot

![Web App Screenshot](https://github.com/nileshdeb/customer-churn-predictor/blob/main/Churn_prediction_screenshot.png)

---

##  Live App (Streamlit Cloud)

 [View Live Demo]([https://github.com/nileshdeb/customer-churn-predictor/blob/main/Churn_prediction_screenshot.png](https://customer-churn-predictor-asysszhf5nnyvjym7tcpz5.streamlit.app/))  

