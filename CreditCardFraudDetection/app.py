import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime


st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        /* Main page styling */
        html, body, [class*="css"] {
            font-family: 'Inter', 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
        }
        
        /* Metric card container styling */
        .metric-card {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #1e3a8a;
            margin-bottom: 4px;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #475569;
            font-weight: 500;
        }
        
        /* Custom prediction outcome banners */
        .prediction-box {
            padding: 24px;
            border-radius: 8px;
            border-width: 1px;
            border-style: solid;
            margin: 20px 0;
        }
        .fraud-alert {
            background-color: #fef2f2;
            border-color: #fca5a5;
            color: #991b1b;
        }
        .legit-alert {
            background-color: #f0fdf4;
            border-color: #bbf7d0;
            color: #166534;
        }
        
        /* Custom section headers */
        .section-header {
            color: #1e293b;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 8px;
            margin-bottom: 20px;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

def check_assets_exist():
    required_files = [
        "fraud_detection_model.pkl",
        "label_encoders.pkl",
        "assets/fraud_distribution.png",
        "assets/roc_curve.png",
        "assets/confusion_matrix.png",
        "assets/feature_importance.png",
        "assets/amount_boxplot.png"
    ]
    return all(os.path.exists(f) for f in required_files)

@st.cache_resource
def load_ml_assets():
    try:
        model = joblib.load("fraud_detection_model.pkl")
        encoders = joblib.load("label_encoders.pkl")
        return model, encoders
    except Exception as e:
        return None, None

def main():
    st.sidebar.title(" Fraud Detection Project")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation Menu",
        ["Home", "Fraud Prediction", "Model Performance", "Data Visualizations"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Final Selected Model:**  
        Random Forest Classifier  
        
        **Test Set Performance:**  
        * Accuracy: **99.86%**  
        * Precision: **95.37%**  
        * Recall: **67.23%**  
        * F1-Score: **78.86%**  
        * ROC-AUC: **97.52%**  
        """
    )
    
    assets_ready = check_assets_exist()
    
    if not assets_ready:
        st.warning("**Notice: Machine learning models or visual assets are missing.**")
        st.info("""
            To train the Random Forest model and generate the static visual assets, please run the following command in your terminal first:
            ```powershell
            python train_and_generate.py
            ```
            Once the training script completes, refresh this page to access full model predictions and visualizations.
        """)

    if page == "Home":
        st.title("Credit Card Fraud Detection System")
        st.markdown("*A Machine Learning Case Study for Fraud Prevention in Financial Transactions*")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("<h3 class='section-header'>Project Introduction</h3>", unsafe_allow_html=True)
            st.write(
                """
                In today's digital economy, credit card transactions are the primary medium of commerce. 
                With this convenience comes the critical challenge of credit card fraud, which costs financial institutions and consumers 
                billions of dollars annually. Fraudsters continuously evolve their tactics, requiring sophisticated, real-time 
                automated detection mechanisms.
                
                This project focuses on building an end-to-end Machine Learning solution that analyzes transactions in real time 
                to determine whether they are legitimate or fraudulent. By training several supervised learning algorithms 
                (Logistic Regression, Decision Trees, and Random Forest), we selected the optimal classifier to deploy.
                """
            )
            
            st.markdown("<h3 class='section-header'>Problem Statement</h3>", unsafe_allow_html=True)
            st.write(
                """
                The primary objective is to classify transactions as **Legitimate** or **Fraudulent** based on transaction characteristics 
                (amount, merchant, category, customer profile, and geographic location). 
                
                A core challenge of fraud detection is the extreme **class imbalance**—only a tiny fraction of total transactions are 
                fraudulent (~0.58% in our training data). A model that predicts all transactions as legitimate would achieve 99.42% accuracy 
                but fail completely to detect fraud. Therefore, we must focus on metrics like **Recall**, **Precision**, and **F1 Score** 
                to evaluate performance.
                """
            )
            
            st.markdown("<h3 class='section-header'>Why Fraud Detection is Important</h3>", unsafe_allow_html=True)
            st.write(
                """
                - **Financial Loss Mitigation**: Prevents unauthorized transactions before they are finalized.
                - **Consumer Protection**: Strengthens customer trust in digital banking services.
                - **Regulatory Compliance**: Helps financial institutions adhere to security and anti-fraud regulations.
                - **Operational Efficiency**: Reduces manual review workloads for risk management teams.
                """
            )
            
        with col2:
            st.markdown("<h3 class='section-header'>Final Model Summary</h3>", unsafe_allow_html=True)
            st.write(
                """
                Our testing showed that the **Random Forest Classifier** outperformed Logistic Regression and Decision Trees, 
                achieving excellent precision and solid recall.
                """
            )
            
            st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">Random Forest</div>
                    <div class="metric-label">Best Performing Model</div>
                </div>
                <br>
                <div class="metric-card">
                    <div class="metric-value">99.86%</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                <br>
                <div class="metric-card">
                    <div class="metric-value">95.37%</div>
                    <div class="metric-label">Precision</div>
                </div>
                <br>
                <div class="metric-card">
                    <div class="metric-value">67.23%</div>
                    <div class="metric-label">Recall</div>
                </div>
                <br>
                <div class="metric-card">
                    <div class="metric-value">97.52%</div>
                    <div class="metric-label">ROC-AUC Score</div>
                </div>
            """, unsafe_allow_html=True)

    elif page == "Fraud Prediction":
        st.title("Transaction Fraud Predictor")
        st.write("Input the transaction parameters below to run the Random Forest fraud classifier.")
        
        if not assets_ready:
            st.error("Model file not found. Prediction form is disabled until you run the training script.")
            return

        model, encoders = load_ml_assets()
        
        if model is None or encoders is None:
            st.error("Failed to load model or label encoders. Please check your project path.")
            return
            
        with st.form("prediction_form"):
            st.markdown("<h3 class='section-header'>Transaction Details</h3>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                amt = st.number_input("Transaction Amount ($)", min_value=0.01, max_value=100000.0, value=50.00, step=5.0, help="Amount of the transaction in USD")
                category = st.selectbox("Transaction Category", options=sorted(encoders['category'].classes_), index=4) # Default to gas_transport index
            
            with col2:
                merchant = st.selectbox("Merchant Name", options=sorted(encoders['merchant'].classes_), index=203)
                trans_date = st.date_input("Transaction Date", value=datetime.today())
                
            with col3:
                trans_time = st.time_input("Transaction Time", value=datetime.now().time())
            
            st.markdown("<h3 class='section-header'>Customer Profile</h3>", unsafe_allow_html=True)
            col4, col5, col6 = st.columns(3)
            
            with col4:
                gender = st.selectbox("Gender", options=encoders['gender'].classes_, index=0)
                age = st.number_input("Age", min_value=18, max_value=120, value=35, step=1, help="Age calculated at transaction time")
                
            with col5:
                job = st.selectbox("Customer Occupation", options=sorted(encoders['job'].classes_), index=394)
                city = st.selectbox("Customer City", options=sorted(encoders['city'].classes_), index=476)
                
            with col6:
                state = st.selectbox("Customer State", options=sorted(encoders['state'].classes_), index=34)
                zip_code = st.number_input("Customer Zip Code", min_value=1, max_value=99999, value=10950, step=1)
                
            st.markdown("<h3 class='section-header'>Geography & Demographics</h3>", unsafe_allow_html=True)
            col7, col8, col9 = st.columns(3)
            
            with col7:
                city_pop = st.number_input("City Population", min_value=1, max_value=20000000, value=5837, step=100)
                
            with col8:
                lat = st.number_input("Customer Latitude", min_value=-90.0, max_value=90.0, value=41.32, step=0.01)
                long_coord = st.number_input("Customer Longitude", min_value=-180.0, max_value=180.0, value=-74.19, step=0.01)
                
            with col9:
                merch_lat = st.number_input("Merchant Latitude", min_value=-90.0, max_value=90.0, value=41.25, step=0.01)
                merch_long = st.number_input("Merchant Longitude", min_value=-180.0, max_value=180.0, value=-74.88, step=0.01)

            st.markdown("---")
            submit_button = st.form_submit_button("Predict Transaction Class", type="primary")

        if submit_button:
            try:
                dt_combined = datetime.combine(trans_date, trans_time)
                unix_time = int(dt_combined.timestamp())
                trans_hour = dt_combined.hour
                trans_day = dt_combined.day
                trans_month = dt_combined.month

                merchant_enc = encoders['merchant'].transform([merchant])[0]
                category_enc = encoders['category'].transform([category])[0]
                gender_enc = encoders['gender'].transform([gender])[0]
                city_enc = encoders['city'].transform([city])[0]
                state_enc = encoders['state'].transform([state])[0]
                job_enc = encoders['job'].transform([job])[0]

                input_df = pd.DataFrame([{
                    'merchant': merchant_enc,
                    'category': category_enc,
                    'amt': amt,
                    'gender': gender_enc,
                    'city': city_enc,
                    'state': state_enc,
                    'zip': zip_code,
                    'lat': lat,
                    'long': long_coord,
                    'city_pop': city_pop,
                    'job': job_enc,
                    'unix_time': unix_time,
                    'merch_lat': merch_lat,
                    'merch_long': merch_long,
                    'trans_hour': trans_hour,
                    'trans_day': trans_day,
                    'trans_month': trans_month,
                    'age': age
                }])

                prediction = model.predict(input_df)[0]
                probabilities = model.predict_proba(input_df)[0]
                confidence = probabilities[prediction] * 100

                if prediction == 1:
                    st.markdown(
                        f"""
                        <div class="prediction-box fraud-alert">
                            <h3 style="margin-top: 0; color: #991b1b;">️ Fraudulent Transaction Detected</h3>
                            <p style="margin-bottom: 0;">
                                The system has classified this transaction as <b>Fraudulent</b> with a confidence of <b>{confidence:.2f}%</b>.
                                <br>It is recommended to block or place this transaction under review.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="prediction-box legit-alert">
                            <h3 style="margin-top: 0; color: #166534;"> Legitimate Transaction Approved</h3>
                            <p style="margin-bottom: 0;">
                                The system has classified this transaction as <b>Legitimate</b> with a confidence of <b>{confidence:.2f}%</b>.
                                <br>This transaction passes all automated fraud checks.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
            except Exception as e:
                st.error(f"Error executing prediction pipeline: {str(e)}")
                st.info("Ensure all input variables are entered correctly.")

    elif page == "Model Performance":
        st.title("Model Evaluation & Metrics")
        st.write("Review the test metrics for our final Random Forest model and learn what they mean in a fraud detection context.")
        
        st.markdown("<h3 class='section-header'>Random Forest Test Metrics</h3>", unsafe_allow_html=True)
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown('<div class="metric-card"><div class="metric-value">99.86%</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><div class="metric-value">95.37%</div><div class="metric-label">Precision</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><div class="metric-value">67.23%</div><div class="metric-label">Recall</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card"><div class="metric-value">78.86%</div><div class="metric-label">F1-Score</div></div>', unsafe_allow_html=True)
        with col5:
            st.markdown('<div class="metric-card"><div class="metric-value">97.52%</div><div class="metric-label">ROC-AUC</div></div>', unsafe_allow_html=True)
            
        st.markdown("<h3 class='section-header'>Understanding the Metrics</h3>", unsafe_allow_html=True)
        
        metrics_info = [
            ("Accuracy (99.86%)", 
             "Accuracy represents the overall proportion of correct classifications (both legitimate and fraudulent). "
             "While 99.86% looks exceptionally high, it is heavily skewed by the massive proportion of legitimate transactions. "
             "Evaluating accuracy alone is insufficient for imbalanced datasets."),
            
            ("Precision (95.37%)", 
             "Precision answers: *Out of all transactions predicted as fraudulent, how many were actually fraudulent?* "
             "A high precision of 95.37% means that when our system flags a transaction as fraud, there is a very low chance of a "
             "false positive (flagging a good transaction by mistake), which prevents customer frustration and unnecessary account freezes."),
            
            ("Recall / Sensitivity (67.23%)", 
             "Recall answers: *Out of all actual fraudulent transactions, how many did the system catch?* "
             "With a recall of 67.23%, our model successfully intercepts approximately two-thirds of all fraud. The remaining 32.77% are "
             "false negatives (fraudulent transactions missed by the model). In fraud systems, optimizing recall is key to saving costs."),
            
            ("F1 Score (78.86%)", 
             "The F1 Score is the harmonic mean of Precision and Recall. It provides a single metric that balances both factors. "
             "A high F1 Score indicates a robust model that effectively catches fraud while keeping false alarms to a minimum."),
            
            ("ROC-AUC Score (97.52%)", 
             "The ROC-AUC (Area Under the Receiver Operating Characteristic curve) measures the model's ability to distinguish between classes. "
             "A score of 97.52% indicates that if we randomly select a fraudulent and a legitimate transaction, the model has a 97.52% probability "
             "of ranking the fraudulent transaction higher than the legitimate one.")
        ]
        
        for name, desc in metrics_info:
            st.markdown(f"##### **{name}**")
            st.write(desc)
            st.write("")

    elif page == "Data Visualizations":
        st.title("Exploratory Data Analysis & Model Visualizations")
        st.write("Visualizations generated from the credit card dataset and Random Forest classification results.")
        
        if not assets_ready:
            st.warning("Visualizations cannot be loaded because they have not been generated. Please run `python train_and_generate.py`.")
            return
            
        tab1, tab2, tab3 = st.tabs([" Data Distribution", " Model Evaluation Plots", " Feature Analysis"])
        
        with tab1:
            st.markdown("<h3 class='section-header'>Data Distribution</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns([1.2, 1])
            with col1:
                st.image("assets/fraud_distribution.png", caption="Fig 1: Class Imbalance in Training Data (Log Scale)", use_container_width=True)
            with col2:
                st.write(
                    """
                    **Fraud vs Non-Fraud Distribution**:  
                    As shown in the graph, there is an extreme imbalance between the legitimate class (Class 0) and the fraudulent class (Class 1). 
                    Out of approximately 1.3 million training transactions, only 7,506 (0.58%) are actual fraud. 
                    
                    Due to this severe imbalance, we train our models using the standard dataset but leverage the Random Forest's inherent bootstrapping 
                    and decision tree structures to identify complex rules. We also evaluated Logistic Regression with balanced class weights as a comparison.
                    """
                )
            
            st.markdown("---")
            col3, col4 = st.columns([1.2, 1])
            with col3:
                st.image("assets/amount_boxplot.png", caption="Fig 2: Transaction Amount Distribution (Capped at $1,500)", use_container_width=True)
            with col4:
                st.write(
                    """
                    **Transaction Amount vs Fraud Boxplot**:  
                    Comparing transaction amounts reveals a distinct pattern. 
                    - Legitimate transactions are highly clustered at lower values, typically averaging below $100.
                    - Fraudulent transactions have a significantly wider spread and a higher median value. Many fraudulent transactions hover between $200 and $1,000. 
                    
                    This visual confirms that transaction amount is a highly discriminative feature for flagging potential credit card fraud.
                    """
                )
                
        with tab2:
            st.markdown("<h3 class='section-header'>ROC Curve & Confusion Matrix</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                st.image("assets/roc_curve.png", caption="Fig 3: ROC Curve Comparison of Logistic Regression, Decision Tree, and Random Forest", use_container_width=True)
                st.write(
                    """
                    **ROC Curve Analysis**:  
                    The Receiver Operating Characteristic (ROC) curve compares the True Positive Rate vs False Positive Rate.
                    - The **Random Forest** achieves the highest Area Under the Curve (**AUC = 0.9752**), showing outstanding ability to rank fraud above non-fraud.
                    - The **Logistic Regression** (AUC = 0.9632) is slightly lower, while the **Decision Tree** (AUC = 0.8361) is the weakest due to its single-tree variance.
                    """
                )
                
            with col2:
                st.image("assets/confusion_matrix.png", caption="Fig 4: Random Forest Test Set Confusion Matrix", use_container_width=True)
                st.write(
                    """
                    **Confusion Matrix Insights**:  
                    Evaluating the Random Forest model on the 555,719 test samples:
                    - **True Negatives**: 553,503 legitimate transactions correctly classified.
                    - **True Positives**: 1,442 fraudulent transactions correctly caught.
                    - **False Negatives**: 703 fraudulent transactions missed (Recall limit).
                    - **False Positives**: Only 70 legitimate transactions wrongly flagged (Precision rate is 95.37%).
                    """
                )

        with tab3:
            st.markdown("<h3 class='section-header'>Feature Importance</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns([1.3, 1])
            
            with col1:
                st.image("assets/feature_importance.png", caption="Fig 5: Random Forest Top 10 Feature Importance", use_container_width=True)
            with col2:
                st.write(
                    """
                    **Top Feature Importance Analysis**:  
                    The Random Forest classifier calculates the relative importance of each feature based on how much it reduces impurity across all trees:
                    
                    1. **amt (Amount)**: By far the most critical feature. As shown in the EDA boxplot, fraudulent amounts deviate significantly from normal spending.
                    2. **category**: Different merchant categories (e.g. online shopping vs grocery) experience different rates of fraud.
                    3. **unix_time**: Time elapsed is crucial. Speed of transactions and sequential patterns help flag unusual account bursts.
                    4. **merch_lat / merch_long**: The geographical location of the merchant.
                    5. **lat / long**: The customer's coordinates. Large geographical distances between customer location and merchant location (spatial velocity) strongly signal fraud.
                    """
                )


if __name__ == "__main__":
    main()