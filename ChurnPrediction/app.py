import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time

st.set_page_config(
    page_title="Customer Churn Prediction System",
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
        .churn-alert {
            background-color: #fef2f2;
            border-color: #fca5a5;
            color: #991b1b;
        }
        .stay-alert {
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
        "best_model.pkl",
        "assets/churn_distribution.png",
        "assets/roc_curve.png",
        "assets/confusion_matrix.png",
        "assets/feature_importance.png",
        "assets/balance_boxplot.png"
    ]
    return all(os.path.exists(f) for f in required_files)

@st.cache_resource
def load_ml_assets():
    try:
        model = joblib.load("best_model.pkl")
        return model
    except Exception as e:
        return None

@st.cache_data
def load_churn_data():
    try:
        if os.path.exists("Churn_Modelling.csv"):
            return pd.read_csv("Churn_Modelling.csv")
    except:
        return None
    return None

def main():
    st.sidebar.title(" Churn Prediction")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation Menu",
        ["Home", "Churn Prediction", "Model Performance", "Data Visualizations"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Final Selected Model:**  
        Random Forest Classifier  
        
        **Test Set Performance:**  
        * Accuracy: **85.85%**  
        * Precision: **77.68%**  
        * Recall: **42.75%**  
        * F1-Score: **55.15%**  
        * ROC-AUC: **85.02%**  
        """
    )
    
    assets_ready = check_assets_exist()
    
    if not assets_ready:
        st.warning("️ **Notice: Machine learning models or visual assets are missing.**")
        st.info("""
            To generate the static visual assets, please run the following command in your terminal first:
            ```powershell
            python generate_churn_assets.py
            ```
            Once the asset script completes, refresh this page to access full model predictions and visualizations.
        """)

    if page == "Home":
        st.title("Customer Churn Prediction System")
        st.markdown("*A Machine Learning Case Study for Customer Retention in Financial Institutions*")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("<h3 class='section-header'>Project Introduction</h3>", unsafe_allow_html=True)
            st.write(
                """
                In the highly competitive banking and financial services sector, retaining existing customers is 
                significantly more cost-effective than acquiring new ones. Customer churn—when a customer terminates 
                their relationship with an institution—directly impacts profitability and market share.
                
                This project focuses on building an automated Machine Learning pipeline to identify customers who are 
                at high risk of leaving the bank. By analyzing key customer metrics such as credit score, account balance, 
                activity level, age, and product usage, we deploy a predictive model that enables retention teams 
                to intervene proactively.
                """
            )
            
            st.markdown("<h3 class='section-header'>Problem Statement</h3>", unsafe_allow_html=True)
            st.write(
                """
                The primary objective is to classify whether a customer will **Exit** (Class 1) or **Stay** (Class 0) 
                based on their financial and demographic profile. 
                
                Like most retention problems, the dataset features a class imbalance where roughly 20.37% of customers 
                historically exited. Building a model that predicts all customers will stay would achieve ~80% accuracy but 
                fail to catch any churn. Therefore, we optimize for **Precision**, **Recall**, and **ROC-AUC** to balance 
                accuracy with true predictive power.
                """
            )
            
            st.markdown("<h3 class='section-header'>Why Churn Prediction is Important</h3>", unsafe_allow_html=True)
            st.write(
                """
                - **Revenue Preservation**: Retains high-value accounts before they move to competitors.
                - **Targeted Incentives**: Helps marketing teams offer discounts, special interest rates, or loyalty perks only to high-risk customers.
                - **Customer Insights**: Identifies pain points in customer experience (e.g. why Germany has higher churn rates).
                - **Customer Lifetime Value (LTV)**: Enhances long-term relationship profitability.
                """
            )
            
        with col2:
            st.markdown("<h3 class='section-header'>Final Model Summary</h3>", unsafe_allow_html=True)
            st.write(
                """
                Our exploration showed that the **Random Forest Classifier** achieved the best performance compared to 
                Logistic Regression and Decision Trees, providing the optimal balance of precision and recall.
                """
            )
            
            st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">Random Forest</div>
                    <div class="metric-label">Best Performing Model</div>
                </div>
                <br>
                <div class="metric-card">
                    <div class="metric-value">85.85%</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                <br>
                <div class="metric-card">
                    <div class="metric-value">77.68%</div>
                    <div class="metric-label">Precision</div>
                </div>
                <br>
                <div class="metric-card">
                    <div class="metric-value">42.75%</div>
                    <div class="metric-label">Recall</div>
                </div>
                <br>
                <div class="metric-card">
                    <div class="metric-value">85.02%</div>
                    <div class="metric-label">ROC-AUC Score</div>
                </div>
            """, unsafe_allow_html=True)

    elif page == "Churn Prediction":
        st.title("Customer Churn Risk Predictor")
        st.write("Input the customer profile parameters below to run the Random Forest churn risk classifier.")
        
        if not assets_ready:
            st.error("Model file not found. Prediction form is disabled until you run the asset generation script.")
            return

        model = load_ml_assets()
        
        if model is None:
            st.error("Failed to load best_model.pkl. Please check your project path.")
            return
            
        with st.form("churn_prediction_form"):
            st.markdown("<h3 class='section-header'>Customer Demographics</h3>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                geography = st.selectbox(
                    "Geography (Country of Residence)",
                    ["France", "Germany", "Spain"],
                    index=0,
                    help="Select the customer's region of residence."
                )
                gender = st.selectbox(
                    "Gender",
                    ["Female", "Male"],
                    index=0,
                    help="Select the customer's gender."
                )
            
            with col2:
                age = st.slider(
                    "Age (Years)",
                    min_value=18,
                    max_value=100,
                    value=38,
                    step=1,
                    help="Select the customer's age in years."
                )
                
            with col3:
                tenure = st.slider(
                    "Tenure (Years as Customer)",
                    min_value=0,
                    max_value=10,
                    value=5,
                    step=1,
                    help="Number of years the customer has been with the bank."
                )
            
            st.markdown("<h3 class='section-header'>Financial & Account Profile</h3>", unsafe_allow_html=True)
            col4, col5, col6 = st.columns(3)
            
            with col4:
                credit_score = st.slider(
                    "Credit Score",
                    min_value=300,
                    max_value=850,
                    value=650,
                    step=1,
                    help="Enter the customer's credit score rating."
                )
                num_products = st.selectbox(
                    "Number of Products Registered",
                    [1, 2, 3, 4],
                    index=0,
                    help="Number of bank products (accounts, loans, cards, etc.) registered by the customer."
                )
                
            with col5:
                balance = st.number_input(
                    "Account Balance ($)",
                    min_value=0.0,
                    max_value=300000.0,
                    value=75000.0,
                    step=1000.0,
                    format="%.2f",
                    help="Enter the customer's current total account balance."
                )
                has_cr_card_val = st.selectbox(
                    "Possesses Credit Card",
                    ["Yes", "No"],
                    index=0,
                    help="Does the customer have an active credit card with the bank?"
                )
                
            with col6:
                estimated_salary = st.number_input(
                    "Estimated Annual Salary ($)",
                    min_value=0.0,
                    max_value=300000.0,
                    value=100000.0,
                    step=1000.0,
                    format="%.2f",
                    help="Enter the customer's estimated annual salary."
                )
                is_active_member_val = st.selectbox(
                    "Is Active Member",
                    ["Yes", "No"],
                    index=0,
                    help="Is the customer actively using bank services?"
                )

            st.markdown("---")
            submit_button = st.form_submit_button("Predict Churn Risk", type="primary")

        if submit_button:
            try:
                balance_salary_ratio = balance / (estimated_salary + 1)
                is_senior = 1 if age > 50 else 0
                
                geography_germany = 1 if geography == 'Germany' else 0
                geography_spain = 1 if geography == 'Spain' else 0
                
                gender_male = 1 if gender == 'Male' else 0
                
                is_active_member = 1 if is_active_member_val == "Yes" else 0
                has_cr_card = 1 if has_cr_card_val == "Yes" else 0

                feature_names = [
                    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
                    "IsActiveMember", "EstimatedSalary", "BalanceSalaryRatio", "IsSenior",
                    "Geography_Germany", "Geography_Spain", "Gender_Male"
                ]
                
                input_dict = {
                    "CreditScore": int(credit_score),
                    "Age": int(age),
                    "Tenure": int(tenure),
                    "Balance": float(balance),
                    "NumOfProducts": int(num_products),
                    "HasCrCard": int(has_cr_card),
                    "IsActiveMember": int(is_active_member),
                    "EstimatedSalary": float(estimated_salary),
                    "BalanceSalaryRatio": float(balance_salary_ratio),
                    "IsSenior": int(is_senior),
                    "Geography_Germany": int(geography_germany),
                    "Geography_Spain": int(geography_spain),
                    "Gender_Male": int(gender_male)
                }

                input_df = pd.DataFrame([input_dict])
                input_df = input_df[feature_names]
                
                prediction = model.predict(input_df)[0]
                probabilities = model.predict_proba(input_df)[0]
                
                churn_prob = probabilities[1] * 100
                stay_prob = probabilities[0] * 100

                if prediction == 1:
                    st.markdown(
                        f"""
                        <div class="prediction-box churn-alert">
                            <h3 style="margin-top: 0; color: #991b1b;">️ High Churn Risk Detected</h3>
                            <p style="margin-bottom: 0;">
                                The system has classified this customer as <b>Likely to Churn</b> with a confidence of <b>{churn_prob:.2f}%</b>.
                                <br>It is recommended that customer loyalty teams contact this customer with proactive retention incentives.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="prediction-box stay-alert">
                            <h3 style="margin-top: 0; color: #166534;"> Low Churn Risk Approved</h3>
                            <p style="margin-bottom: 0;">
                                The system has classified this customer as <b>Likely to Stay</b> with a confidence of <b>{stay_prob:.2f}%</b>.
                                <br>The customer exhibits behaviors consistent with historical retention standards.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                st.markdown('<div class="section-title"> Calculated Predictors</div>', unsafe_allow_html=True)
                col_ins1, col_ins2 = st.columns(2)
                with col_ins1:
                    with st.container(border=True):
                        st.markdown("<p style='font-size: 0.9rem; color: #475569; margin-bottom: 0.25rem; font-weight: 500;'>BALANCE TO SALARY RATIO</p>", unsafe_allow_html=True)
                        st.markdown(f"<h3 style='margin: 0; color: #1e3a8a;'>{balance_salary_ratio:.4f}</h3>", unsafe_allow_html=True)
                        st.markdown("<p style='font-size: 0.8rem; color: #64748b; margin-top: 0.25rem;'>Account balance divided by estimated annual salary. High ratios imply low liquidity.</p>", unsafe_allow_html=True)
                
                with col_ins2:
                    with st.container(border=True):
                        st.markdown("<p style='font-size: 0.9rem; color: #475569; margin-bottom: 0.25rem; font-weight: 500;'>SENIOR CLASSIFICATION</p>", unsafe_allow_html=True)
                        senior_text = "Yes (Age > 50)" if is_senior else "No (Age <= 50)"
                        senior_color = "#991b1b" if is_senior else "#166534"
                        st.markdown(f"<h3 style='margin: 0; color: {senior_color};'>{senior_text}</h3>", unsafe_allow_html=True)
                        st.markdown("<p style='font-size: 0.8rem; color: #64748b; margin-top: 0.25rem;'>Indicates if the customer is over 50 years of age (statistically higher churn risk).</p>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error executing prediction pipeline: {str(e)}")
                st.info("Ensure all input variables are entered correctly.")

    elif page == "Model Performance":
        st.title("Model Evaluation & Metrics")
        st.write("Review the test metrics for our final Random Forest model and learn what they mean in a churn prediction context.")
        
        st.markdown("<h3 class='section-header'>Random Forest Test Metrics</h3>", unsafe_allow_html=True)
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown('<div class="metric-card"><div class="metric-value">85.85%</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><div class="metric-value">77.68%</div><div class="metric-label">Precision</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><div class="metric-value">42.75%</div><div class="metric-label">Recall</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card"><div class="metric-value">55.15%</div><div class="metric-label">F1-Score</div></div>', unsafe_allow_html=True)
        with col5:
            st.markdown('<div class="metric-card"><div class="metric-value">85.02%</div><div class="metric-label">ROC-AUC</div></div>', unsafe_allow_html=True)
            
        st.markdown("<h3 class='section-header'>Understanding the Metrics</h3>", unsafe_allow_html=True)
        
        metrics_info = [
            ("Accuracy (85.85%)", 
             "Accuracy represents the overall proportion of correct classifications (both stayed and exited). "
             "While 85.85% is strong, it can hide weaknesses because only ~20% of customers churned. Evaluated alone, accuracy is not "
             "a complete performance descriptor."),
            
            ("Precision (77.68%)", 
             "Precision answers: *Out of all customers predicted to churn, how many actually did?* "
             "A precision of 77.68% indicates that when the model alerts that a customer is at risk, there is a 77.68% chance they are "
             "indeed likely to exit. This helps prevent wasting retention resources (discounts, fee waivers) on low-risk customers."),
            
            ("Recall / Sensitivity (42.75%)", 
             "Recall answers: *Out of all customers who actually churned, how many did the system catch?* "
             "A recall of 42.75% indicates that the model catches slightly over 40% of the total churners, leaving ~57% missed. In churn mitigation, "
             "there is a constant trade-off between higher recall (catching more churners) and precision (avoiding false alarms)."),
            
            ("F1 Score (55.15%)", 
             "The F1 Score is the harmonic mean of Precision and Recall. It provides a single metric that balances both factors. "
             "A score of 55.15% is solid and indicates the model finds a reasonable balance between catching churn and keeping false positives low."),
            
            ("ROC-AUC Score (85.02%)", 
             "The ROC-AUC (Area Under the Receiver Operating Characteristic curve) measures the model's ability to distinguish between classes. "
             "An AUC of 85.02% is excellent, demonstrating that the model has a very strong capability to rank churn-prone customers higher in probability "
             "than loyal customers.")
        ]
        
        for name, desc in metrics_info:
            st.markdown(f"##### **{name}**")
            st.write(desc)
            st.write("")

    elif page == "Data Visualizations":
        st.title("Exploratory Data Analysis & Model Visualizations")
        st.write("Visualizations generated from the churn dataset and Random Forest classification results.")
        
        if not assets_ready:
            st.warning("Visualizations cannot be loaded because they have not been generated. Please run `python generate_churn_assets.py`.")
            return
            
        tab1, tab2, tab3 = st.tabs([" Data Distribution", " Model Evaluation Plots", " Feature Analysis"])
        
        with tab1:
            st.markdown("<h3 class='section-header'>Data Distribution</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns([1.2, 1])
            with col1:
                st.image("assets/churn_distribution.png", caption="Fig 1: Customer Exit Class Distribution", use_container_width=True)
            with col2:
                st.write(
                    """
                    **Exited vs Retained Distribution**:  
                    As shown in the graph, the dataset exhibits class imbalance. 
                    Out of 10,000 customers, 7,963 (79.63%) stayed with the bank (Class 0), while 2,037 (20.37%) exited (Class 1). 
                    
                    Due to this imbalance, standard classifiers might over-predict Class 0. We evaluate model performance using precision, 
                    recall, and ROC-AUC curves on test datasets to ensure robust results.
                    """
                )
            
            st.markdown("---")
            col3, col4 = st.columns([1.2, 1])
            with col3:
                st.image("assets/balance_boxplot.png", caption="Fig 2: Account Balance Distribution by Churn Status", use_container_width=True)
            with col4:
                st.write(
                    """
                    **Account Balance vs Churn Boxplot**:  
                    Comparing account balances of retained vs churned customers shows:
                    - Retained customers have a wider distribution including a large cluster of zero-balance accounts.
                    - Churned customers have a higher median balance. 
                    
                    This indicates that customers with higher balances (who are more valuable to the bank) may churn more frequently, 
                    highlighting the urgency of using prediction tools to safeguard these high-value accounts.
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
                    The ROC Curve displays performance tradeoffs:
                    - **Random Forest** dominates with the highest Area Under Curve (**AUC = 0.8502**).
                    - **Logistic Regression** (AUC = 0.7745) achieves a reasonable linear separation.
                    - **Decision Tree** (AUC = 0.6701) experiences overfitting and high variance on single splits.
                    """
                )
                
            with col2:
                st.image("assets/confusion_matrix.png", caption="Fig 4: Random Forest Test Set Confusion Matrix", use_container_width=True)
                st.write(
                    """
                    **Confusion Matrix Insights**:  
                    Evaluating the Random Forest model on the 2,000 test samples (20% split):
                    - **True Negatives**: 1,542 loyal customers correctly classified.
                    - **True Positives**: 174 churned customers correctly caught.
                    - **False Negatives**: 233 churned customers missed.
                    - **False Positives**: Only 51 loyal customers incorrectly flagged.
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
                    The Random Forest classifier calculates feature importance scores based on node purity reduction:
                    
                    1. **Age**: By far the most critical feature. Churn probabilities differ significantly across age groups (older demographics show higher churn rates).
                    2. **NumOfProducts**: Registered products strongly correlate with retention. Customers with 1 product churn far more than those with 2 products, but 3 or 4 products signal high-risk outliers.
                    3. **Balance**: High savings deposits are strong indicators of potential exits.
                    4. **CreditScore**: Measures credit health.
                    5. **EstimatedSalary**: Customer income correlates with financial behavior.
                    6. **BalanceSalaryRatio**: Custom feature calculating account asset ratio to salary.
                    """
                )


if __name__ == "__main__":
    main()