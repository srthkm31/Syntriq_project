import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

def main():
    print("Step 1: Loading datasets...")
    train_df = pd.read_csv("fraudTrain.csv")
    test_df = pd.read_csv("fraudTest.csv")
    print(f"Loaded train shape: {train_df.shape}, test shape: {test_df.shape}")

    print("Step 2: Preprocessing and feature engineering...")
    drop_cols = ['Unnamed: 0', 'trans_num', 'first', 'last', 'street', 'cc_num']
    train_df = train_df.drop(columns=drop_cols)
    test_df = test_df.drop(columns=drop_cols)

    train_df['trans_date_trans_time'] = pd.to_datetime(train_df['trans_date_trans_time'])
    test_df['trans_date_trans_time'] = pd.to_datetime(test_df['trans_date_trans_time'])
    train_df['dob'] = pd.to_datetime(train_df['dob'])
    test_df['dob'] = pd.to_datetime(test_df['dob'])

    for df in [train_df, test_df]:
        df['trans_hour'] = df['trans_date_trans_time'].dt.hour
        df['trans_day'] = df['trans_date_trans_time'].dt.day
        df['trans_month'] = df['trans_date_trans_time'].dt.month
        df['age'] = 2026 - df['dob'].dt.year

    train_df.drop(columns=['trans_date_trans_time', 'dob'], inplace=True)
    test_df.drop(columns=['trans_date_trans_time', 'dob'], inplace=True)

    print("Step 3: Encoding categorical variables...")
    categorical_cols = ['merchant', 'category', 'gender', 'city', 'state', 'job']
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        all_values = pd.concat([train_df[col], test_df[col]]).astype(str)
        le.fit(all_values)
        
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        
        encoders[col] = le

    X_train = train_df.drop('is_fraud', axis=1)
    y_train = train_df['is_fraud']
    X_test = test_df.drop('is_fraud', axis=1)
    y_test = test_df['is_fraud']

    print("Step 4: Scaling data for Logistic Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs("assets", exist_ok=True)

    print("Step 5: Training Logistic Regression...")
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]

    print("Step 6: Training Decision Tree...")
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    dt_probs = dt.predict_proba(X_test)[:, 1]

    print("Step 7: Training Random Forest (Selected Model)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    y_pred_rf = rf.predict(X_test)

    print("Step 8: Saving models and encoders...")
    joblib.dump(rf, "fraud_detection_model.pkl")
    joblib.dump(encoders, "label_encoders.pkl")
    print("Model saved to 'fraud_detection_model.pkl'")
    print("Encoders saved to 'label_encoders.pkl'")

    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    rf_precision = precision_score(y_test, y_pred_rf)
    rf_recall = recall_score(y_test, y_pred_rf)
    rf_f1 = f1_score(y_test, y_pred_rf)
    rf_auc = roc_auc_score(y_test, rf_probs)
    print("\nRandom Forest Test Set Metrics:")
    print(f"Accuracy:  {rf_accuracy * 100:.2f}%")
    print(f"Precision: {rf_precision * 100:.2f}%")
    print(f"Recall:    {rf_recall * 100:.2f}%")
    print(f"F1 Score:  {rf_f1 * 100:.2f}%")
    print(f"ROC-AUC:   {rf_auc * 100:.2f}%")

    print("\nStep 9: Generating visualizations...")
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=y_train, hue=y_train, palette=["#2E7D32", "#C62828"], legend=False)
    plt.title("Class Distribution (Train Set)", fontsize=12, fontweight='bold', pad=10)
    plt.xlabel("Transaction Class (0 = Legitimate, 1 = Fraudulent)", fontsize=10)
    plt.ylabel("Count (Log Scale)", fontsize=10)
    plt.yscale('log')
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height):,}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=9, xytext=(0, 5),
                    textcoords='offset points')
    plt.tight_layout()
    plt.savefig("assets/fraud_distribution.png", dpi=150)
    plt.close()
    print("Saved assets/fraud_distribution.png")

    plt.figure(figsize=(7, 5.5))
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
    fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_probs)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
    
    auc_lr = roc_auc_score(y_test, lr_probs)
    auc_dt = roc_auc_score(y_test, dt_probs)
    auc_rf = roc_auc_score(y_test, rf_probs)

    plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.4f})', color='#FF9800', lw=2)
    plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {auc_dt:.4f})', color='#03A9F4', lw=2)
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.4f})', color='#E91E63', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing', alpha=0.5)
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate", fontsize=10)
    plt.ylabel("True Positive Rate", fontsize=10)
    plt.title("ROC Curve Comparison", fontsize=12, fontweight='bold', pad=10)
    plt.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    plt.savefig("assets/roc_curve.png", dpi=150)
    plt.close()
    print("Saved assets/roc_curve.png")

    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred_rf)
    labels = np.array([['True Legitimate\n(TN)', 'False Fraudulent\n(FP)'],
                       ['False Legitimate\n(FN)', 'True Fraudulent\n(TP)']])
    
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            annot[i, j] = f"{labels[i, j]}\n{cm[i, j]:,}"

    sns.heatmap(cm, annot=annot, fmt="", cmap="Greens", cbar=True,
                xticklabels=['Legitimate', 'Fraudulent'],
                yticklabels=['Legitimate', 'Fraudulent'],
                annot_kws={"fontsize": 10})
    plt.title("Random Forest Confusion Matrix", fontsize=12, fontweight='bold', pad=10)
    plt.xlabel("Predicted Label", fontsize=10)
    plt.ylabel("Actual Label", fontsize=10)
    plt.tight_layout()
    plt.savefig("assets/confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved assets/confusion_matrix.png")

    plt.figure(figsize=(8, 5))
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    top10 = feature_importance.head(10)
    sns.barplot(x='Importance', y='Feature', hue='Feature', data=top10, palette="viridis", legend=False)
    plt.title("Top 10 Important Features (Random Forest)", fontsize=12, fontweight='bold', pad=10)
    plt.xlabel("Relative Importance Score", fontsize=10)
    plt.ylabel("Feature", fontsize=10)
    plt.tight_layout()
    plt.savefig("assets/feature_importance.png", dpi=150)
    plt.close()
    print("Saved assets/feature_importance.png")

    plt.figure(figsize=(7, 4.5))
    plot_df = test_df[test_df['amt'] <= 1500].copy()
    plot_df['is_fraud_label'] = plot_df['is_fraud'].map({0: 'Legitimate', 1: 'Fraudulent'})
    
    sns.boxplot(x='is_fraud_label', y='amt', hue='is_fraud_label', data=plot_df, palette=["#2E7D32", "#C62828"], legend=False)
    plt.title("Transaction Amount Distribution: Legitimate vs. Fraudulent", fontsize=12, fontweight='bold', pad=10)
    plt.xlabel("Transaction Type", fontsize=10)
    plt.ylabel("Transaction Amount ($) (capped at $1,500)", fontsize=10)
    plt.tight_layout()
    plt.savefig("assets/amount_boxplot.png", dpi=150)
    plt.close()
    print("Saved assets/amount_boxplot.png")
    
    print("\nScript completed successfully!")

if __name__ == "__main__":
    main()
