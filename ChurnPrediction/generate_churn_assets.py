import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

def main():
    print("Step 1: Loading Churn dataset...")
    df = pd.read_csv("Churn_Modelling.csv")
    
    df_clean = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    df_clean['BalanceSalaryRatio'] = df_clean['Balance'] / (df_clean['EstimatedSalary'] + 1)
    df_clean['IsSenior'] = (df_clean['Age'] > 50).astype(int)
    
    df_eda = df_clean.copy()
    
    df_clean = pd.get_dummies(df_clean, columns=['Geography', 'Gender'], drop_first=True)
    
    X = df_clean.drop('Exited', axis=1)
    y = df_clean['Exited']
    X = X.astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Step 2: Training benchmark models...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
    
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    dt_prob = dt.predict_proba(X_test)[:, 1]
    
    print("Step 3: Loading Random Forest model...")
    rf = joblib.load("best_model.pkl")
    rf_prob = rf.predict_proba(X_test)[:, 1]
    rf_pred = rf.predict(X_test)
    
    os.makedirs("assets", exist_ok=True)
    
    print("Step 4: Generating visualizations...")
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=y, hue=y, data=df_clean, palette=["#2E7D32", "#C62828"], legend=False)
    plt.title("Customer Churn Distribution", fontsize=12, fontweight='bold', pad=10)
    plt.xlabel("Customer Class (0 = Retained, 1 = Churned)", fontsize=10)
    plt.ylabel("Count", fontsize=10)
    
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height):,}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=9, xytext=(0, 5),
                    textcoords='offset points')
    plt.tight_layout()
    plt.savefig("assets/churn_distribution.png", dpi=150)
    plt.close()
    print("Saved assets/churn_distribution.png")

    plt.figure(figsize=(7, 5.5))
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
    fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_prob)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
    
    auc_lr = roc_auc_score(y_test, lr_prob)
    auc_dt = roc_auc_score(y_test, dt_prob)
    auc_rf = roc_auc_score(y_test, rf_prob)

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
    cm = confusion_matrix(y_test, rf_pred)
    labels = np.array([['True Retained\n(TN)', 'False Churned\n(FP)'],
                       ['False Retained\n(FN)', 'True Churned\n(TP)']])
    
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            annot[i, j] = f"{labels[i, j]}\n{cm[i, j]:,}"

    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=True,
                xticklabels=['Retained', 'Churned'],
                yticklabels=['Retained', 'Churned'],
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
        'Feature': X.columns,
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
    df_eda['Exited_Label'] = df_eda['Exited'].map({0: 'Retained', 1: 'Churned'})
    sns.boxplot(x='Exited_Label', y='Balance', hue='Exited_Label', data=df_eda, palette=["#2E7D32", "#C62828"], legend=False)
    plt.title("Account Balance Distribution by Churn Status", fontsize=12, fontweight='bold', pad=10)
    plt.xlabel("Customer Status", fontsize=10)
    plt.ylabel("Account Balance ($)", fontsize=10)
    plt.tight_layout()
    plt.savefig("assets/balance_boxplot.png", dpi=150)
    plt.close()
    print("Saved assets/balance_boxplot.png")
    
    print("All assets generated successfully!")

if __name__ == "__main__":
    main()
