import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
# Libraries for Machine Learning
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score, f1_score, precision_score, recall_score,roc_curve, roc_auc_score


###############Load Data##############
data = pd.read_csv("Liver_disease_data.csv")
######## Correlation ############
corr = data.corr()
diag_corr = corr['Diagnosis'].drop('Diagnosis')
plt.figure(figsize=(15, 6))
diag_corr_df = pd.DataFrame(diag_corr)
sns.heatmap(diag_corr_df, annot=True, fmt=".2f", cmap='Set1', 
            linecolor='gray', center=0)
plt.title('Correlation of Features with Diagnosis', fontsize=16)
## Save the plot locally
plt.savefig('Correlation.png', bbox_inches='tight', dpi=300)
plt.close()

############ input - output #############
X = data.drop(['Diagnosis'],axis=1)
y = data['Diagnosis']
keys = X.columns
### Scaling
scale = MinMaxScaler()
X = scale.fit_transform(X)
X = pd.DataFrame(X,columns=keys)
# Calculate class weights
class_weights = compute_class_weight(
class_weight='balanced', 
classes=np.unique(y), 
y=y
)
class_weight_dict = dict(zip(np.unique(y), class_weights))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
## Clear metrics.txt file at the beginning
with open('metrics.txt', 'w') as f:
    pass
def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    # Train the model
    model.fit(X_train, y_train)
    # Predictions for training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    # Training metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, zero_division=1)
    train_recall = recall_score(y_train, y_train_pred, zero_division=1)
    train_f1 = f1_score(y_train, y_train_pred, zero_division=1)
    # Test metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=1)
    test_recall = recall_score(y_test, y_test_pred, zero_division=1)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=1)
    # Plot ROC curve if available
    y_test_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    roc_auc = roc_auc_score(y_test, y_test_prob)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc='lower right')
    plt.savefig(f'{model_name}-Roc Curve.png', bbox_inches='tight', dpi=300)
    plt.close()
    # Confusion matrix and classification report
    cm = confusion_matrix(y_test, y_test_pred)
    # Plot confusion matrix as a heatmap
    plt.figure(figsize=(8, 5))
    sns.heatmap(cm, annot=True, fmt='g', cmap="Blues", xticklabels=['NO', 'YES'], yticklabels=['NO', 'YES'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.savefig(f'Confusion Matrix for {model_name}.png', bbox_inches='tight', dpi=300)
    plt.close()
    # Print the classification report and confusion matrix
    classification_report_str = classification_report(y_test, y_test_pred, target_names=['NO', 'YES'])
    # Print evaluation metrics
    with open('metrics.txt', 'a') as f:
        f.write(f'{model_name}\n')
        f.write(f"{model_name} - Training Accuracy: {train_accuracy:.3f} %\n")
        f.write(f"{model_name} - Test Accuracy: {test_accuracy:.3f} %\n")
        f.write(f"{model_name} - Training Precision: {train_precision:.3f} %\n")
        f.write(f"{model_name} - Test Precision: {test_precision:.3f} %\n")
        f.write(f"{model_name} - Training Recall: {train_recall:.3f} %\n")
        f.write(f"{model_name} - Test Recall: {test_recall:.3f} %\n")
        f.write(f"{model_name} - Training F1 Score: {train_f1:.3f} %\n")
        f.write(f"{model_name} - Test F1 Score: {test_f1:.3f} %\n")
        f.write(f"{model_name} - Confusion Matrix:\n {cm} \n")
        f.write(f"{model_name} - Classification Report:\n {classification_report_str} \n")
        f.write('----'*10 + '\n')
        
parameters = {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 200}

train_and_evaluate_model(RandomForestClassifier(**parameters,random_state=42), 'RandomForest-without-Class_Weight', X_train, y_train, X_test, y_test)
train_and_evaluate_model(RandomForestClassifier(**parameters,class_weight=class_weight_dict,random_state=42), 'RandomForest-with-Class_Weight', X_train, y_train, X_test, y_test)