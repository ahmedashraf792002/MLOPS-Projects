import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pk
import os
from PIL import Image
# Libraries for Machine Learning
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score, f1_score, precision_score, recall_score,roc_curve, roc_auc_score


###############Load Data##############
data_path = os.path.join(os.getcwd(), 'data', 'Liver_disease_data.csv')
data = pd.read_csv(data_path)
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
confusion_matrix_paths = []
roc_curve_paths = []
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
    plt.savefig(f'{model_name}_Roc_Curve.png', bbox_inches='tight', dpi=300)
    plt.close()
    roc_curve_paths.append(f'./{model_name}_Roc_Curve.png')
    # Confusion matrix and classification report
    cm = confusion_matrix(y_test, y_test_pred)
    # Plot confusion matrix as a heatmap
    plt.figure(figsize=(8, 5))
    sns.heatmap(cm, annot=True, fmt='g', cmap="Blues", xticklabels=['NO', 'YES'], yticklabels=['NO', 'YES'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.savefig(f'Confusion_Matrix_for_{model_name}.png', bbox_inches='tight', dpi=300)
    plt.close()
    confusion_matrix_paths.append(f'./Confusion_Matrix_for_{model_name}.png')
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
    os.makedirs(os.path.join(os.getcwd(), 'models'), exist_ok=True)
    # Save the model
    pk.dump(model, open(os.path.join(os.getcwd(), 'models', f'{model_name}.pkl'), 'wb'))
        
parameters = {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 200, 'subsample': 0.8}
parameters["class_weight"] = class_weight_dict

train_and_evaluate_model(XGBClassifier(**parameters,random_state=42), 'XGBClassifier without Class-Weight', X_train, y_train, X_test, y_test)
train_and_evaluate_model(XGBClassifier(**parameters,random_state=42), 'XGBClassifier with Class-Weight', X_train, y_train, X_test, y_test)

## Load and plot each confusion matrix
plt.figure(figsize=(15, 5))  # Adjust figure size as needed
for i, path in enumerate(confusion_matrix_paths, 1):
    img = Image.open(path)
    plt.subplot(1, len(confusion_matrix_paths), i)
    plt.imshow(img)
    plt.axis('off')  # Disable axis for cleaner visualization

clf_name = "XGB Classifier Model"
## Save combined plot locally
plt.suptitle(clf_name, fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'conf_matrix.png', bbox_inches='tight', dpi=300)

## Delete old image files
for path in confusion_matrix_paths:
    os.remove(path)


## Load and plot each roc curve
plt.figure(figsize=(15, 5))  # Adjust figure size as needed
for i, path in enumerate(roc_curve_paths, 1):
    img = Image.open(path)
    plt.subplot(1, len(roc_curve_paths), i)
    plt.imshow(img)
    plt.axis('off')  # Disable axis for cleaner visualization

## Save combined plot locally
plt.suptitle(clf_name, fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'roc_curve.png', bbox_inches='tight', dpi=300)

## Delete old image files
for path in roc_curve_paths:
    os.remove(path)