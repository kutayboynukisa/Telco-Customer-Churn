# Import dataset and define X
import pandas as pd
dataset = pd.read_csv('Data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
X = dataset.drop(columns=['customerID', 'Churn'], axis=1)

# pd.set_option('display.max_columns', None) # Display all columns

# Convert the TotalCharges feature from object to numeric, set non-convertible values as NaN
X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')

# Define categorical and numerical columns for label encoding, one-hot encoding, and scaling
categorical_features_onehot = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                               'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaymentMethod']
categorical_features_label = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'Contract', 'PaperlessBilling']
numerical_features = X.select_dtypes(exclude=['object']).columns

# Apply LabelEncoder to categorical columns, include StandardScaler for feature scaling
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
label_encoder = LabelEncoder()
for col in categorical_features_label:
    X[col] = label_encoder.fit_transform(X[col])

# Define ColumnTransformer and Pipeline to handle feature scaling, filling NaN values, and one-hot encoding
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values
            ('scaler', StandardScaler())                  # Apply scaling
        ]), numerical_features),
        ('cat_onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features_onehot)
    ]
)

# Apply a different label encoder to define y
label_encoder_2 = LabelEncoder()
y = label_encoder_2.fit_transform(dataset['Churn'])

# Split into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# Apply feature scaling, NaN filling, and one-hot encoding
X_train_scaled = ct.fit_transform(X_train)
X_test_scaled = ct.transform(X_test)

# Logistic Regression Model
# Train the model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10]}
grid_search = GridSearchCV(LogisticRegression(random_state=0, max_iter=1000), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
print(grid_search.best_params_)

# Test the model
y_pred = grid_search.predict(X_test_scaled)

# Evaluate the model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print(f"Model Accuracy Percentage: {accuracy*100:.0f}%")

# Create and visualize the Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
disp = ConfusionMatrixDisplay.from_estimator(grid_search, X_test_scaled, y_test)
plt.title("Confusion Matrix")
plt.show()

# Report for Logistic Regression
from sklearn.metrics import classification_report
print("\nClassification Report for Logistic Regression:\n",classification_report(y_test, y_pred))

# Calculate feature importance
import numpy as np
feature_importance = np.abs(grid_search.best_estimator_.coef_[0])
feature_names = ct.get_feature_names_out()
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df.sort_values(by='Importance', ascending=False, inplace=True)
print(importance_df)

# Bar Plot for Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance for Logistic Regression')
plt.gca().invert_yaxis()
plt.show()

# ROC-AUC Curve
from sklearn.metrics import roc_curve, auc
y_pred_prob = grid_search.best_estimator_.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()