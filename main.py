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
