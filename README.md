# Data Preprocessing Phase - Telco Customer Churn Project

This document outlines the steps performed during the data preprocessing phase of the Telco Customer Churn project. Each step is explained in detail, including the reasoning behind key decisions.

## **1. Dataset Import and Initial Setup**
- **Dataset:** The dataset was imported from a CSV file named `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
- **Features and Target:**
  - Features (`X`): All columns except `customerID` and `Churn`.
  - Target (`y`): The `Churn` column, indicating whether a customer churned (Yes) or not (No).

## **2. Data Cleaning**
- **Conversion of `TotalCharges`:**
  - The `TotalCharges` column was initially of `object` type, even though it represents numerical data.
  - It was converted to `float` using `pd.to_numeric`, with non-convertible values replaced by `NaN`.
  - Missing values in `TotalCharges` were filled using the column's mean.

## **3. Feature Categorization**
- **Categorical Features:**
  - **One-Hot Encoded Features:** Nominal (unordered) categorical columns such as `gender`, `MultipleLines`, `InternetService`, etc.
  - **Label Encoded Features:** Ordinal (ordered) or binary categorical columns such as `SeniorCitizen`, `Partner`, and `Contract`.
- **Numerical Features:**
  - Continuous numerical columns such as `tenure`, `MonthlyCharges`, and `TotalCharges`.

## **4. Encoding and Scaling**
- **Why `LabelEncoder` for Ordinal Features?**
  - Ordinal features, such as `Contract` (Month-to-month, One year, Two year), have a meaningful order.
  - Using `LabelEncoder` ensures that the ordinal relationship is preserved.
  - For binary categorical columns, `LabelEncoder` simplifies the encoding process.
- **Why `OneHotEncoder` for Nominal Features?**
  - Nominal features have no intrinsic order, so they are one-hot encoded to ensure equal treatment of all categories.
- **Scaling Numerical Features:**
  - Numerical columns were standardized using `StandardScaler` to ensure all features have a mean of 0 and a standard deviation of 1.

## **5. Missing Values**
- **Handling Missing Data:**
  - Missing values in numerical features were filled with the mean using `SimpleImputer`.
  - This approach is simple yet effective for continuous data, especially when the number of missing values is small.

## **6. Splitting the Dataset**
- The dataset was split into training and test sets using an 80-20 ratio.
- Stratification was applied to ensure the target variable's distribution remains consistent across both sets.

## **7. Use of ColumnTransformer and Pipeline**
- A `ColumnTransformer` was used to streamline preprocessing steps for numerical and nominal features:
  - **Numerical Pipeline:** Imputation and scaling were combined using `SimpleImputer` and `StandardScaler`.
  - **Categorical Pipeline:** One-hot encoding was applied to nominal features.
- This modular approach improves code clarity and maintainability.

---

### **Key Takeaways**
1. **Choice of Encoders:**
   - `LabelEncoder` was chosen for ordinal and binary features to preserve meaningful order.
   - `OneHotEncoder` was used for nominal features to treat categories equally.
2. **Scaling:**
   - Standardizing numerical features ensures that all features contribute equally to the model.
3. **Efficiency:**
   - Using `ColumnTransformer` and pipelines simplifies preprocessing and ensures reproducibility.

The data preprocessing phase has prepared the dataset for the next steps: model building and evaluation. If further preprocessing is needed during model tuning, adjustments can be made without disrupting the current structure.
