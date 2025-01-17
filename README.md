## Telco Customer Churn Project

### Updated Documentation: Logistic Regression Model and Evaluation Phase

---

### Model Selection and Training

#### 1. Model Selection
Before choosing Logistic Regression, several models were evaluated, including Random Forest, XGBoost, and Support Vector Machine. 
- **Evaluation Metric:** Accuracy score was used to compare model performances.
- **Reason for Choosing Logistic Regression:** Logistic Regression outperformed other models in accuracy, achieving a score of 81.12%, which was the highest among the tested models.

#### 2. Hyperparameter Tuning
- GridSearchCV was used to optimize the regularization parameter C for Logistic Regression.
- Parameters tested: {'C': [0.1, 1, 10]}.
- Best parameter: C = 0.1.

#### 3. Model Training
The Logistic Regression model was trained on the preprocessed training dataset using the optimal hyperparameters obtained from GridSearchCV.

---

### Model Evaluation

#### 1. Accuracy and Classification Report
- The model achieved an accuracy score of **81.12%** on the test dataset.
- A detailed classification report was generated, including precision, recall, F1-score, and support for each class.

#### 2. Confusion Matrix
- The confusion matrix was visualized to better understand the model's performance in distinguishing between churned and non-churned customers.

#### 3. Feature Importance
- **Why Use np.abs for Coefficients?**
  - The absolute values of the Logistic Regression coefficients were used to evaluate feature importance. This approach focuses on the magnitude of the effect, regardless of whether it is positive or negative.
  - Features with higher absolute coefficients have a stronger impact on churn prediction.
- **Visualization:**
  - Feature importance was visualized using a horizontal bar plot to identify the most influential features.

#### 4. ROC-AUC Curve
- The Receiver Operating Characteristic (ROC) curve was plotted to evaluate the model's performance at various classification thresholds.
- **AUC Value:** The model achieved an AUC score of **0.85**, indicating strong discriminatory power.

---

### Key Takeaways from the Logistic Regression Phase

1. **Model Selection:** Logistic Regression was chosen based on its superior accuracy score compared to other models.
2. **Feature Analysis:**
   - The analysis highlighted the features most associated with churn, providing insights for targeted business strategies.
   - The absolute values of coefficients allowed for an unbiased comparison of feature importance.
3. **Visualization:**
   - Confusion matrix, feature importance bar plot, and ROC-AUC curve provided a comprehensive view of model performance.
4. **Performance Summary:**
   - Accuracy: 81.12%
   - AUC: 0.85

---