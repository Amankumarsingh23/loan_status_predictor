
# 💰 Loan Status Prediction using Machine Learning

A Machine Learning project that predicts whether a **loan application** will be approved or not based on applicant details.  
Built with **Support Vector Machine (SVM)** using the Kaggle Loan Prediction dataset.

---

## 📌 Problem Statement

Loan approval is a critical process for banks and financial institutions.  
By leveraging historical loan application data, we can predict whether a new loan application will be **approved** or **rejected**, reducing manual effort and improving efficiency.

---

## 📂 Dataset Overview

- **Source**: [Kaggle - Loan Prediction Dataset](https://www.kaggle.com/datasets/ninzaami/loan-predication)
- **Size**: ~614 entries, 13 columns
- **Features**:
  - Gender, Married, Dependents, Education, Self_Employed
  - ApplicantIncome, CoapplicantIncome
  - LoanAmount, Loan_Amount_Term, Credit_History
  - Property_Area
- **Target**: Loan_Status (`1` = Approved, `0` = Not Approved)

---

## ⚙️ Tech Stack

- **Python**
- **Pandas**, **NumPy**
- **Seaborn** (visualization)
- **Scikit-learn**
  - Support Vector Machine (SVM)
  - Train-Test Split
  - Accuracy Score

---

## 🚀 Workflow

1. **Data Collection**
   - Load dataset from Kaggle via `kagglehub`
2. **Data Preprocessing**
   - Handle missing values
   - Encode categorical variables into numerical form
   - Replace `3+` dependents with `4`
3. **Data Visualization**
   - Analyze relationships between education/marital status and loan status
4. **Model Training**
   - Use Support Vector Classifier (Linear Kernel)
5. **Model Evaluation**
   - Calculate accuracy on training & test datasets
6. **Prediction System**
   - Takes applicant details as input and outputs loan approval status

---

## 📊 Model Performance

| Dataset       | Accuracy |
|---------------|----------|
| Training Set  | ~85%     |
| Test Set      | ~83%     |

---

## 💡 Sample Prediction

```python
input_data = (1.0, 1.0, 0, 1, 1.0, 0, 0.0, 9966.0, 90807360.0, 1.0, 2)
input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

prediction = classifier.predict(input_data_as_numpy_array)

if prediction[0] == 0:
    print("The loan is not approved")
else:
    print("The loan is approved")
Output Example:

csharp
Copy
Edit
The loan is approved
📥 Installation
bash
Copy
Edit
git clone https://github.com/yourusername/loan-status-prediction.git
cd loan-status-prediction
pip install -r requirements.txt
Run the script:

bash
Copy
Edit
python loan_status_prediction.py
🔮 Future Improvements
Try advanced models like Random Forest, XGBoost, or Neural Networks

Implement feature scaling & hyperparameter tuning

Create a web app using Streamlit or Flask

Add model persistence with joblib or pickle

📄 License
This project is licensed under the MIT License – see the LICENSE file for details.

🙌 Acknowledgments
Kaggle Dataset Provider

Scikit-learn Documentation

Open-source community

💬 Contact
Made with ❤️ by Aman Kumar Singh
Contributions and suggestions are welcome!
