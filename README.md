 **🚢 Titanic Survival Prediction – ML Pipeline Project**
This project builds a complete machine learning pipeline using the Titanic dataset. It covers everything from data cleaning to model evaluation and feature selection.

**📊 Dataset**
Source: [https://www.kaggle.com/datasets/yasserh/titanic-dataset]
Target Variable: "Survived" (0 = No, 1 = Yes)

**🛠️ Technologies Used**
1- Python
2- Pandas
3- NumPy
4- Seaborn & Matplotlib
5- Scikit-learn (sklearn)
**IDE:**
1-Jupyter Notebook or Google Collab (User's Choice)

**🔍 Steps Performed**
**1. Data Exploration**
•	Basic ".head()", ".describe()", and ".info()" checks
•	Null value inspection

**2. Data Cleaning** 
•	Filling missing values (Age & Embarked with mode)
•	Dropping duplicates
•	Handling categorical variables:
•	"Sex": Label Encoding
•	"Embarked": One-hot Encoding

**3. Outlier Detection & Handling**
•	"Fare": IQR method
•	"Age": Z-score filtering

**4. Feature Scaling**
•	Standardization (using “StandardScaler”)
•	Normalization (using “MinMaxScaler”)

**5. Train-Test Split**
•	"train_test_split()" with 80/20 ratio
•	Dropped irrelevant columns like “Name”, “Ticket”, “Cabin”, “PassengerId”

**6. Model Training**
•	"LogisticRegression()" used
•	Evaluated using:
1.	Accuracy
2.	Confusion matrix
3.	Classification report

**7. Feature Selection (Optional)**
•	Used Recursive Feature Elimination (RFE)
•	Selected top 5 features and retrained model

**📈 Final Evaluation**
•	Accuracy before RFE: "XX%" (replace with your result)
•	Accuracy after RFE: "YY%" (replace with your result)

 📂 **How to Run**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python titanic_model.py  # or run in Jupyter Notebook

📌 Notes
You can easily convert this script to a Jupyter Notebook for better visualization.
Extendable to other models like Random Forest, SVM, or XGBoost.

🤝 **Contributing**
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

📧 **Author**
Taimour Mushtaq
🎓 BSCS Student at Federal Urdu University of Arts,Science and Technology, Islamabad Pakistan
🔗 https://www.linkedin.com/in/taimourmushtaq/ |https://github.com/TAIMOURMUSHTAQ


