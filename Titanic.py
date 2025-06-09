#Importing important Libraries 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder,MinMaxScaler

#Loading and reading dataset
dataset=pd.read_csv("C:\\Users\\LENOVO\\Downloads\\TitanicSurvival\\Titanic-Dataset.csv") 
dataset.head(4)

#Describing data
dataset.describe()

#Getting info\an overview of entire data
dataset.info()

#Checking for null values
dataset.isnull().sum()

#Droping null columns of Embarked as it has less null values
# dataset['Embarked']=dataset['Embarked'].dropna()
dataset['Embarked']=dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])

#Filling null values of age and embarked with Mode
dataset["Age"]=dataset["Age"].fillna(dataset['Age'].mode()[0])
dataset['Embarked']=dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])

#Label Encoding on Sex and OneHotEncoding on Embarked
#Label encoding on Sex
le=LabelEncoder()
dataset['Sex'] = le.fit_transform(dataset['Sex'])
# le.fit(dataset['Sex'])
# le.transform(dataset['Sex'])
dataset['Sex'].unique()
dataset['Embarked'].unique() 
#One hot encoding on Embarked 
oh=OneHotEncoder()
dataset = pd.get_dummies(dataset, columns=['Embarked'], drop_first=True)
# oh.fit_transform([["Embarked"]])

#Outliers handling of Fare via IQR
import matplotlib.pyplot as plt
sns.boxplot(x="Fare",data=dataset)
plt.show()
sns.histplot(dataset[["Fare"]],kde=True,color="Green")
plt.show()
Q1=dataset['Fare'].quantile(0.25)
Q3=dataset['Fare'].quantile(0.75)
IQR=Q3-Q1
min_range=Q1-(1.5*IQR)
max_range=Q3+(1.5*IQR)
min_range,max_range
IQR
new_dataset=dataset[dataset['Fare']<=max_range]
dataset.shape
new_dataset.shape
sns.boxplot(x="Fare",data=new_dataset)
plt.show() 
sns.histplot(new_dataset[["Fare"]],kde=True,color="Green")
plt.show()
dataset['Fare'].value_counts()

#Outliers handling of age using Z-Score
z_score=((dataset['Age']-dataset['Age'].mean())/(dataset['Age'].std()))
# dataset["z_score"]=z_score
# z_score
dataset = dataset[abs(z_score) < 3]
dataset
sns.histplot(dataset[["Age"]],kde=True,color="Green")
plt.show() 
sns.boxplot(x="Age",data=dataset)
plt.show() 

# dataset.duplicated()
print(f"Duplicate Rows: {dataset.duplicated().sum()}")
dataset = dataset.drop_duplicates()

# Feature Scaling
# Using StandardScaler on Age and Fare
ss=StandardScaler()
ss.fit(dataset[['Age']])
dataset["Age_ss"]=pd.DataFrame(ss.transform(dataset[['Age']]),columns=['x'])
dataset["Fare_ss"] = ss.fit_transform(dataset[["Fare"]])
plt.subplot(1,2,1)
plt.title("Before")
sns.histplot(data=dataset['Age'])
plt.subplot(1,2,2)
plt.title("After")
sns.histplot(data=dataset['Age_ss'])
plt.show()
dataset
#Using MinMaxSclar on Age and Fare for Normalization
mm = MinMaxScaler()
dataset[['Age_nm', 'Fare_nm']] = mm.fit_transform(dataset[['Age', 'Fare']])

#Train Test split on  Survived 
from sklearn.model_selection import train_test_split
# Drop unnecessary columns (e.g., PassengerId, Name, Ticket, Cabin if present)
features = dataset.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId'], errors='ignore')
target = dataset['Survived']
# Split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Model Selection (Linear Model) 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
# Predict
y_pred = model.predict(X_test)
# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Optional Feature Elimination (Recursive) 
from sklearn.feature_selection import RFE
rfe = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=5)
rfe.fit(X_train, y_train)
selected_features = X.columns[rfe.support_]
print("Selected Features:", selected_features)
#Now use only selected features: 
X_train_rfe = X_train[selected_features]
X_test_rfe = X_test[selected_features]
model.fit(X_train_rfe, y_train)
y_pred_rfe = model.predict(X_test_rfe)
print("Accuracy with selected features:", accuracy_score(y_test, y_pred_rfe))


