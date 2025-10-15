# 📱 Mobile Price Classification Project  

###  Predicting Smartphone Price Ranges Using Machine Learning  

In today’s competitive smartphone market, pricing a new model correctly can make or break a brand’s success.  
This project aims to predict the **price range of mobile phones** based on their features — such as RAM, battery power, internal storage, and camera quality — using **machine learning models**.  

---

##  Project Overview  

The goal of this project is to help smartphone companies estimate the **optimal price segment** for new models by analyzing key technical specifications.  
By training multiple classification models on real-world mobile data, we identify which features influence price the most.  

---

##  Objectives  
- Perform **data cleaning and preprocessing** on mobile specifications data.  
- Apply **One-Hot Encoding** to handle categorical variables.  
- Train and evaluate multiple **classification algorithms** (e.g., Logistic Regression, Random Forest, Decision Tree).  
- Compare model performances using **accuracy, confusion matrix, and classification reports**.  
- Visualize the impact of features on mobile pricing.  

---

## Key Steps  

### 1️⃣ Data Preparation  
- Checked for missing values and duplicates.  
- Encoded categorical data using `pd.get_dummies()` and `LabelEncoder`.  
- Split the dataset into **training and testing sets**.  

### 2️⃣ Model Building  
- Implemented several models including:  
  - Logistic Regression  
  - Random Forest Classifier  
  - Decision Tree Classifier  
  - Support Vector Machine (SVM)  

### 3️⃣ Model Evaluation  
- Compared model performances using:  
  - **Accuracy Score**  
  - **Confusion Matrix**  
  - **Classification Report**  

### 4️⃣ Feature Importance  
- Identified which factors (like RAM, battery, etc.) most influence the predicted price range.

---

## Results & Insights  
- **RAM** was found to be the most influential feature in determining the price range.  
- Higher **battery power** and **storage** also contribute to premium pricing.  
- The **Random Forest Classifier** gave the highest accuracy among all tested models.  

---

##Technologies Used  
| Category | Tools & Libraries |
|-----------|------------------|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Environment | Jupyter Notebook |

---

##Sample Code Snippet  

```python
# Splitting data into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training a Random Forest model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluating performance
from sklearn.metrics import accuracy_score

predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
