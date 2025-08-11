
# Insurance Charges Prediction using Regression

## 📌 Project Overview
This project predicts medical insurance charges based on several features such as age, sex, BMI, children, smoking habits, and region.

Multiple regression models were tested and compared using **R² Score** as the main metric.

**Best Model:** Support Vector Machine (SVM) Regression with R² = **0.86**.

---

## 📊 Dataset
- The dataset contains **N rows** and **M columns**.
- Features: Age, Sex, BMI, Children, Smoker, Region
- Target: Charges (Continuous value)

---

## 🔍 Models Evaluated

| Model                   | R² Score |
|-------------------------|----------|
| Linear Regression       | 0.78     |
| Support Vector Machine  | 0.86     |
| Decision Tree           | 0.73     |
| Random Forest           | 0.853    |

---

## ⚙️ Requirements
Install dependencies:
```
pip install -r requirements.txt
```

---

## ▶️ Running the Code
```
python insurance_regression.py
```

---

## 📈 Results
- **Best Model**: SVM with RBF Kernel (`C=3000`)
- Random Forest also performed well with `max_depth` tuning.
- All results are saved in the console output.

---

## 📜 License
This project is licensed under the MIT License.
```

***

### **2️⃣ insurance_regression.py**
```python
# insurance_regression.py
# Author: Your Name
# Description: Predict insurance charges using multiple regression models.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# -------------------
# 1. Load the dataset
# -------------------
df = pd.read_csv("data/insurance.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# -------------------
# 2. Encode categorical variables
# -------------------
categorical_cols = ['sex', 'smoker', 'region']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# -------------------
# 3. Split data
# -------------------
X = df.drop("charges", axis=1)
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------
# 4. Model Training
# -------------------
models = {
    "Linear Regression": LinearRegression(),
    "SVM (RBF, C=3000)": SVR(kernel='rbf', C=3000),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(max_depth=10, n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    if "SVM" in name:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    results[name] = r2
    print(f"{name} R² Score: {r2:.4f}")

# -------------------
# 5. Final Recommendation
# -------------------
best_model = max(results, key=results.get)
print("\nBest Model:", best_model, "with R² =", results[best_model])
```

***

### **3️⃣ requirements.txt**
```
pandas
scikit-learn
```

***
