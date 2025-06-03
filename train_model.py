import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pickle


# Load data
df = pd.read_csv("data/train.csv")
df = df[["Survived", "Pclass", "Sex", "Age", "Fare", "SibSp", "Parch"]].dropna()

# Encode categorical variables
df["Sex"] = LabelEncoder().fit_transform(df["Sex"])

# Train-test split
X = df.drop("Survived", axis=1)
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
with open("titanic_model.pkl", "wb") as f:
    pickle.dump(model, f)
