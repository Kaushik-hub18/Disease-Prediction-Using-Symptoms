import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Load cleaned data
df = pd.read_csv("symptoms_clean.csv")
print("🟢 Data loaded:", df.shape)

# Step 2: Separate features and target
X = df.drop("Disease", axis=1)
y = df["Disease"]

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Step 4: Train the model
model = MultinomialNB()
model.fit(X_train, y_train)
print("✅ Model trained.")

# Step 5: Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"📊 Accuracy: {acc*100:.2f}%")

# Step 6: Save the model
joblib.dump(model, "model.pkl")
print("💾 Model saved to model.pkl")
