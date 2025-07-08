import pandas as pd

# Step 1: Load data
df = pd.read_csv("data/dataset.csv")
print("🟢 Raw shape:", df.shape)

# Step 2: Replace missing values with "None"
df.fillna("None", inplace=True)

# Step 3: Convert all symptoms to lowercase
for col in df.columns:
    df[col] = df[col].str.lower()

# Step 4: Get list of unique symptoms
symptoms = set()
for i in range(1, 18):  # Symptom_1 to Symptom_17
    symptoms.update(df[f"Symptom_{i}"].unique())

symptoms.discard("none")   # remove placeholder
symptoms = sorted(list(symptoms))

print(f"🩺 Unique symptoms found: {len(symptoms)}")

# Step 5: Create new dataframe with 1-hot encoded symptoms
encoded_df = pd.DataFrame(0, index=range(len(df)), columns=symptoms)

for i, row in df.iterrows():
    for j in range(1, 18):
        s = row[f"Symptom_{j}"]
        if s != "none":
            encoded_df.loc[i, s] = 1

# Step 6: Add target label (disease)
encoded_df["Disease"] = df["Disease"]

# Step 7: Save cleaned dataset
encoded_df.to_csv("symptoms_clean.csv", index=False)
print("✅ Cleaned data saved to symptoms_clean.csv")
