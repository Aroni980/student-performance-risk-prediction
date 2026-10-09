import pandas as pd

# Load the UCI Student Performance dataset
df = pd.read_csv("data/student-mat.csv", sep=";")

# Create the target variable
# 1 = At Risk, 0 = Not At Risk
df["at_risk"] = (df["G3"] < 10).astype(int)

# Save the processed dataset
df.to_csv("data/student_performance_processed.csv", index=False)

print("Dataset processed successfully!")
print("Shape:", df.shape)
print("\nAt-risk distribution:")
print(df["at_risk"].value_counts())