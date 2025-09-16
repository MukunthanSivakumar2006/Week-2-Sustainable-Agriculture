"""
EDA Script for Crop Recommendation System - Week 2
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/crop_recommendation.csv")

# Display first few rows
print("Dataset Preview:")
print(df.head())

# Dataset info
print("\nDataset Info:")
print(df.info())

print("\nDataset Description:")
print(df.describe())

# Crop distribution
plt.figure(figsize=(12,5))
sns.countplot(data=df, x="label", order=df['label'].value_counts().index)
plt.xticks(rotation=90)
plt.title("Crop Distribution")
plt.show()


