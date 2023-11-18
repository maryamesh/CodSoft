# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
dataset_path='C:/Users/dell/Desktop/fraudTrain.csv'
fraud_data = pd.read_csv(dataset_path)

# Display basic information about the dataset
print(fraud_data.info())

# Summary statistics for numerical columns
print(fraud_data.describe())

# Check for missing values
print(fraud_data.isnull().sum())
import seaborn as sns


# Explore categorical variables
categorical_columns = ['merchant', 'category', 'gender','city']

for column in categorical_columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=fraud_data, x=column, palette='Set3')
    plt.title(f'Distribution of {column}')
    plt.xticks(rotation=45)
    plt.show()

# Convert 'trans_date_trans_time' and 'dob' to datetime
fraud_data['trans_date_trans_time'] = pd.to_datetime(fraud_data['trans_date_trans_time'])
fraud_data['dob'] = pd.to_datetime(fraud_data['dob'])

# Time-based analysis
plt.figure(figsize=(12, 5))
fraud_data['trans_date_trans_time'].dt.hour.plot(kind='hist', bins=24, rwidth=0.9, color='skyblue')
plt.title('Hourly Transaction Distribution')
plt.xlabel('Hour of the Day')
plt.ylabel('Frequency')
plt.show()

# Visualize the class distribution
plt.figure(figsize=(6, 4))
fraud_data['is_fraud'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Class Distribution (0: Non-Fraud, 1: Fraud)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Geospatial data - Scatter plot of transactions
plt.figure(figsize=(10, 8))
plt.scatter(fraud_data['merch_long'], fraud_data['merch_lat'], c=fraud_data['is_fraud'], cmap='coolwarm', alpha=0.5)
plt.title('Geospatial Distribution of Transactions (Fraud vs. Non-Fraud)')
plt.xlabel('Merchant Longitude')
plt.ylabel('Merchant Latitude')
plt.colorbar(label='0: Non-Fraud, 1: Fraud')
plt.show()
