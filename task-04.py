# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File paths for training and validation datasets
train_file_path = "C:/Users/shrey/OneDrive/Desktop/Task-04/archive (3)/twitter_training.csv"
validation_file_path = "C:/Users/shrey/OneDrive/Desktop/Task-04/archive (3)/twitter_validation.csv"

# Load the training dataset
training_data = pd.read_csv(train_file_path, header=None, names=['Tweet_ID', 'Topic', 'Sentiment', 'Tweet'])

# Load the validation dataset
validation_data = pd.read_csv(validation_file_path, header=None, names=['Tweet_ID', 'Topic', 'Sentiment', 'Tweet'])

# Display the first few rows of the training dataset to understand its structure
print("Training Dataset:")
print(training_data.head())

# Display the first few rows of the validation dataset to understand its structure
print("\nValidation Dataset:")
print(validation_data.head())

# Perform sentiment analysis and visualize sentiment patterns for training data

# Analyzing sentiment distribution in training data
train_sentiment_counts = training_data['Sentiment'].value_counts()

# Visualizing sentiment distribution in training data
plt.figure(figsize=(8, 6))
sns.barplot(x=train_sentiment_counts.index, y=train_sentiment_counts.values)
plt.title('Sentiment Distribution in Training Data')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Perform sentiment analysis and visualize sentiment patterns for validation data

# Analyzing sentiment distribution in validation data
validation_sentiment_counts = validation_data['Sentiment'].value_counts()

# Visualizing sentiment distribution in validation data
plt.figure(figsize=(8, 6))
sns.barplot(x=validation_sentiment_counts.index, y=validation_sentiment_counts.values)
plt.title('Sentiment Distribution in Validation Data')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
