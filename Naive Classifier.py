import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

file_path = '/content/drive/MyDrive/Machine/Dataset-CLP.csv'
df = pd.read_csv(file_path)
df.head()

combined_df= df = df.drop(df.columns[-1], axis=1)#last column delete
combined_df

print(combined_df.columns)  # List all column names

combined_df.rename(columns={'Type ': 'Type'}, inplace=True) 

combined_df = pd.read_csv('/content/drive/MyDrive/Machine/Dataset-CLP.csv')
print(combined_df.columns)

combined_df.iloc[:, 1] = combined_df.iloc[:, 1].map({"Sports News": 0, "Political News": 1})

print(combined_df.head())  # Check if 'Type' column now shows numerical values


from sklearn.model_selection import train_test_split

# Clean up the column names by stripping leading/trailing spaces
combined_df.columns = combined_df.columns.str.strip()

# Verify cleaned column names
print("Cleaned columns in combined_df:", combined_df.columns)

# Now, 'News Title' and 'Type' are the correct column names
N = combined_df['News Title']
C = combined_df['Type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(N, C, random_state=1)

# Print the training and testing sets
print("Training set (X_train):", X_train.head())
print("Testing set (X_test):", X_test.head())
print("Training labels (y_train):", y_train.head())
print("Testing labels (y_test):", y_test.head())

# Print sizes of the datasets
print("Original dataset contains", combined_df.shape[0], "News articles")
print("Training set contains", X_train.shape[0], "News articles")
print("Testing set contains", X_test.shape[0], "News articles")


from sklearn.feature_extraction.text import CountVectorizer

# Ensure that X_train and X_test contain valid text data (no NaN or missing values)
X_train = X_train.dropna()
X_test = X_test.dropna()

# Convert X_train and X_test to lists if they are pandas Series (if needed)
X_train = X_train.tolist()
X_test = X_test.tolist()

# Initialize the CountVectorizer
count_vector = CountVectorizer()

# Fit the model and transform the training data
X_train_vectorized = count_vector.fit_transform(X_train)

# Transform the test data using the fitted vectorizer
X_test_vectorized = count_vector.transform(X_test)

# Optional: print the shape of the resulting sparse matrices to verify
print("Shape of X_train_vectorized:", X_train_vectorized.shape)
print("Shape of X_test_vectorized:", X_test_vectorized.shape)


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Ensure y_train and y_test are 1D arrays of integers
y_train = y_train.astype(int)
y_test = y_test.astype(int)

naive_bayes = MultinomialNB()

naive_bayes.fit(X_train_vectorized, y_train)

predictions = naive_bayes.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy of the model: {accuracy * 100:.2f}%")

print("Predictions:", predictions[:10])
