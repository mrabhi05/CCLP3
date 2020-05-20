# To build a model to accurately classify a piece of news as REAL or FAKE.

# Data Preprocessing 
import numpy as np
import pandas as pd

# Import the dataset
dataset = pd.read_csv('news.csv')
X = dataset.iloc[:,2]
y = dataset.iloc[:,-1]

# Splitting the dataset into Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 0)

# TfidfVectorizer
# Initialize 
from sklearn.feature_extraction.text import TfidfVectorizer
TDfVector = TfidfVectorizer(stop_words = 'english', max_df = 0.7)
# Fitting the 
TDfVector_train = TDfVector.fit_transform(X_train)
TDfVector_test = TDfVector.transform(X_test)

# PassiveAgressiveClassifer 
#initialize
from sklearn.linear_model import PassiveAggressiveClassifier
PAClassifier = PassiveAggressiveClassifier(max_iter=50)
PAClassifier.fit(TDfVector_train,y_train)

# Prediction 
y_pred = PAClassifier.predict(TDfVector_test)

#Accuracy in %
from sklearn.metrics import accuracy_score, confusion_matrix
Score = accuracy_score(y_test,y_pred)
final = Score*100
print(f'Accuracy Score : {round(final)}%')
# Confusion Matrix
cm = confusion_matrix(y_test,y_pred)


