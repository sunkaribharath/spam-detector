# Import necessary libraries
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import pandas as pd

# Load the dataset
df = pd.read_csv('spam_ham_dataset.csv')

# Data Preprocessing
X = df['text']
y = df['label_num']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Create Flask app
app = Flask(__name__)

# Define route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for predicting spam
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        prediction = model.predict([text])[0]
        result = "Spam" if prediction == 1 else "Not Spam"
        return render_template('index.html', text=text, prediction=result)
# Run the app
if __name__ == '__main__':
    app.run(debug=True)
