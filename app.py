from flask import Flask, render_template, request, redirect, session
from pymongo import MongoClient
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Connect to MongoDB
client = MongoClient("mongodb+srv://waleed:waleed123@cluster0.aqwbdzl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client['HousePriceApp']
users_col = db['users']
history_col = db['search_history']

# Initialize model
model = LinearRegression()

# Train model using CSV + MongoDB history
def train_model():
    X, y = [], []

    # Load base data from CSV
    if os.path.exists("initial_housing_data.csv"):
        df = pd.read_csv("initial_housing_data.csv")
        X.extend(df[['bedrooms', 'size', 'bathrooms', 'garage']].values.tolist())
        y.extend(df['price'].values.tolist())

    # Load historical data from MongoDB
    data = list(history_col.find())
    for d in data:
        X.append(d['features'])
        y.append(d['price'])

    if X and y:
        model.fit(X, y)

# Train model at startup
train_model()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_id = request.form['user_id']
        password = request.form['password']
        if users_col.find_one({'user_id': user_id}):
            return "User ID already exists."
        users_col.insert_one({'user_id': user_id, 'password': password})
        return redirect('/login')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form['user_id']
        password = request.form['password']
        user = users_col.find_one({'user_id': user_id, 'password': password})
        if user:
            session['user_id'] = user_id
            return redirect('/predict')
        return "Invalid credentials."
    return render_template('login.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        return redirect('/login')
    prediction = None
    if request.method == 'POST':
        try:
            bedrooms = float(request.form['bedrooms'])
            size = float(request.form['size'])
            bathrooms = float(request.form['bathrooms'])
            garage = float(request.form['garage'])
            features = [bedrooms, size, bathrooms, garage]
            price = model.predict([features])[0]
            prediction = round(price, 2)

            # Save user search to history
            history_col.insert_one({
                'user_id': session['user_id'],
                'features': features,
                'price': price
            })

            # Retrain model after new data
            train_model()
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template('predict.html', prediction=prediction)

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect('/login')
    user_history = history_col.find({'user_id': session['user_id']})
    return render_template('history.html', history=user_history)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
