from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

app = Flask(__name__)

asteroids = pd.read_csv("asteroids_for_modelling.csv")

features = ["absolute_magnitude", "albedo", "n", "rms"]
target = "diameter"

X = asteroids[features]
y = asteroids[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

random_forest = RandomForestRegressor(max_features='sqrt', n_jobs=-1)
random_forest.fit(X_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        absolute_magnitude = float(request.form['absolute_magnitude'])
        albedo = float(request.form['albedo'])
        n = float(request.form['n'])
        rms = float(request.form['rms'])
        
        input_data = scaler.transform([[absolute_magnitude, albedo, n, rms]])
        predicted_diameter = random_forest.predict(input_data)
        predicted_diameter = round(predicted_diameter[0], 2)
        
        return jsonify(predicted_diameter)
    else:
        return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)

