from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import joblib

app = Flask(__name__)

random_forest = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
X_test, y_test = joblib.load('test_set.pkl')

rf_predictions = random_forest.predict(X_test)

mae = mean_absolute_error(y_test, rf_predictions)
mae = round(mae, 2)

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

        lower_bound = round(predicted_diameter - mae, 2)
        upper_bound = round(predicted_diameter + mae, 2)
        
        return jsonify({
            'predicted_diameter': predicted_diameter,
            'mae': mae,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        })
    else:
        return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)

