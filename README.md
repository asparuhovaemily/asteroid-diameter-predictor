
# Diameter Predictive Tool using Machine Learning

## Overview

The objective of this project is to develop an asteroid diameter predictive tool using Machine Learning.

After thorough exploration and experimentation, the most effective machine learning model was effectively incorporated into a user-friendly web application.


## File Descriptions

- `static/`: This directory contains static files for the web application, such as CSS stylesheets or images.

- `templates/`: Here you'll find the HTML template used for rendering the web application's page.

- `app.py`: This Python file contains the code for the web application.

- `asteroids_for_modelling.csv`: CSV file containing the dataset used for training the machine learning model. This is the modified and final version of the dataset.

- `asteroids_notebook.ipynb`: Jupyter Notebook file used for data cleaning, exploratory data analysis (EDA), data preparation, model training, and evaluation. It provides a detailed walkthrough of the Data Analysis and ML processes.

- `random_forest_model.pkl`: This is the trained machine learning model that showed the best performance. It's used by the web application to make predictions.

- `scaler.pkl`: Serialized scaler object used for feature scaling. It ensures consistency in feature scaling between training and prediction phases.

- `test_set.pkl`: Pickle file containing the test set used for evaluating the performance of the trained machine learning model (`random_forest_model.pkl`). This test set is essential for calculating metrics such as Mean Absolute Error (MAE) to assess the model's predictive accuracy.

## Technologies

### Libraries

- `sklearn`
- `pandas`
- `seaborn`
- `numpy`
- `statsmodels.api`
- `missingno`
- Others

### Models

- `DecisionTreeRegressor`
- `RandomForestRegressor`
- `XGBRegressor`

Previous versions of the project featured `NeighborsRegressor` and `LinearRegression` as well, but they proved unsatisfactory in performance, leading to their removal in subsequent iterations.

## Demo

![Demo GIF](https://github.com/asparuhovaemily/asteroid-diameter-predictor/raw/main/demo.gif)

