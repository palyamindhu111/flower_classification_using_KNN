# Flower Classification

A Streamlit web app that predicts an Iris flower species from its sepal and petal measurements. Predictions are made with a K-Nearest Neighbors (KNN) classifier trained on scikit-learn's built-in Iris dataset.

## Features

- Enter sepal length, sepal width, petal length, and petal width in centimeters.
- Predict the flower species using a KNN model with 3 neighbors.
- View the result in a simple, styled Streamlit interface.

## Requirements

- Python 3
- Packages listed in `requirements.txt`

## Setup and run

From the project directory, install the dependencies and start the app:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The trained model (`iris_model.joblib`) is included, so the app is ready to run after installing the dependencies.

To retrain the model and regenerate that file, run:

```bash
python "ml code"
```

Then start the app with `streamlit run app.py`.

## Project files

| File | Description |
| --- | --- |
| `app.py` | Streamlit interface and prediction logic |
| `ml code` | Trains the KNN classifier on the Iris dataset and saves the model |
| `iris_model.joblib` | Trained model and Iris species names used by the app |
| `flower-background.svg` | App background image |
| `requirements.txt` | Python package dependencies |
