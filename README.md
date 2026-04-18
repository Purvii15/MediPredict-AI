# MediPredict AI - Disease Prediction System

An end-to-end Machine Learning web application that predicts diseases from user-selected symptoms.

## Features
- 3 ML Models: Random Forest (97.8%), Decision Tree (95.1%), Naive Bayes (96.7%)
- 41 diseases, 132 symptom features
- Weighted symptom system for medical realism
- Safety gates to prevent false severe disease predictions
- Smart symptom suggestions
- Visual analytics dashboard (EDA, confusion matrix, feature importance)
- Prediction history and downloadable reports

## Tech Stack
Python 3.11, Flask, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

## Run Locally
pip install -r requirements.txt
python model.py
python app.py

Open http://127.0.0.1:5000

## Disclaimer
For educational purposes only. Not a medical device.
