# Breast-Cancer-Prediction-System

This repository features an AI-driven clinical decision support system designed to predict whether a breast mass is malignant or benign. The project encompasses the entire machine learning lifecycle, from exploratory data analysis and model training to a user-friendly web interface for real-time predictions.

Features
Predictive Modeling: Utilizes a neural network trained on the Wisconsin Breast Cancer dataset to classify tumors with high accuracy.

Interactive Dashboard: A Streamlit-based web application that allows users to upload patient data and receive instant diagnostic predictions.

Automated Preprocessing: Includes a custom feature-mapping engine to ensure that user-uploaded CSV data matches the model's required input format.

Clinical Insights: Provides probability scores for both Malignant (M) and Benign (B) classifications to assist in medical decision-making.

Tech Stack:

Python: Core programming language.

Machine Learning: TensorFlow/Keras (for the neural network) and Scikit-Learn (for data scaling and splitting).

Data Processing: Pandas and NumPy.

Web Framework: Streamlit.

Model Persistence: Joblib.

How to Use
Launch the Streamlit application.

Upload a patient CSV file (ensure it contains an id column and the relevant diagnostic features).

Select a specific Patient ID from the dropdown menu.

Click "Predict Cancer" to view the AI's classification and confidence levels.

