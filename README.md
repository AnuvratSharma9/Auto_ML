# End-to-End AutoML System

This project is a full automated machine learning web application built using Streamlit and PyCaret.

## Features
- Upload CSV datasets
- Automated EDA using ydata-profiling
- Auto detect classification or regression
- Train multiple ML models automatically
- Select best performing model
- Download trained model (.pkl)

## Tech Stack
- Python
- Streamlit
- PyCaret
- Pandas
- Scikit-learn
- ydata-profiling

## Use Case
Helps automate machine learning workflow from raw dataset to trained model with minimal user input.

## Installation & Run Locally

### 1. Clone repository
git clone https://github.com/AnuvratSharma9/Auto_ML.git

### 2. Go into folder
cd Auto_ML

### 3. Create virtual environment
python -m venv venv

### 4. Activate environment

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

### 5. Install requirements
pip install -r requirements.txt

### 6. Run app
streamlit run app.py
