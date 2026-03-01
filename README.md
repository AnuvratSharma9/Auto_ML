# 🚀 Mini AutoML System (Streamlit + PyCaret)

A fully deployed mini AutoML web app that performs data analysis and trains multiple machine learning models automatically from a CSV file.

Upload dataset → analyze → train → download best model.


GitHub: https://github.com/AnuvratSharma9/Auto_ML

---

## 🧠 What This Project Does

This system allows anyone to upload a dataset and automatically:

- Perform exploratory data analysis (EDA)
- Detect whether problem is classification or regression
- Train multiple ML models
- Compare performance
- Select best model
- Download trained model (.pkl)

All from a simple web interface.

---

## ⚙️ Features

- Upload any CSV dataset
- Automatic EDA dashboard
- Correlation heatmaps & distributions
- Smart task detection (classification vs regression)
- Multiple model training using PyCaret
- Best model selection
- Download trained model as pickle file
- Fully deployed on cloud

---

## 🛠 Tech Stack

**Core**
- Python
- Streamlit (UI + deployment)
- PyCaret (AutoML engine)

**Data & Visualization**
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn

---

## 🤖 Why I Built This

I wanted to understand how real AutoML tools work internally instead of just using them.

So I built my own mini version that:
- runs multiple ML algorithms
- compares performance
- selects the best one
- deploys everything on cloud

PyCaret even started pulling out ML algorithms I hadn’t heard of outperforming the usual ones — I genuinely had moments of “what is going on”.

Obviously this is not competing with enterprise AutoML platforms in speed or optimization.

But for a fully free deployed project, I’m genuinely satisfied with how much it can do.

---

## 🌐 Live Demo

Try it here:
https://auto-ml-amnv.onrender.com

Upload any dataset and test.

For demo purposes, a small dataset works best so models train quickly.

---

## 💻 Run Locally

Clone repository:

```bash
git clone https://github.com/AnuvratSharma9/Auto_ML.git
cd Auto_ML
