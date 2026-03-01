import streamlit as st
import pandas as pd
from operator import index
from pycaret.regression import setup, compare_models, pull, save_model, load_model

from pycaret.classification import setup, compare_models, pull, save_model
import os 
from ydata_profiling import ProfileReport
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
from pycaret import classification as clf
from pycaret import regression as reg



if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar:
    st.image("https://blog.reffascode.de/content/images/2018/08/ml-cover-1.jpg")
    st.title("AutoML")
    choice=st.radio("Navigation",["Upload","Data Analysis","ML","Download"])
    st.info("This application allows you to perform data analysis and making the best ml pipeline for the model.")


if choice == "Upload":
    st.title("Upload Your Dataset")
    file=st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, sep=None, engine='python', encoding='latin-1')
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Data Analysis":
    st.title("Exploratory Data Analysis")

    if st.button("Generate Profiling Report"):
        with st.spinner("Generating full report..."):

            

            profile = ProfileReport(df, explorative=True)
            profile.to_file("report.html")

            with open("report.html", "r", encoding="utf-8") as f:
                html = f.read()
                components.html(html, height=800, scrolling=True)
if choice == "ML":
    st.title("Machine Learning AutoML")

    if 'df' not in locals():
        st.warning("Please upload dataset first.")
        st.stop()

    chosen_target = st.selectbox("Choose Target Column", df.columns)

    if st.button("Run Modelling"):

        
        df = df.replace(['?', 'NA', 'N/A', 'null', '-'], None)
        df = df.apply(lambda col: pd.to_numeric(col, errors='ignore'))
        df = df.dropna(subset=[chosen_target]).copy()

        # convert target safely
        df[chosen_target] = pd.to_numeric(df[chosen_target], errors='ignore')

        
        unique_count = df[chosen_target].nunique()
        total_count = len(df)
        is_numeric_target = pd.api.types.is_numeric_dtype(df[chosen_target])

        if unique_count <= 20 and not (is_numeric_target and unique_count > 0.05 * total_count):
            task = "classification"
        else:
            task = "regression"

        st.subheader(f"Detected Task: {task.upper()}")
        st.write(f"Unique target values: {unique_count} / {total_count}")

        # CLASSIFICATION
        if task == "classification":

            
            st.write("Class distribution:")
            class_counts = df[chosen_target].value_counts()
            st.write(class_counts)

            # remove rare classes (<2 rows)
            valid_classes = class_counts[class_counts >= 2].index
            df_clean = df[df[chosen_target].isin(valid_classes)].copy()

            removed = set(class_counts.index) - set(valid_classes)
            if removed:
                st.warning(f"Removed rare classes: {removed}")

            if df_clean[chosen_target].nunique() < 2:
                st.error("Not enough classes after cleaning.")
                st.stop()

            with st.spinner("Running classification models..."):

                clf.setup(
                    data=df_clean,
                    target=chosen_target,
                    session_id=42,
                    verbose=False,
                    data_split_stratify=False,
                    fold_strategy="kfold",
                    fold=3
                )

                st.subheader("Setup Summary")
                st.dataframe(clf.pull())

                best_model = clf.compare_models()

                st.subheader("Model Comparison")
                st.dataframe(clf.pull())

                clf.save_model(best_model, "best_model")

        # REGRESSION
        else:

            

            df[chosen_target] = pd.to_numeric(df[chosen_target], errors="coerce")
            df = df.dropna(subset=[chosen_target])

            if df.shape[0] < 30:
                st.error("Dataset too small for regression.")
                st.stop()

            with st.spinner("Running regression models..."):

                reg.setup(
                    data=df,
                    target=chosen_target,
                    session_id=42,
                    verbose=False
                )

                st.subheader("Setup Summary")
                st.dataframe(reg.pull())

                best_model = reg.compare_models()

                st.subheader("Model Comparison")
                st.dataframe(reg.pull())

                reg.save_model(best_model, "best_model")

        st.subheader("Best Model Selected")
        st.success(str(best_model))
        st.success("Model training completed successfully!")
        
if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")

