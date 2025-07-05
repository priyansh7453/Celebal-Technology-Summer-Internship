import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Load or train a simple model for demonstration
@st.cache_resource
def load_or_train_model():
    try:
        with open('iris_model.pkl', 'rb') as file:
            return pickle.load(file)
    except:
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        with open('iris_model.pkl', 'wb') as file:
            pickle.dump(model, file)
        return model

model = load_or_train_model()
iris = load_iris()

# Streamlit app
st.set_page_config(page_title="Iris Classifier", layout="wide")
st.title("Iris Flower Classification")

# Input method
option = st.sidebar.radio("Input method:", ("Single Prediction", "Batch Prediction"))

if option == "Single Prediction":
    st.header("Single Prediction")
    col1, col2 = st.columns(2)
    
    with col1:
        sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8)
        sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
    
    with col2:
        petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.0)
        petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.2)
    
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
else:
    st.header("Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV file with iris measurements", type=["csv"])
    
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        st.write("Data preview:")
        st.dataframe(input_data.head())

if st.button("Predict"):
    if option == "Single Prediction":
        prediction = model.predict(input_data)
        proba = model.predict_proba(input_data)[0]
        
        st.subheader("Result")
        st.write(f"Predicted species: **{iris.target_names[prediction[0]]}**")
        
        st.write("Prediction probabilities:")
        proba_df = pd.DataFrame({
            "Species": iris.target_names,
            "Probability": proba
        })
        st.dataframe(proba_df)
        
        fig, ax = plt.subplots()
        sns.barplot(x='Probability', y='Species', data=proba_df, ax=ax)
        ax.set_title("Prediction Probabilities")
        st.pyplot(fig)
        
    else:
        if len(input_data.columns) != 4:
            st.error("Input data must have exactly 4 numerical features")
        else:
            predictions = model.predict(input_data)
            results = input_data.copy()
            results['Prediction'] = [iris.target_names[p] for p in predictions]
            
            st.subheader("Results")
            st.dataframe(results)
            
            fig, ax = plt.subplots()
            sns.countplot(x='Prediction', data=results, ax=ax, order=iris.target_names)
            ax.set_title("Prediction Distribution")
            st.pyplot(fig)

# Model explanation
st.header("Model Insights")
st.subheader("Feature Importance")
importance = model.feature_importances_
features = iris.feature_names

fig, ax = plt.subplots()
sns.barplot(x=importance, y=features, ax=ax)
ax.set_title("Feature Importance")
st.pyplot(fig)