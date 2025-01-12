import streamlit as st
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb


df = pd.read_csv("test.csv")
df.set_index("datetime", inplace=True)



y_test = pd.read_csv("target.csv")
y_test.set_index("datetime", inplace=True)

df_ = pd.read_csv("comb_data.csv")
df_.set_index("datetime", inplace=True)

y_t = pd.read_csv("comb_target.csv")
y_t = y_t.set_index("datetime", drop=True)


st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a Page", ["About", "LGBM", "LGBM With DNN"])



if page == "About":
    st.title("About")
    st.write("""
    This is a Streamlit application that demonstrates the use of machine learning models. 
    You can navigate to different pages using the sidebar to load models, visualize them, and make predictions.
    """)


elif page == "LGBM":
    st.title("LGBM: Predict and Visualize")
    
    
    model = joblib.load("model_w_lags.pkl")
    
    
    X_test = df  
    y_pred = model.predict(X_test)
    
    
    st.write("Predictions:")
    st.write(y_pred)

    st.write("Real Values:")
    st.write(y_test)
    
    mae = mean_absolute_error(y_test, y_pred)

    st.write("MAE:")
    st.write(mae)

    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index, y_test['target'], label='Real Values', color='blue')
    ax.plot(X_test.index, y_pred, label='Predicted Values', color='red')
    ax.set_title('TSLA Stock Price Forecasting')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    st.pyplot(fig)


elif page == "LGBM With DNN":
    st.title("LGBM With DNN: Predict and Visualize")
    
    
    model = joblib.load("lgbm_comb_model.pkl")
    
    
    X_test = df_  
    y_pred = model.predict(X_test)
    
    
    st.write("Predictions:")
    st.write(y_pred)

    st.write("Real Values:")
    st.write(y_t["target"])

    mae = mean_absolute_error(y_t["target"], y_pred)
    
    st.write("MAE:")
    st.write(mae)
    
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_.index, y_t["target"], label='Real Values', color='blue')
    ax.plot(X_test.index, y_pred, label='Predicted Values', color='red')
    ax.set_title('TSLA Stock Price Forecasting')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    st.pyplot(fig)



















