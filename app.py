########## Work in this ##########

from pycaret.classification import setup as class_setup, pull as class_pull, compare_models as class_compare_models, save_model as class_save_model
from pycaret.regression import setup as reg_setup, pull as reg_pull, compare_models as reg_compare_models, save_model as reg_save_model
import streamlit as st
import os, warnings, time
warnings.filterwarnings('ignore')
import pandas as pd

#from pycaret.classification import setup as class_setup, compare_models as class_compare_models, pull as class_pull, save_model as class_save_model
#from pycaret.regression import setup as reg_setup, compare_models as reg_compare_models, pull as reg_pull, save_model as reg_save_model

#import pycaret

from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

if os.path.exists('./dataset.csv'):
    df = pd.read_csv('./dataset.csv', index_col=None)

with st.sidebar: 
    st.title("AutoInsight Dashboard")
    choice = st.radio("Select Action", ["Upload Dataset", "Exploratory Data Analysis", "Model Building", "Download Results"])
    st.info("Explore and analyze your data with ease using AutoInsight.")

if choice == "Upload Dataset":
    st.title("Upload Your Dataset")
    try:   
        if st.button('Delete Previous Dataset'):
            os.remove('dataset.csv')
            placeholder = st.empty()
            placeholder.write("Dataset Deleted Successfully...")
            time.sleep(2)
            placeholder.empty()
    except:
        st.error('No Previous Dataset Found')
        
    file = st.file_uploader("Upload your dataset here")
    if os.path.exists('./dataset.csv'):
        st.write('Current Dataset')
        st.dataframe(df)
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)
    
if choice == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    try:
        if st.button('Generate Report'):
            profile_df = ProfileReport(df)
            st_profile_report(profile_df)
    except:
        st.error('Please upload dataset first')
        
if choice == "Model Building":
    model_choice = st.radio("Select Model", ["Classification", "Regression"])
    try:
        if model_choice == "Classification":
            chosen_target = st.selectbox('Choose the Target Column', df.columns)
            if st.button('Run Modelling'): 
                
                progress_text = "Training Regression Model..."
                my_bar = st.progress(0, text=progress_text)
                my_bar.progress(33)
                
                class_setup(df, target=chosen_target)
                setup_df = class_pull()
                st.write('Training Parameters')
                st.dataframe(setup_df)
                
                my_bar.progress(66)

                best_model = class_compare_models()
                compare_df = class_pull()
                st.write('Best Performing Models :')
                st.dataframe(compare_df)
                class_save_model(best_model, 'best_model')
                
                my_bar.progress(100)
                my_bar.empty()
             
                placeholder = st.empty()
                placeholder.write("Model Saved Successfully...")
                time.sleep(2)
                placeholder.empty()
                    
        elif model_choice == "Regression":
            chosen_target = st.selectbox('Choose the Target Column', df.columns)
            if st.button('Run Modelling'): 
                
                #start progress bar
                progress_text = "Training Regression Model..."
                my_bar = st.progress(0, text=progress_text)
                my_bar.progress(33)
                
                
                reg_setup(df, target=chosen_target)
                setup_df = reg_pull()
                st.write('Training Parameters')
                st.dataframe(setup_df)
                
                my_bar.progress(66)
                
                best_model = reg_compare_models()
                compare_df = reg_pull()
                st.write('Best Performing Models :')
                st.dataframe(compare_df)
                reg_save_model(best_model, 'best_model')
                
                my_bar.progress(100)
                my_bar.empty()

                
                placeholder = st.empty()
                placeholder.write("Model Saved Successfully...")
                time.sleep(2)
                placeholder.empty()
                
                
    except ValueError:
        if model_choice == 'Classification':
            st.error(f"Cannot perform classification where the target is continuous")
        if model_choice == 'Regression':
            st.error(f"Cannot perform Regression where the target is categorical")
    except NameError:
        st.error('Please upload a dataset first.')
        
        
        
if choice == "Download Results": 
    st.title('Download Results')
    
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download the Best Model', f, file_name="best_model.pkl")