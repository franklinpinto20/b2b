import streamlit as st
from firebase_admin import firestore
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import polars as pl
import csv
import pandas as pd
import numpy as np
import math
import re
import random
#from ydata_profiling import ProfileReport
#import seaborn as sns

#Entrenamiento del modelo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay, roc_curve, roc_auc_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

#Visualización de datos
import matplotlib.pyplot as plt
import seaborn as sns

import tarfile
import pandas as pd

import seaborn as sns
import csv
import json
#from pickle import dump
#from pickle import load
import requests
from io import BytesIO
import datetime

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import train_test_split
import pickle
from pyspark.sql import SparkSession
import os
# Set the JAVA_HOME variable to the path of your Java installation, and define the exact path where you have installed java. 

os.environ['JAVA_HOME'] = 'C://Program Files//Java//jdk-22'

# Crea una instancia de SparkSession con la configuración especificada
spark = SparkSession.builder\
    .master('local[*]') \
    .config("spark.driver.memory", "10g")\
    .appName("HYB").getOrCreate()
#%matplotlib inline


from surprise import SVD, Dataset, Reader, dump,Dataset, Reader, KNNBasic



import pickle
from sklearn.datasets import load_iris
import joblib
from pyspark.sql.functions import col,when

# open a file, where you stored the pickled data
#file = open('C://Users//Acer//repos//sysrecsongs//dataset//model_CF.pickle', 'rb')

# dump information to that file
#loaded_model = pickle.load(file)

# close the file
#file.close()



def app():
    #db=firestore.client()
    # Define la URL base donde están almacenados los archivos de datos
    url = 'C://Users//Acer//repos//sysrecsongs//dataset//'


    # Define la ruta del archivo desde el cual se cargará el modelo
    file_name = 'C://Users//Acer//repos//sysrecsongs//dataset//model_CF.pickle'

    # Carga el modelo desde el archivo
    _, loaded_model = dump.load(file_name)
    #_, loaded_model = pickle.load(open(file_name, 'rb'))
            
   # Lee los conjuntos de datos de entrenamiento y prueba desde archivos Parquet
    dataset_train = pd.read_parquet(f'{url}train_dataset.parquet')
    dataset_train = dataset_train[['user_id', 'business_id', 'rating']]

    dataset_test = pd.read_parquet(f'{url}test_dataset.parquet')
    dataset_test = dataset_test[['user_id', 'business_id', 'rating']]

    # Define el lector con la escala de calificación (rating scale) entre 1 y 5
    reader = Reader(rating_scale=(1, 5))

    # Carga los datos de entrenamiento y prueba en formato Dataset
    train_data = Dataset.load_from_df(dataset_train, reader)
    test_data = Dataset.load_from_df(dataset_test, reader)

    # Construye el conjunto de entrenamiento completo (trainset) a partir de los datos
    trainset = train_data.build_full_trainset()

    # Crea un conjunto de test para el trainset
    trainset_2 = train_data.build_full_trainset().build_testset()

    # Crea un conjunto de test para el testset
    testset = test_data.build_full_trainset().build_testset()
   
  

    # Load the model from the pickle file
    #with open(file_name, 'rb') as f:
    #    _, loaded_model = pickle.load(f)

 
   
   # Realiza predicciones en el conjunto de entrenamiento
    predictions_train = loaded_model.test(trainset_2)

    # Realiza predicciones en el conjunto de prueba
    predictions_test = loaded_model.test(testset)
    
  # Corrigiendo la conversión para predictions_train
    predictions_train_df = pd.DataFrame([
        {
            'user_id': pred.uid,
            'business_id': pred.iid,
            'actual_rating': pred.r_ui,
            'estimated_rating': pred.est
        } for pred in predictions_train
    ])

    # Corrigiendo la conversión para predictions_test
    predictions_test_df = pd.DataFrame([
        {
            'user_id': pred.uid,
            'business_id': pred.iid,
            'actual_rating': pred.r_ui,
            'estimated_rating': pred.est
        } for pred in predictions_test
    ])

    # Convertir DataFrame de Pandas a DataFrame de Spark
    #train_predictions_cf_df = spark.createDataFrame(predictions_train_df)
    #test_predictions_cf_df = spark.createDataFrame(predictions_test_df)




    # Cargar el archivo CSV
    data_cb = pd.read_csv(f'{url}result_content_based.csv')

    # Convertir DataFrame de Pandas a DataFrame de Spark
    train_predictions_cb_df = data_cb #spark.createDataFrame(data_cb)
    test_predictions_cb_df = data_cb #spark.createDataFrame(data_cb)

    json_data_business = pd.read_json('C://Users//Acer//repos//sysrecsongs//dataset//yelp_academic_dataset_business.json', lines=True)
    #json_data_checkin = pd.read_json('C://Users//Acer//repos//sysrecsongs//dataset//yelp_academic_dataset_checkin.json', lines=True)
    #json_data_users = pd.read_json('C://Users//Acer//repos//sysrecsongs//dataset//yelp_academic_dataset_user-001.json')
    
    
    #st.title('train_predictions_cf_df: ')
    #st.dataframe(train_predictions_cf_df, use_container_width=True)
    
    
    #Hibridación
    # Unir los DataFrames de predicciones de CF y CB en una sola DataFrame
    #two_models_df = predictions_train_df.join(train_predictions_cb_df, ['business_id', 'user_id'], 'inner')
    two_models_df = pd.merge(predictions_train_df,train_predictions_cb_df, on=['business_id', 'user_id'])
    
    two_models_df=two_models_df.sort_values(by=['estimated_rating'], ascending=True)
   
    st.title('BUSINESS RECOMENDATIONS: ')
    st.title('In Philadelphia: ')
  
    business_df = pd.merge(two_models_df,json_data_business, on=['business_id'])
    
    business_df=business_df.sort_values(by='estimated_rating', ascending=False)
    
    business_df=business_df[['name','estimated_rating','address','categories']]
       
    st.dataframe(business_df, use_container_width=True)
    
    alpha_final = 0.05600202115931041
    beta_final = 0.9439979788406896
    total=0.8987002713879976
    st.write("RMSE", total)
    
    st.title('PROCESS: ')
    
    st.title('hybrid: ')
    st.dataframe(two_models_df, use_container_width=True)
        
    st.title('predictions train: ')
    st.dataframe(predictions_train_df, use_container_width=True)
    
    
    st.title('predictions train: ')
    st.dataframe(train_predictions_cb_df, use_container_width=True)
    
    st.title('predictions test: ')
    st.dataframe(test_predictions_cb_df, use_container_width=True)
    
    
    
       
    #hibrid_df= two_models_df['business_id']
    #two_models_df_sp = spark.createDataFrame(two_models_df)
    # Renombrar columnas en el DataFrame
    #two_models_df_sp = two_models_df_sp.withColumnRenamed("actual_rating", "rating") \
    #                         .withColumnRenamed("estimated_rating", "cf_rating") \
    #                         .withColumnRenamed("predicted_rating", "cb_rating")

    
    #np.sqrt(two_models_df_sp.withColumn('final_rating', when(col('cb_rating').isNull(), col('cf_rating')) \
    #                              .otherwise((0.5 * col('cb_rating') + 0.5 * col('cf_rating')) / (1))) \
    #                  .withColumn('error', (col('final_rating') - col('rating')) ** 2) \
    #                  .groupBy().sum('error') \
    #                  .collect()[0][0] / two_models_df_sp.count())
    
    #st.title('Business: ')
    #st.dataframe(json_data_business, use_container_width=True)
    
    #st.title('Checkin: ')
    #st.dataframe(json_data_checkin, use_container_width=True)
    
 
    #st.dataframe(data, use_container_width=True)
    """
    dataset_train=pd.read_parquet(f'{url}train_dataset.parquet')
    dataset_train=dataset_train[['user_id','business_id','rating']]
        
    dataset_test=pd.read_parquet(f'{url}test_dataset.parquet')
    dataset_test=dataset_test[['user_id','business_id','rating']]
    
    #st.title('dataset_train: ')
    #st.dataframe(dataset_train, use_container_width=True)
    
    reader = Reader( rating_scale = ( 1, 5 ) )
    train_data = Dataset.load_from_df( dataset_train , reader )
    test_data = Dataset.load_from_df( dataset_test, reader )

    trainset = train_data.build_full_trainset()  # Create the trainset from data
    trainset_2 = train_data.build_full_trainset().build_testset()
    testset = test_data.build_full_trainset().build_testset()
    
    # Define a parameter grid to search over
    param_grid = {
    #    'n_factors':[50],
        'n_factors':[50,100,150],
        'n_epochs': [5, 10,15,20],
        'lr_all': [0.002 , 0.01, 0.02],
        'reg_all': [0.2,0.4, 0.6]
    }

    # Configure the grid search for the SVD algorithm
    #gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=10, joblib_verbose=10)
    gs = GridSearchCV(SVD, param_grid,                           cv=3, n_jobs=10, verbose=1, scoring='accuracy')

    # Perform grid search
    #gs.fit(train_data)
    
    
    # Access the best parameters for the 'rmse' metric
    #params = gs.best_params['rmse']

    # Create an instance of the SVD algorithm using the best parameters found
    svdtuned = SVD(n_factors=param_grid['n_factors'], n_epochs=param_grid['n_epochs'], lr_all=param_grid['lr_all'], reg_all=param_grid['reg_all'])

    


    #ds_business=pd.read_json('C://Users//Acer//repos//sysrecsongs//dataset//yelp_academic_dataset_business.json')
    user_timestamp=pd.read_csv('C://Users//Acer//repos//sysrecsongs//dataset//userid-timestamp-artid-artname-traid-traname.tsv',sep='\t')
    user_profile = pd.read_csv('C://Users//Acer//repos//sysrecsongs//dataset//userid-profile.tsv',sep='\t')
  

    try:
       # st.title('negocio: '+st.session_state['username'] )

        #st.write("Data Frame 1", dataset_train)
        

       

        #funcion para clasificar las 
        def classify(num):
            if num == 0:
                return 'Metal'
            elif num == 1:
                return 'Rock'
            else:
                return 'Reggae'
            
        if st.button('Recommend business'):
                if 1 == 1:
                    st.success(classify(1))
                
                
    except:
        if st.session_state.username=='':
            st.text('Please Login first')        
"""