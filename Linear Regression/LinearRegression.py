# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 09:13:37 2017

@author: Anjali Kumari
"""
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Activation
from sklearn import datasets, linear_model
from sklearn import preprocessing
from pylab import polyfit, poly1d
import matplotlib.pyplot as plt
import math



location =  r'E:\python\assignment\data_carsmall.xlsx'
seed=7
np.random.seed(seed)

def read_Data(filepath):
    dataframe = pd.read_excel(filepath,names=['X1','X2','X3','X4','X5','Y'])
    data = dataframe.as_matrix()
    d= list()
    m,n=  data.shape
    for i in range(m-1):
        if (math.isnan(data[i+1,5])):
            d.append(data[i+1,0:5])
    
    d= np.array(d)
    test_data= preprocessing.scale(d.astype(float))
    filter_data = dataframe.dropna().as_matrix()
    train_data = filter_data[1:,0:5]       #getting train data
    fd= filter_data[3,0:5]
    
    train_data = preprocessing.scale(train_data.astype(float))
    labels = filter_data[1:,5]
    #labels= preprocessing.scale(labels)
     
    return train_data.astype(float), labels,test_data,fd


 
def dataFitting(x,y):
    regr = linear_model.LinearRegression()
    regr.fit(x,y)
    prediction ={}
    prediction['intercept'] = regr.intercept_
    prediction['coefi']= regr.coef_
    return prediction,regr
    
    
def create_Model(inp,lab):
    model = Sequential()
    model.add(Dense(5, input_dim=5,init='normal',activation='relu'))
    model.add(Dense(1,init='normal'))
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
    model.fit(inp,lab,nb_epoch=100,batch_size=32)
    model.predict
    return model

    

    
x_input, labels,test_data,fd = read_Data(location)

pred_data,model= dataFitting(x_input,labels)

intercept=model.intercept_
coef= model.coef_



print 'coef of hypothesis is ',coef
print 'intercept of hypothesis is ',intercept

m,n= test_data.shape
result= {}    # to store prediction 
for i in range(m):
    t= test_data[i,0:5]
    result[i+1]=model.predict([t]) 
    
print 'prediction for test datas are',result

# computation for polynomial fitting


