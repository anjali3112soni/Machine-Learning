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
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures 

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


 


x_input, labels,test_data,fd = read_Data(location)





# computation for polynomial fitting
poly = PolynomialFeatures(degree=2)

x_= poly.fit_transform(x_input)
predict_=poly.fit_transform(test_data)
clf= linear_model.LinearRegression()
clf.fit(x_,labels)
print 'prediction for test_data',clf.predict(predict_)


