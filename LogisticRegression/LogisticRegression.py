# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 20:43:03 2017

@author: Anjali Kumari
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 17:00:23 2017

@author: Anjali Kumari
"""
import numpy as np
import pandas as pd
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Dense,Activation
from sklearn import datasets,linear_model
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
location =  r'E:\python\assignment\ex2data1-logistic.xls'
location2 =  r'E:\python\assignment\ex2data2-logistic.xls'
seed = 7
np.random.seed(seed)

def getData(location):
    dataset= pd.read_excel(location,names=['X1','X2','Y'])
    dataset2= pd.read_excel(location2,names=['X1','X2','Y'])
    m1,n1= dataset.shape
    m2,n2= dataset2.shape
    
    dataset= dataset.append(dataset2)
    dataframe=dataset.as_matrix()
    dataframe2= dataset2.as_matrix()
    
    num = m1 * 90/100                  # taking 90% of data from ex2data1-logistic.xls
    input1 = dataframe[0:num,0:2]
    test1= dataframe[num:m1,0:2]
    label1= dataframe[0:num,2]
    test_label1= dataframe[num:m1,2]
                                        # taking 90% of data from ex2data2-logistic.xls
    num= m2 * 90/100
    input2 = dataframe2[0:num,0:2]
    test2=dataframe2[num:,0:2]
    label2= dataframe2[0:num,2]
    test_label2= dataframe2[num:m2,2]
    
    labels=np.concatenate((label1, label2))     #get final labeled data
    inputs=np.concatenate((input1, input2))     #get final inputs data
    test_data= np.concatenate((test1, test2))
    test_labels= np.concatenate((test_label1,test_label2))      #get final test_data
    
    positive = dataset[dataset['Y'].isin([1])]
    negative = dataset[dataset['Y'].isin([0])]
                          

    print 'done entering dataset'
    return inputs,labels,test_data,test_labels,positive,negative
    
    
    
    
def create_Model(inp,lab):
    model = Sequential()
    model.add(Dense(2, input_dim=2,init='normal',activation='relu'))
    weight= Dense(2, input_dim=2,init='normal',activation='relu')
    model.add(Dense(1,init='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(inp,lab,nb_epoch=1000,batch_size=20)
    return model,weight
    
def dataFit(inp, lab):
    model = linear_model.LogisticRegression()
    model.fit(inp,lab)
    return model
    
    
def sigmoid_Function(z):
    sf= 1/(1+np.exp(-z))
    return sf
    
    
def costFunction(theta0,theta1,theta2,test_y,test_x):
    z= theta0+theta1*test_x[0,0]+ theta2*test_x[0,1]
    
    cost=  np.multiply(test_y, np.log(sigmoid_Function(z)))+np.multiply((1-test_y), np.log(sigmoid_Function(z)))
    
    #print 'sssssssss',sigmoid_Function(z)
    #print 'test_y',test_y
   
    return -cost
    
def linearCost(theta0,theta1,theta2,test_y,test_x):
    z= theta0+theta1*test_x[0,0]+ theta2*test_x[0,1]
    cost= (sigmoid_Function(z)-test_y)**2
    return cost
    
    
#call getData function to get trainnind data and test data    
inputs,labels,test_data,test_labels,pos,neg = getData(location)

#logistic model
model,weight = create_Model(inputs,labels)


m,n= test_data.shape

z= list()
t=np.asarray(test_labels).reshape(1,-1)
single_test_example_y= t[0,3]
single_test_example_x=np.asarray(test_data)[3,0:2].reshape(1,-1)
print 'single_test_example_y',single_test_example_y
result = {}
for i in range(m):
    test_X=np.asarray(test_data)[i,0:2].reshape( 1,-1)
    
    y=model.predict(test_X)
    prob  = model.predict_proba(test_X)
    clas=model.predict_classes(test_X, batch_size=10)
    output=clas
    #loss_and_metrics = model.evaluate(t[0,i], output, batch_size=32)
    #print 'loss_and_metrics',loss_and_metrics
    result[i+1]=output
    
print result 
   

    
 #point PLOTTING    
fig, ax = plt.subplots(figsize=(10,8))  
 
ax.scatter(pos['X1'], pos['X2'], marker='o', c='b')
ax.scatter(neg['X1'], neg['X2'], marker='x', c='r')
    
ax.legend()  
ax.set_xlabel('X1')  
ax.set_ylabel('X2') 

#Linear cost function plotting

fig = plt.figure()
ax = fig.gca(projection='3d')
theta0= 100
theta1 = np.arange(-10, 10, 0.02)
theta2 = np.arange(-10, 10, 0.02)
X, Y = np.meshgrid(theta1, theta2)
surf=ax.plot_surface(X, Y, linearCost(theta0,X,Y,single_test_example_y,single_test_example_x), cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_xlabel('Theta1')
ax.set_ylabel('Theta2')
ax.set_zlabel('Linear Cost')
fig.colorbar(surf, shrink=0.5, aspect=10)
fig.savefig('E:/python/assignment/linearCost.png')

plt.show()

# plotting Logistic cost function
fig = plt.figure()
ax = fig.gca(projection='3d')
surf=ax.plot_surface(X, Y, costFunction(theta0,X,Y,single_test_example_y,single_test_example_x), cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel('Theta1')
ax.set_ylabel('Theta2')
ax.set_zlabel('Logistic Cost')
fig.savefig('E:/python/assignment/LogisticCost.png')
plt.show()


