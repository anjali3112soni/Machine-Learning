# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 14:26:04 2017

@author: Anjali Kumari
"""
import matplotlib.pyplot as plt
# Plot ad hoc mnist instances
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.utils import np_utils
from keras.layers import Dropout
import numpy as np
from PIL import Image

seed= 7
np.random.seed(seed)
# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# plot 4 images as gray scale
#plt.subplot(221)
#plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
#plt.subplot(222)
#plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
#plt.subplot(223)
#plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
#plt.subplot(224)
#plt.imshow(X_train[9], cmap=plt.get_cmap('gray'))
## show the plot
#plt.show()

plt.imshow(X_test[0], cmap=plt.get_cmap('gray'))
plt.show()
# coverting 28X28 size into array of 784

print y_train[0]
num_pixels = X_train.shape[1]*X_train.shape[2] 
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

#normalizing inputs 

X_train= X_train / 255
X_test = X_test / 255

#converting integer into vector representation
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
print num_classes

def create_Model():
    model= Sequential()
    model.add(Dense(num_pixels,input_dim=num_pixels,init='normal',activation='relu'))
    model.add(Dense(num_classes,init='normal',activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
 
def input_image():
    img= Image.open('image/1bmp.bmp')
    
    img= img.resize((28,28))
#plt.imshow(img,cmap=plt.get_cmap('gray'))
    img= np.asarray(img)
    print img.shape
    x0=1
    x1=img.shape[1]*img.shape[0]
    img= img.reshape((x0,x1))
    print img.shape
    return img   
    
t= X_test[0]
print t.shape
t=[t]
t=np.array(t)
print t.shape 

model= create_Model()
model.fit(X_train,y_train,batch_size=32,nb_epoch=10,verbose=2)
model.fit(X_test,y_test,batch_size=32,nb_epoch=10,verbose=2)
scores = model.evaluate(X_test, y_test, verbose=0)
print 'score', scores
 
prediction=model.predict_classes(t)
print prediction


imgg= input_image()
pred= model.predict_classes(imgg)

print 'imput image predi',pred

#saving model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
 # serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

    



