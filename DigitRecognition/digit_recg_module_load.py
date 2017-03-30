# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 17:22:48 2017

@author: Anjali Kumari
"""
from keras.models import model_from_json
import numpy as np
from PIL import Image
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

def input_image():
    img= Image.open('image/digitsImages/9.bmp')
    
    img= img.resize((28,28))
#plt.imshow(img,cmap=plt.get_cmap('gray'))
    img= np.asarray(img)
    print img.shape
    x0=1
    x1=img.shape[1]*img.shape[0]
    img= img.reshape((x0,x1))
    print img.shape
    return img    
    
img= input_image ()  

prediction=loaded_model.predict_classes(img)
print 'prediction', prediction