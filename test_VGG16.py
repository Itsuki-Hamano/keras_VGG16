# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 23:44:24 2018

@author: IstukiHamano
"""

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16,preprocess_input,decode_predictions
import numpy as np

model=VGG16(weights='imagenet',include_top=True)

img_path='img/road1.jpg'
img=image.load_img(img_path,target_size=(224,224))
img.show()
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)

model.summary()

preds=model.predict(preprocess_input(x))
print(preds) #学習済み１０００個のマトリクスで確率出力
results=decode_predictions(preds,top=10)[0]#上位個出力
for result in results:
    print(result)
    
    
