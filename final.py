import csv
import tensorflow as tf
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization 
from keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten
from google.colab.patches import cv2_imshow 

#đọc và xử lý DATA
folder = '/content/drive/MyDrive/DATAKEYPOINT' 
arrayphoto=[] 
photos=list() 
y_train=list()

for stt in range(1,639): 
  image = load_img(
      '/content/drive/MyDrive/DATAKEYPOINT/Screenshot ('+str(stt)+').png',
      color_mode='grayscale',
      target_size = (108,192)
      )
  photo = img_to_array(image).astype(np.uint8)
  photos.append(photo)
  arrayphoto=np.array(photos)
  x_train = arrayphoto
  print(stt)
  data=list()
  data = pd.read_csv(
      '/content/drive/MyDrive/DATAKEYPOINT/Keypoint ('+str(stt)+').csv', 
      delimiter=',',
      header=None
      )
  data=data.T
  y_now=list()
  for i in range(0,68):
    x,y=data[i].astype('float32')
    x=x/10
    y=y/10
    y_now.append(x)
    y_now.append(y)
  y_now=np.array(y_now)  
  y_train.append(y_now)
y_train = np.array(y_train)
y_train=np.array(y_train,dtype='float32')
y_train = y_train 
print(x_train.shape)
print(y_train.shape)

#Tách data test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=2)

#tạo model
from keras.layers import Convolution2D
from keras.layers.convolutional import Deconv2D as Conv2DTranspose
model = Sequential()

model.add(Convolution2D(16, (3,3), padding='same', activation='relu', input_shape=(108,192,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Convolution2D(32, (3,3), padding='same', activation='relu', input_shape=(108,192,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Convolution2D(64, (3,3), padding='same', activation='relu', input_shape=(108,192,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Convolution2D(128, (3,3), padding='same', activation='relu', input_shape=(108,192,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Flatten())

model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(136))
print(model.summary())
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

#train model
from keras.callbacks import EarlyStopping
history=model.fit(x_train,y_train,epochs=1000,batch_size=7,verbose=1,callbacks=[EarlyStopping(monitor='accuracy',patience=10)])
#lưu model
model.save('/content/drive/MyDrive/landmarks_653data.h5')

#chạy video
import matplotlib.pyplot as plt
import cv2
vidcap = cv2.VideoCapture('/content/drive/MyDrive/videoplayback.mp4')
count = 0
while success:

  
  success,image3 = vidcap.read()
  cv2.imwrite("frame.jpg", image3)
  image3 = load_img('/content/frame.jpg',color_mode='grayscale',target_size = (108,192)) #/10
  image3=np.array(image3)
  plt.imshow( image3,cmap='gray' ) 
  image3 = np.reshape(image3,(-1,108,192,1)).astype('float32')
  pred = model.predict( image3 ) 
  pred = np.reshape( pred[0], ( 68 , 2 ) )  
  plt.scatter( pred[ : , 0 ] , pred[ : , 1 ] , c='yellow',s=2 )
  plt.show()
