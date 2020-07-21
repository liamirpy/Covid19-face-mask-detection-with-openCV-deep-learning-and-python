import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1000)
import cv2
import os
from PIL import Image
import keras
os.environ['KERAS_BACKEND']='tensorflow'
image_directory='/content/drive/My Drive/Covid_19/'
SIZE=256
dataset=[]
label=[]
n=0
o=0
########READ DATA
mask_images=os.listdir(image_directory+'with_mask/')
for i,image_name in enumerate(mask_images):
    if(image_name.split('.')[1]=='png' or image_name.split('.')[1]=='jpeg' or image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'with_mask/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((SIZE,SIZE))
        n=n+1
        dataset.append(np.array(image))
        label.append(1)

nomask_images=os.listdir(image_directory+'without_mask/')
for i,image_name in enumerate(nomask_images):
    if(image_name.split('.')[1]=='png' or image_name.split('.')[1]=='jpeg' or image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'without_mask/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((SIZE,SIZE))
        o=o+1
        dataset.append(np.array(image))
        label.append(0)
print(n)
print(o)
#####MODEL
INPUT_SHAPE=(SIZE,SIZE,3)
inp=keras.layers.Input(shape=INPUT_SHAPE)
conv1=keras.layers.Conv2D(64,kernel_size=(3,3),
                          activation='relu',padding='same')(inp)
pool1=keras.layers.MaxPooling2D(pool_size=(2,2))(conv1)
norm1=keras.layers.BatchNormalization(axis=-1)(pool1)
drop1=keras.layers.Dropout(rate=0.2)(norm1)

conv2=keras.layers.Conv2D(32,kernel_size=(3,3),
                          activation='relu',padding='same')(drop1)
pool2=keras.layers.MaxPooling2D(pool_size=(2,2))(conv2)
norm2=keras.layers.BatchNormalization(axis=-1)(pool2)
drop2=keras.layers.Dropout(rate=0.2)(norm2)

flat=keras.layers.Flatten()(drop2)

hidden1=keras.layers.Dense(128,activation='relu')(flat)
norm3=keras.layers.BatchNormalization(axis=-1)(hidden1)
drop3=keras.layers.Dropout(rate=0.2)(norm3)

hidden2=keras.layers.Dense(50,activation='relu')(drop3)
norm4=keras.layers.BatchNormalization(axis=-1)(hidden2)
drop4=keras.layers.Dropout(rate=0.2)(norm4)

out=keras.layers.Dense(2,activation='softmax')(drop4)

model=keras.Model(input=inp,outputs=out)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
print(model.summary())
##################
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
X_train,X_test,y_train,y_test=train_test_split(dataset,to_categorical(np.array(label)),test_size=0.20,random_state=0)


history=model.fit(np.array(X_train),y_train,batch_size=32,verbose=1,epochs=15,validation_split=0.1)

print("Test_Acuracy:{:.2f}%".format(model.evaluate(np.array(X_test),np.array(y_test))[1]*100))

model.save('mask_nomask.h5')

