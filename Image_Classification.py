#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import tensorflow as tf


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras


# In[3]:


tf.test.is_built_with_cuda()


# In[4]:


(X_train ,Y_train),(X_test,Y_test)=tf.keras.datasets.cifar10.load_data()


# In[5]:


X_train.shape


# In[6]:


Y_train.shape


# In[7]:


X_test.shape


# In[8]:


Y_test.shape


# In[9]:


Y_train[:8]


# In[10]:


X_train[1]


# In[11]:


classes=["aeroplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


# In[12]:


classes[Y_train[3][0]] #y_train is array so we want index 0 of it


# In[13]:


def ploty(x,y,i):
    plt.figure(figsize=(11,2))
    plt.imshow(x[i])
    plt.xlabel(classes[y[i]])


# In[14]:


Y_train=Y_train.reshape(-1,)
ploty(X_train,Y_train,4)


# In[15]:


X_train_S=X_train/255
Y_train_S=Y_train/255


# In[16]:


X_train_S[1]


# In[17]:


y_train_cat=keras.utils.to_categorical(Y_train,num_classes=10,dtype='float32') #one hot encoding


# In[18]:


y_train_cat[4]


# In[19]:


y_test_cat=keras.utils.to_categorical(Y_test,num_classes=10,dtype='float32') 


# In[21]:


ann= keras.Sequential([
    keras.layers.Flatten(input_shape=(32,32,3)),
    keras.layers.Dense(3072,activation='relu'),
    keras.layers.Dense(2000,activation='relu'),
    keras.layers.Dense(1000,activation='relu'),
    keras.layers.Dense(10,activation='sigmoid')
])
ann.compile(optimizer='SGD',loss='categorical_crossentropy',metrics=['accuracy'])
ann.fit(X_train,y_train_cat,epochs=2)


# In[23]:


import joblib
joblib.dump(ann, 'first_model.pkl')


# In[24]:


cnn= keras.Sequential([
    #cnn
    #1st block
    keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',input_shape=(32,32,3)),
     keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'),
    keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    #2nd block
    keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'),
     keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'),
    keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    #3rd block
    keras.layers.Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'),
     keras.layers.Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    
    keras.layers.Flatten(),
    keras.layers.Dense(64,activation='relu'),
    keras.layers.Dense(32,activation='relu'),
    keras.layers.Dense(20,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])
cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
cnn.fit(X_train,y_train_cat,epochs=10)


# In[25]:


Y_predicted=cnn.predict(X_test)
Y_predicted[6]


# In[26]:


m=[np.argmax(x) for x in Y_predicted]


# In[27]:


m[:6]


# In[28]:


Y_test=Y_test.reshape(-1,)
Y_test[:6]


# In[29]:


ploty(X_test,Y_test,2)


# In[30]:


classes[m[2]]


# In[31]:


cnn.evaluate(X_test,y_test_cat)


# In[32]:


g=tf.math.confusion_matrix(labels=Y_test,predictions=m)
g
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(g,annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:





# In[33]:


joblib.dump(cnn, 'second_model.pkl')


# In[34]:


j=joblib.load('second_model.pkl')
o=joblib.load('first_model.pkl')


# In[35]:


j.predict(X_test)


# In[ ]:




