import pandas as pd
import numpy as np
data= pd.read_csv(r'C:\Users\sbans\Desktop\bank.csv',header=None)
data=np.array(data)
label=data[:,-1]
x = data[:,:-1]
from keras import layers
from keras import models
from keras.utils import to_categorical
y=to_categorical(label)[:,1:]
size_of_x=np.arange(len(x))
np.random.shuffle(size_of_x)
x=x[size_of_x]
y=y[size_of_x]
model=models.Sequential()
a=int(input("Enter the number of neurons :"))

if a>1:
    b=int(input("Enter the hidden units for first neuron :"))
    model.add(layers.Dense(b,activation='relu',input_shape=(4,)))

while(a>2):
    c=int(input("Enter the Number of Hidden units :"))
    model.add(layers.Dense(c,activation='relu'))
    a=a-1
if a==1:
    model.add(layers.Dense(1,activation='sigmoid',input_shape=(4,)))
else:
    model.add(layers.Dense(1,activation='sigmoid',))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
history=model.fit(x,y,epochs=5,batch_size=16)