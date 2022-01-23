import tensorflow as tf
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
import numpy as np
from bilinear import BilinearInterpolation
import cv2
import os
from sklearn.utils import shuffle
from keras.layers import Input
from keras import backend as K

img_path='stn_data/img_resize/'
gt_path='stn_data/img_resize_gt/'
list=os.listdir(img_path)

n_samples = len(list)
n_train = int(0.9* n_samples)
n_valid = n_samples - n_train
batch_size = 4
label = []
list_files = shuffle(list)

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def get_initial_weights(output_size):
    b=np.zeros((3,3),dtype='float32')
    b[0,0]=1
    b[1,1]=1
    b[2,2]=1
    W=np.zeros((output_size,9),dtype='float32')
    weights=[W,b.flatten()]
    return weights

def data_generator(list,batch_size=16,number_of_batches=None):
    counter=0
    while True:
        idx_start = batch_size * counter
        idx_end = batch_size * (counter + 1)
        x_batch = []
        y_batch = []
        for file in list[idx_start:idx_end]:
            img = cv2.imread(img_path+file.split('.')[0]+'.jpg')
            x_batch.append(img)
            img_gt=cv2.imread(gt_path+file.split('.')[0]+'.jpg')
            y_batch.append(img_gt)
        counter += 1
        x_train = np.array(x_batch,dtype=np.float32)
        y_train = np.array(y_batch,dtype=np.float32)

        yield x_train, y_train
        if (counter == number_of_batches):
            counter = 0

def TESTMODEL(input_shape=(535,397,3),sampling_size=(300,300)):
    # image=Input(shape=(535,397,3))
    base=tf.keras.applications.Xception(include_top=False,weights='imagenet',input_shape=input_shape,pooling='avg')
    locnet=Dense(50,activation='relu')(base.output)
    # weights=get_initial_weights(50)
    locnet=Dense(9)(locnet)
    x=BilinearInterpolation(sampling_size)([base.input,locnet])
    return Model(inputs=base.input,outputs=x)
            

sgd = tf.keras.optimizers.SGD(lr=0.9, decay=1e-6, momentum=0.5, nesterov=True)


model=TESTMODEL(input_shape=(535,397,3),sampling_size=(300,300))
model.summary()
model.compile(optimizer = sgd, 
              loss = 'huber', 
              metrics = ['accuracy'])


model.load_weights('D:/Heejoo/Research/stntestweights2/weights-improvement-55136-8.6471-0.9554.h5')
        
filepath="D:/Heejoo/Research/stntestweights2/weights-improvement-{epoch:02d}-{loss:.4f}-{accuracy:.4f}.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=1)
callbacks_list = [checkpoint]

model.fit(data_generator(list_files[:n_train], batch_size, number_of_batches= n_train // batch_size),
            steps_per_epoch=max(1, n_train//batch_size), initial_epoch =0, 
            validation_data= data_generator(list_files[n_train:], batch_size, number_of_batches= n_valid // batch_size),
            validation_steps=max(1, n_valid//batch_size),
            epochs=60000,
            callbacks=callbacks_list)