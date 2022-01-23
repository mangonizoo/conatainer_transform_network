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

vgg_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(300,300,3))

vgg_block5 = vgg_model.get_layer('block5_conv4').output
vgg = tf.keras.models.Model(vgg_model.input,  vgg_block5)

def psnr(target, ref):        
    return tf.image.psnr(target, ref, max_val=1.0)

def ssim(target, ref):        
    return tf.image.ssim(target, ref, max_val=1.0)
    
def content_loss(hr, sr):
    sr = tf.keras.applications.vgg19.preprocess_input(sr)
    hr = tf.keras.applications.vgg19.preprocess_input(hr)
    sr_features = vgg(sr) #/ 12.75
    hr_features = vgg(hr) #/ 12.75
    return tf.keras.metrics.mean_squared_error(hr_features, sr_features)

def get_initial_weights(output_size):
    b=np.zeros((3,3),dtype='float32')
    b[0,0]=1
    b[1,1]=1
    b[2,2]=1
    W=np.zeros((output_size,9),dtype='float32')
    weights=[W,b.flatten()]
    return weights

def TESTMODEL(input_shape=(535,397,3),sampling_size=(300,300)):
    # image=Input(shape=(535,397,3))
    base=tf.keras.applications.Xception(include_top=False,weights='imagenet',input_shape=input_shape,pooling='avg')
    locnet=Dense(50,activation='relu')(base.output)
    weights=get_initial_weights(50)
    locnet=Dense(9)(locnet)
    x=BilinearInterpolation(sampling_size)([base.input,locnet])
    return Model(inputs=base.input,outputs=[x,locnet])
            

model=TESTMODEL(input_shape=(535,397,3),sampling_size=(300,300))
model.summary()

imgpath='stn_data/img_resize/'
gtpath='stn_data/img_resize_gt/'
imglist=os.listdir(imgpath)

for name in imglist:
    imgname=imgpath+name
    img_gt_name=gtpath+name
    # modelpath='stntestweights/weights-improvement-17660-13.1261-0.9431.h5'
    modelpath='stntestweights2/weights-improvement-55136-8.6471-0.9554.h5'

    model.load_weights(modelpath)

    img=cv2.imread(imgname)
    # img=np.array([np.transpose(np.float32(oimg),(0,1,2))])
    img=np.array(img)
    img_gt=cv2.imread(img_gt_name)
    img=img[tf.newaxis,:]
    prediction,transform=model.predict(img)
    prediction=np.uint8(prediction[0])
    # print(prediction.shape)
    # print(transform[0])
    # cv2.imshow('predict',prediction)
    # cv2.imshow('original',cv2.resize(img,(300,300)))
    # cv2.imshow('gt',img_gt)
    # cv2.waitKey(0)
    # print(name)
    # cv2.imwrite('resultforpaper/'+name,prediction)
    img_gt=img_gt[tf.newaxis,:]
    prediction=prediction[tf.newaxis,:]
    ll=content_loss(img_gt,prediction)
    ll=np.sum(ll)
    ll/=(18*18)
    print(ll)

