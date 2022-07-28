from tensorflow.keras.layers import Input, Activation, Dropout, Conv2D, MaxPooling2D, Reshape, UpSampling2D, concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
import matplotlib
matplotlib.use('Agg')
import os, sys, pylab, h5py
import matplotlib.pyplot as plt


###############################################################################
#                       PARAMETERS                                            #
###############################################################################

#kernel_size = (3,3)
#pooling_size = (2,2)
#dropout rate is set to 20%, meaning one in 5 inputs will be randomly excluded from each update cycle.
dropout_ratio=0.5
reg =  0.0002 #0.0010
im_shape = 128 # default shape value



###############################################################################
#                       ARCHITECTURE                                          #
###############################################################################



def unet_late_fusion(pretrained_weights = None, input_size = (im_shape,im_shape,1), classes=3):
	features = [8,16,32,64,128]
	# ENCODER

	# DWI
	input_dwi = Input(input_size)
	conv1_dwi = Conv2D(features[0], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(input_dwi)
	conv1_dwi = BatchNormalization()(conv1_dwi)
	conv1_dwi = Conv2D(features[0], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(conv1_dwi)
	conv1_dwi = BatchNormalization()(conv1_dwi)
	pool1_dwi = MaxPooling2D(pool_size=(2, 2))(conv1_dwi)
	conv2_dwi = Conv2D(features[1], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(pool1_dwi)
	conv2_dwi = BatchNormalization()(conv2_dwi)
	conv2_dwi = Conv2D(features[1], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(conv2_dwi)
	conv2_dwi = BatchNormalization()(conv2_dwi)
	pool2_dwi = MaxPooling2D(pool_size=(2, 2))(conv2_dwi)
	conv3_dwi = Conv2D(features[2], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(pool2_dwi)
	conv3_dwi = BatchNormalization()(conv3_dwi)
	conv3_dwi = Conv2D(features[2], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(conv3_dwi)
	conv3_dwi = BatchNormalization()(conv3_dwi)
	pool3_dwi = MaxPooling2D(pool_size=(2, 2))(conv3_dwi)
	conv4_dwi = Conv2D(features[3], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(pool3_dwi)
	conv4_dwi = BatchNormalization()(conv4_dwi)
	conv4_dwi = Conv2D(features[3], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(conv4_dwi)
	conv4_dwi = BatchNormalization()(conv4_dwi)
	drop4_dwi = Dropout(0.5)(conv4_dwi)
	pool4_dwi = MaxPooling2D(pool_size=(2, 2))(drop4_dwi)

	conv5_dwi = Conv2D(features[4], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(pool4_dwi)
	conv5_dwi = BatchNormalization()(conv5_dwi)
	conv5_dwi = Conv2D(features[4], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(conv5_dwi)
	conv5_dwi = BatchNormalization()(conv5_dwi)
	drop5_dwi = Dropout(0.5)(conv5_dwi)

	# ADC
	input_adc = Input(input_size)
	conv1_adc = Conv2D(features[0], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(input_adc)
	conv1_adc = BatchNormalization()(conv1_adc)
	conv1_adc = Conv2D(features[0], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(conv1_adc)
	conv1_adc = BatchNormalization()(conv1_adc)
	pool1_adc = MaxPooling2D(pool_size=(2, 2))(conv1_adc)
	conv2_adc = Conv2D(features[1], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(pool1_adc)
	conv2_adc = BatchNormalization()(conv2_adc)
	conv2_adc = Conv2D(features[1], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(conv2_adc)
	conv2_adc = BatchNormalization()(conv2_adc)
	pool2_adc = MaxPooling2D(pool_size=(2, 2))(conv2_adc)
	conv3_adc = Conv2D(features[2], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(pool2_adc)
	conv3_adc = BatchNormalization()(conv3_adc)
	conv3_adc = Conv2D(features[2], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(conv3_adc)
	conv3_adc = BatchNormalization()(conv3_adc)
	pool3_adc = MaxPooling2D(pool_size=(2, 2))(conv3_adc)
	conv4_adc = Conv2D(features[3], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(pool3_adc)
	conv4_adc = BatchNormalization()(conv4_adc)
	conv4_adc = Conv2D(features[3], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(conv4_adc)
	conv4_adc = BatchNormalization()(conv4_adc)
	drop4_adc = Dropout(0.5)(conv4_adc)
	pool4_adc = MaxPooling2D(pool_size=(2, 2))(drop4_adc)

	conv5_adc = Conv2D(features[4], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(pool4_adc)
	conv5_adc = BatchNormalization()(conv5_adc)
	conv5_adc = Conv2D(features[4], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(conv5_adc)
	conv5_adc = BatchNormalization()(conv5_adc)
	drop5_adc = Dropout(0.5)(conv5_adc)

	# TMAX
	input_tmax = Input(input_size)
	conv1_tmax = Conv2D(features[0], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(input_tmax)
	conv1_tmax = BatchNormalization()(conv1_tmax)
	conv1_tmax = Conv2D(features[0], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(conv1_tmax)
	conv1_tmax = BatchNormalization()(conv1_tmax)
	pool1_tmax = MaxPooling2D(pool_size=(2, 2))(conv1_tmax)
	conv2_tmax = Conv2D(features[1], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(pool1_tmax)
	conv2_tmax = BatchNormalization()(conv2_tmax)
	conv2_tmax = Conv2D(features[1], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(conv2_tmax)
	conv2_tmax = BatchNormalization()(conv2_tmax)
	pool2_tmax = MaxPooling2D(pool_size=(2, 2))(conv2_tmax)
	conv3_tmax = Conv2D(features[2], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(pool2_tmax)
	conv3_tmax = BatchNormalization()(conv3_tmax)
	conv3_tmax = Conv2D(features[2], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(conv3_tmax)
	conv3_tmax = BatchNormalization()(conv3_tmax)
	pool3_tmax = MaxPooling2D(pool_size=(2, 2))(conv3_tmax)
	conv4_tmax = Conv2D(features[3], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(pool3_tmax)
	conv4_tmax = BatchNormalization()(conv4_tmax)
	conv4_tmax = Conv2D(features[3], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(conv4_tmax)
	conv4_tmax = BatchNormalization()(conv4_tmax)
	drop4_tmax = Dropout(0.5)(conv4_tmax)
	pool4_tmax = MaxPooling2D(pool_size=(2, 2))(drop4_tmax)

	conv5_tmax = Conv2D(features[4], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(pool4_tmax)
	conv5_tmax = BatchNormalization()(conv5_tmax)
	conv5_tmax = Conv2D(features[4], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(conv5_tmax)
	conv5_tmax = BatchNormalization()(conv5_tmax)
	drop5_tmax = Dropout(0.5)(conv5_tmax)

	# LATE FUSION

	late_fusion = concatenate([drop5_dwi,drop5_adc,drop5_tmax])

	# DECODER



	up6 = Conv2D(features[3], (2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(UpSampling2D(size = (2,2))(late_fusion))
	merge6 = concatenate([drop4_dwi,drop4_adc,drop4_tmax,up6], axis = 3)
	conv6 = Conv2D(features[3], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(merge6)
	conv6 = BatchNormalization()(conv6)
	conv6 = Conv2D(features[3], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(conv6)
	conv6 = BatchNormalization()(conv6)

	up7 = Conv2D(features[2], (2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(UpSampling2D(size = (2,2))(conv6))
	merge7 = concatenate([conv3_dwi,conv3_adc,conv3_tmax,up7], axis = 3)
	conv7 = Conv2D(features[2], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(merge7)
	conv7 = BatchNormalization()(conv7)
	conv7 = Conv2D(features[2], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(conv7)
	conv7 = BatchNormalization()(conv7)

	up8 = Conv2D(features[1], (2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(UpSampling2D(size = (2,2))(conv7))
	merge8 = concatenate([conv2_dwi,conv2_adc,conv2_tmax,up8], axis = 3)
	conv8 = Conv2D(features[1], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(merge8)
	conv8 = BatchNormalization()(conv8)
	conv8 = Conv2D(features[1], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(conv8)
	conv8 = BatchNormalization()(conv8)

	up9 = Conv2D(features[0], (2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(UpSampling2D(size = (2,2))(conv8))
	merge9 = concatenate([conv1_dwi,conv1_adc,conv1_tmax,up9], axis = 3)
	conv9 = Conv2D(features[0], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(merge9)
	conv9 = BatchNormalization()(conv9)
	conv9 = Conv2D(features[0], (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(conv9)
	conv9 = BatchNormalization()(conv9)

	conv10 = Conv2D(classes, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(reg))(conv9)
	conv10 = Reshape((input_size[0],input_size[1], classes))(conv10)
	conv10 = Activation("softmax")(conv10)

	model = Model(inputs=[input_dwi,input_adc,input_tmax], outputs=conv10)

	if(pretrained_weights):
		model.load_weights(pretrained_weights)

	model.summary()

	return model


