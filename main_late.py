# USAGE
# python main.py --dataset Houses-dataset/Houses\ Dataset/

# import the necessary packages
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import locale
import os, sys, csv, math
import json

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model
from tensorflow.keras.callbacks import LearningRateScheduler,EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data_loader import data_loader
from model_late import unet_late_fusion
from metric_and_losses import classes_dice_loss, multiclass_dice

import tensorflow as tf

tf.config.run_functions_eagerly(True)


###############################################################################
#                       LOAD DATA                                             #
###############################################################################


#set directories
data_dir = sys.argv[1]
out_dir = sys.argv[2]
im_shape = 112

#load data from text files containing the path to the images for the three modalities
#first fold with "easy" patients
train_data_dwi_1,train_label_1= data_loader(data_dir,'train_dwi_1',im_shape)
train_data_adc_1,train_label_1= data_loader(data_dir,'train_adc_1',im_shape)
train_data_flair_1,train_label_1= data_loader(data_dir,'train_flair_1',im_shape)

#second fold with fisrt fold and some more difficult patients
train_data_dwi_2,train_label_2= data_loader(data_dir,'train_dwi_2',im_shape)
train_data_adc_2,train_label_2= data_loader(data_dir,'train_adc_2',im_shape)
train_data_flair_2,train_label_2= data_loader(data_dir,'train_flair_2',im_shape)

#third fold
train_data_dwi_3,train_label_3= data_loader(data_dir,'train_dwi_3',im_shape)
train_data_adc_3,train_label_3= data_loader(data_dir,'train_adc_3',im_shape)
train_data_flair_3,train_label_3= data_loader(data_dir,'train_flair_3',im_shape)

#forth fold all patients
train_data_dwi_4,train_label_4= data_loader(data_dir,'train_dwi_4',im_shape)
train_data_adc_4,train_label_4= data_loader(data_dir,'train_adc_4',im_shape)
train_data_flair_4,train_label_4= data_loader(data_dir,'train_flair_4',im_shape)

#test set
test_data_dwi,test_label_original= data_loader(data_dir,'test_dwi',im_shape)
test_data_adc,test_label= data_loader(data_dir,'test_adc',im_shape)
test_data_flair,test_label= data_loader(data_dir,'test_flair',im_shape)







###############################################################################
#                       RESHAPE DATA                                          #
###############################################################################

def reshape_im(data_loaded):
    return np.reshape(data_loaded,(data_loaded.shape[0],data_loaded.shape[1],data_loaded.shape[2],1))


train_data_dwi_1 = reshape_im(train_data_dwi_1)
train_data_adc_1 = reshape_im(train_data_adc_1)
train_data_flair_1 = reshape_im(train_data_flair_1)
train_label_1 = to_categorical(train_label_1)

train_data_dwi_2 = reshape_im(train_data_dwi_2)
train_data_adc_2 = reshape_im(train_data_adc_2)
train_data_flair_2 = reshape_im(train_data_flair_2)
train_label_2 = to_categorical(train_label_2)

train_data_dwi_3 = reshape_im(train_data_dwi_3)
train_data_adc_3 = reshape_im(train_data_adc_3)
train_data_flair_3 = reshape_im(train_data_flair_3)
train_label_3 = to_categorical(train_label_3)

train_data_dwi_4 = reshape_im(train_data_dwi_4)
train_data_adc_4 = reshape_im(train_data_adc_4)
train_data_flair_4 = reshape_im(train_data_flair_4)
train_label_4 = to_categorical(train_label_4)

test_data_dwi = reshape_im(test_data_dwi)
test_data_adc = reshape_im(test_data_adc)
test_data_flair = reshape_im(test_data_flair)
test_label = to_categorical(test_label)



# ###############################################################################
# #                       PARAMETERS                                            #
# ###############################################################################

LearnRate = 0.001
Decay = 0.0005
# set small batch size: too large batch size leads to overfitting
batch_size = 12
nb_epoch = 200


###############################################################################
#                       MODEL                                                 #
###############################################################################

#classical runs
def run_mod(m,model_name,trainMRI,trainLABEL, epoch_max, step):
    # es = EarlyStopping(monitor='val_multiclass_dice',mode='max',min_delta=0.005,patience=100)

    # Compile model
    print("[INFO] Compiling model...")
    optimizer = Adam(lr=LearnRate,decay=Decay)
    m.compile(loss=classes_dice_loss, optimizer=optimizer, metrics=[multiclass_dice])

    # Model visualization
    plot_model(m, to_file=model_name+'.png', show_shapes=True)

    # Train model and save it
    print("[INFO] Training model...")
    history = m.fit(trainMRI, trainLABEL, batch_size=batch_size, epochs=epoch_max, validation_split=0.1)
    m.save_weights(model_name + '_weights_%s.hdf5' % (step))
    m.save(model_name + '_model_%s.h5' % (step))
    history_dict = history.history
    json.dump(history_dict, open(model_name + '_history_%s.json' % (step), 'w'))

#fianl run making predictions
def run_final(m,model_name,trainMRI,trainLABEL,testMRI, test_label_original, epoch_max, step):
    # Compile model
    print("[INFO] Compiling model...")
    optimizer = Adam(lr=LearnRate,decay=Decay)
    m.compile(loss=classes_dice_loss, optimizer=optimizer, metrics=[multiclass_dice])

    # Model visualization
    plot_model(m, to_file=model_name+'.png', show_shapes=True)

    # Train model and save it
    print("[INFO] Training model...")
    history = m.fit(trainMRI, trainLABEL, batch_size=batch_size, epochs=epoch_max, validation_split=0.1)
    m.save_weights(model_name + '_weights_%s.hdf5' % (step))
    m.save(model_name + '_model_%s.h5' % (step))
    history_dict = history.history
    json.dump(history_dict, open(model_name + '_history_%s.json' % (step), 'w'))
    
    
    f = open("unet_late_history_0.json")
    data_0 = json.load(f)
    f = open("unet_late_history_1.json")
    data_1 = json.load(f)
    f = open("unet_late_history_2.json")
    data_2 = json.load(f)
    f = open("unet_late_history_3.json")
    data_3 = json.load(f)
    
    loss = data_0["loss"] + data_1["loss"] + data_2["loss"] + data_3["loss"]
    val = data_0["val_loss"] + data_1["val_loss"] + data_2["val_loss"] + data_3["val_loss"]
    x = list(range(1,len(loss)+1))
    
    plt.rcParams['figure.figsize'] = [15,12]
    plt.plot(x, loss)
    plt.plot(x, val)
    plt.legend( ['training', 'validation'], fontsize = 25)
    plt.xlabel('epochs', fontsize = 25)
    plt.ylabel('loss', fontsize = 25)
    plt.savefig('loss.png')






###############################################################################
#                    RUN   MODEL                                              #
###############################################################################

nb_groupes = 4

if nb_epoch%nb_groupes==0:
    steps = [nb_epoch//nb_groupes]*nb_groupes
else:
    steps = [nb_epoch//nb_groupes]*(nb_groupes-1)+[nb_epoch//nb_groupes + nb_epoch%nb_groupes]
    
    
model_name = 'unet_late'


train_data = [[train_data_dwi_1,train_data_adc_1,train_data_flair_1], [train_data_dwi_2,train_data_adc_2,train_data_flair_2], [train_data_dwi_3,train_data_adc_3,train_data_flair_3], [train_data_dwi_4,train_data_adc_4,train_data_flair_4]]
train_label = [train_label_1, train_label_2,train_label_3, train_label_4]

for k in range(0,nb_groupes):
    if os.path.exists(model_name + '_weights_'+str(k-1)+'.hdf5'):
        pretrained_weights = model_name + '_weights_'+str(k-1)+'.hdf5'
    else: 
        pretrained_weights = None
    unet_late = unet_late_fusion(pretrained_weights = pretrained_weights, input_size = (im_shape,im_shape,1), classes=2)
    if k<nb_groupes-1:
        run_mod(unet_late,'unet_late',train_data[k], train_label[k], steps[k], k)
    else:
        run_final(unet_late,'unet_late',train_data[k], train_label[k], [test_data_dwi,test_data_adc,test_data_flair], test_label_original, steps[-1], k)



