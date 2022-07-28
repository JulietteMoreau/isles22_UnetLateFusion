import numpy as np
import itertools
import nibabel as nib #reading MR images
import os, math
import random as rd


def load_data(DataPath, mode, im_shape):
    data = []
    label = []
    info_patients = open(DataPath + "/info_testing_patients.txt","w+")
    with open(DataPath + '/'+ mode + '.txt') as f:
        txt = f.readlines()
        txt = [line.split('\t') for line in txt]
    for i in range(len(txt)):
        listez = []
        patient_name = os.path.basename(os.path.dirname(txt[i][0]))
        im = nib.load(txt[i][0])
        im = im.get_data()
        lab = nib.load(txt[i][1][:-1])
        lab = lab.get_data()
        nz_patient = im.shape[2]
        im[np.isnan(im)]=0
        for z in range(nz_patient-1):
            denom = (im[:,:,z].max()-im[:,:,z].min())
            if denom >0:
                im[:,:,z] = (im[:,:,z]-im[:,:,z].min())/denom
            else :
                im[:,:,z] = (im[:,:,z]-im[:,:,z].min())
            data.append(im[:,:,z])
            label.append(lab[:,:,z])
            if lab[:,:,z].max() == 1 :
                listez.append(z)
        info_patients.write("%s\t%d\t%d\t%d\n" % (patient_name,nz_patient,min(listez)+1,max(listez)+1))
        
    info_patients.close()
    data=np.array(data)
    label=np.array(label)
    return data, label


def data_saver(DataPath,mode,im_shape):
    train_data, train_label = load_data(DataPath,mode,im_shape)
    np.save(os.path.join(DataPath, mode+'_data'), train_data)
    np.save(os.path.join(DataPath, mode+'_label'), train_label)



def data_loader(data_dir,mode,im_shape):
    DataPath = data_dir
    # IMAGE
    if (os.path.isfile(os.path.join(DataPath,mode+'_data.npy')) and os.path.isfile(os.path.join(DataPath,mode+'_label.npy'))) == False :
        print('Npy files do not exist. Save new files.')
        data_saver(DataPath,mode,im_shape)
    data=np.load(DataPath+'/'+mode+'_data.npy')
    label=np.load(DataPath+'/'+mode+'_label.npy')
    return data,label
