from metrics import compute_dice, compute_absolute_volume_difference, compute_absolute_lesion_difference, compute_lesion_f1_score
from data_loader import load_data, data_saver, data_loader
from model_late import unet_late_fusion
import sys
import numpy as np
import os
import nibabel as nib

def separate_patients(DataPath):
    with open(os.path.join(DataPath,'info_testing_patients.txt')) as f:
        txt = f.readlines()
        txt = [line.split('\t') for line in txt]
    nz_patients = []
    patients = []
    nz_start = []
    nz_end = []
    for i in range(len(txt)):
        patients.append(txt[i][0])
        nz_patients.append(int(txt[i][1]))
        nz_start.append(int(txt[i][2]))
        nz_end.append(int(txt[i][3][:-1]))
    return patients,nz_patients,nz_start,nz_end



WeightPath = sys.argv[1]
DataPath = sys.argv[2]
im_shape = 112

test_data_adc,test_label_original= data_loader(DataPath,'test_adc',im_shape)
test_data_dwi,test_label= data_loader(DataPath,'test_dwi',im_shape)
test_data_flair,test_label= data_loader(DataPath,'test_flair',im_shape)
mode = 'test_flair'



#create model with weights
unet_late = unet_late_fusion(input_size = (im_shape,im_shape,1), classes=2)
unet_late.load_weights(WeightPath)


output = unet_late.predict([test_data_dwi,test_data_adc,test_data_flair], batch_size=12)
np.save("prediction.npy", output)

patients,nz_patients,nz_start,nz_end = separate_patients(DataPath)
z = 1
ind_p = 0
all_z = 0
list_dc = list()
list_avd = list()
list_ald = list()
list_f1 = list()

with open(DataPath + '/'+ mode + '.txt') as f:
    txt = f.readlines()
    txt = [line.split('\t') for line in txt]
    
for i,p in enumerate(patients):
    for j in range(len(patients)):
        if p==os.path.basename(os.path.dirname(txt[j][0])):
            data_ref = nib.load(txt[j][1][:-1])
            print(txt[j][1][:-1])
            head = data_ref.header
            aff = data_ref.affine
            voxel_dim = np.prod((data_ref.header["pixdim"])[1:4])/1000
            mask_ref = data_ref.get_fdata()

    mask_pred = np.zeros((im_shape, im_shape, nz_patients[i]))
    
    if i > 0:
        all_z += nz_patients[i-1]-1

    for z in range(0, nz_patients[i]-1):
        mask_pred[:,:,z] = np.argmax(output[z+all_z], axis=-1)
  
    list_dc.append(compute_dice(mask_pred, mask_ref))  
    list_avd.append(compute_absolute_volume_difference(mask_pred, mask_ref, voxel_dim))
    list_ald.append(compute_absolute_lesion_difference(mask_ref, mask_pred))
    list_f1.append(compute_lesion_f1_score(mask_ref, mask_pred))


print(np.mean(np.array(list_dc)),np.std(np.array(list_dc)))
print(np.mean(np.array(list_avd)),np.std(np.array(list_avd)))
print(np.mean(np.array(list_ald)),np.std(np.array(list_ald)))
print(np.mean(np.array(list_f1)),np.std(np.array(list_f1)))





