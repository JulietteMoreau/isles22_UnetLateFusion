# Late Fusion U-Net, ISLES 2022 participation

This GitHub gather the codes we used for our participation in ISLES 2022 segmentation challenge which goal is to segment stroke lesions from acute and subacute MR images from three modalities: DWI, ADC and FLAIR.

Link to the challenge: https://www.isles-challenge.org/

We proposed a late fusion U-Net that was trained with curriculum learning method based on the ratio of the volume of the bounding box of the lesion and the volume of the lesion.

## Training

The data is retrieved through .txt files in each path to the images and correponding masks are stored.
For each step there are three files -one for each modality- called train_<modality>-<step>.txt. A random example is provided in data directory.

Command to launch training
```bash
python main_late.py /path/to/data /output/path
```
In output you got the weights file for each step of curriculum learning, a loss curve, a diagram of the network and training history for each step of learning.

## Evaluation

Evaluation is made from weights file and save types of .txt files to access data called test_<modality>.txt. A .npy array is saved with all the predicted masks.

Command to evaluate
```bash
python prediction.py /path/to/weight/file /path/to/data
```

## Docker image

The files from which the final docker image proposed for the chalenge are in the docker directory. To export the same image run

```bash
./build.sh
./export.sh
```
