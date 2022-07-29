# Histological methylation subtype classification

## System requirements
### Software dependencies
Most sofware dependencies are specified in the Docker File. Additionally a installed version of [CUDA](https://developer.nvidia.com/cuda-toolkit) is needed to enable GPU Support. If you want to use [Docker](https://www.docker.com/) please make sure Docker is properly installed and additionally [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).
The software was tested on a Linux Ubuntu Client. For visualization of Attention Maps [QuPath](https://qupath.github.io) is used. 
Additional package dependencies as defined in the Docker File are : [OpenSlide](https://openslide.org), [json](https://docs.python.org/3/library/json.html), [NumPy](https://numpy.org), [pandas](https://pandas.pydata.org), [pillow](https://python-pillow.org), [albumentations](https://albumentations.ai), [OpenCV](opencv.org/), [PyTorch](pytorch.org), [progress](https://pypi.org/project/progress/), [matplotlib](matplotlib.org), [scikit-learn](scikit-learn.org), [ptitprince](https://github.com/pog87/PtitPrince) and [seaborn](https://seaborn.pydata.org).
### Software versions
The used software versions are:  Linux Ubuntu /version 20.04.3 LTS), CUDA 11.2, Docker 20.10.12, Python 3.8, OpenSlide 1.12, json 2.0.9, NumPy 1.19.2, pandas 1.2.4, pillow 8.1.2, albumentations 1.0.0, OpenCV 4.5.2, PyTorch 1.8.1, progress 1.5, matplotlib 3.4.2, scikit-learn 0.24.2, seaborn 0.11.1, QuPath 0.2.3
### Required hardware
The software was tested on a machine with 128 GB RAM and an Intel Core i9-10900K CPU @ 3.70GHz x 20 and Ubuntu 20.04.3 LTS 64bit installed.
For speeding up network training a GPU with CUDA support is needed. We tested this project on a Nvidia GeForce RTX 3090 graphics card with 24 GB. 

## Installation guide
We provide a guide on how to set up the project using Docker.
### Instructions
Build a docker image you need to be in the folder with the Dockerfile.
```
docker build -t methnet:latest .
```
Alternatively you can pull the already build image from dockerhub using the following command.
```
docker pull kaischmid/methnet
```

Run the docker container.
```
docker run -it --rm methnet:latest
```


## Demo
### Instructions
This is how to run an classification of Meningioma benign-1 vs. Meningioma intermediate-A.
You need WSIs and a .csv holding patient information as specified in Data.
  data_directories is a list of folders holding WSIs
  csv_file is the csv holding patient information
  working_directory is the folder where information like tiling, features and results will be stored.

```
python3 main.py -d <data_directories> -c <csv_file> -w <working_directory>

python3 main.py --data_directories=<data_directories> --csv_file=<csv_file> --working_directory=<working_directory> 
```

Example call
```
python3 main.py -d ["./data/wsis/"] -c "./patients.csv" -w "./data/"

python3 main.py --data_directories=["./data/wsis/"] --csv_file="./patients.csv" --working_directory="./data/" 
```
### Data
#### Whole Slide images
Currently .ndpi and .svs files are supported. We expect the files to follow a specific naming convention in order to extract information about the slide.
The following naming convention is used: `ID_Staining_Scanner_Magnificationx.file`. 
E.g.: `B19-21_HE_Hamamatsu_40x.ndpi` or `1-2019_PHH3_Leica_20x.svs`
It is important that the ID is the same as in patient information.

#### Patient information
|ID|Histological ID|WHO Grade|Methylation Classes Meningioma-Classifier|Methylation class family member (EPIC) Meningioma-Classifier|probability class| probability subclass|Age|Sex|
|---|---|---|---|---|---|---|---|---|
|B1-21|B1-21|Atypical Meningiom (WHO Grade 2)|Meningioma intermediate|Meningioma intermediate-A|0.85|0.84|80|Female|
|B2-21|B2-21|Meningiom (WHO Grade 1)|Meningioma benign|Meningioma benign-1|0.9|0.74|60|Male|
### Expected output
The demo will generate multiple files in the working directory.
#### Tiling
A JSON per Tiling performed. The naming scheme is the following
```
ID_Staining_Scanner_Magnificationx_UsedMagnification_SizeX_SizeY_OverlapX_OverlapY.json
```
The first four informations are the same as in the WSI. Used magnification is the image pyramid level used. Size describes Tile size in pixel and Overlap the amount of shared pixels between two adjacent Tiles.
The JSON holds information in this scheme with x and y describing the Tiles position:
```
{"tiles":[{"x":0,"y":0}]}
```
#### splits
Holds a .csv file per class and iteration of Monte-Carlo cross-validation to remember the order of random shuffling this is to ensure the same split in train/validation/test sets. Naming scheme is Iteration_ClassLabel.csv.
#### Features
Holds one folder per image property. Inside one file holding the encoded features per WSI are stored.
#### Models
Holds a file per model trained with its parameters.
#### Results
Holds predicted scores per patient.
#### Attention maps
Holds an attention map per patient in test set.
#### Attention statistics
Holds files relevant for attention statistics as described in the paper.
### Expected runtime
The demo run might take around 2 hours including tiling and feature encoding.
Depending wheter WSIs are stored on SSD or HDD and the system specs the runtime might differ.

## Credits
Parts of the network were adapted from
[CLAM](https://github.com/mahmoodlab/CLAM) by [mahmoodlab](https://github.com/mahmoodlab)

## License
This code is made available under the GPLv3 License and is available for non-commercial academic purposes.

## Funding
This work was funded by the German Federal Ministry of Education and Research: MIRACUM (BMBF FKZ 01ZZ1801) and AI-RON (BMBF FKZ 01ZZ2017).

## Reference
