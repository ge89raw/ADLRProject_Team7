# ADLRProject

This repository contains the project files and code for the our Advanced Deep Learning for Robotics project, made by Maria Romeo and Jaume Gual (Team 7)

The project consists of two parts: 
(1) sEMG classification for user intention, and
(2) precise grasp generation through generative modeling, given and object and user's intention.

**FIRST PART**
Files related to this first sEMG classification part are:
- Database: folder containing the database of the raw sEMG signals to be converted to spectrograms.
- sEMG_preprocessing.ipynb: this file preprocesses the raw signals and generates the sEMG dataset consisting of spectrograms. The output of this file is "semg_dataset" folder.
- semg_dataset: folder containing dataset of spectrograms ready for classification.
- sEMG_dataset.py: modified dataset class for dataset and dataloader creation.
- sEMG_classifier.ipynb: file for ResNet-18 classifier training.

**SECOND PART**
Files related to the generative diffusion model are:
- dataset_grasps_full: folder that contains the dataset of grasps with which the model was trained.
- dataset_objects_full: folder that contains the dataset of objects with which the model was trained.
- dataset_val: folder containing the validation files on which the diffusion results were obtained.
- mjcf, thirdparty, utils: folders for support files, contianing 3D hand functions and objects.
- diffusion.ipynb: main jupyter file, full pipeline can be found here.
- ddpm_ours.py: diffusion model file. Can be called directly.
- grasp_object_dataset.py: file containing the dataset and dataloader generation for the diffusion model.
- positional_embeddings.py: support file for the diffusion model.
- model_XXX.pth: model files that were trained (3 files: PCA, full model, rotation_translation only)

**NOTE**
The folders containing the sEMG spectrogram database (sEMG_dataset), and the folders containing the grasps, objects and validation files for the generative model (dataset_grasps_full, dataset_objects_full, dataset_val) couldn't be uploaded to the GitHub.
As a result, these are publically available in the following Drive link: https://drive.google.com/drive/folders/1ZxY_UN6fsv8XJhpWT3txyy-MTtA8r1LS?usp=sharing


