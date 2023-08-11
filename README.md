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
