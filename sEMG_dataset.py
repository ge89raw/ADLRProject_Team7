import numpy as np
import os
import urllib.request
import random
import pickle
from torch.utils.data import Dataset

# This file defines our own dataset class for the sEMG signals

class sEMGDataset(Dataset):
    def __init__(self, main_dir, mode = 'train', split = {'train': 0.6, 'val': 0.2, 'test': 0.2}, limit_files = None, transform = None):
        assert mode in ["train", "val", "test"], "wrong mode for dataset given"
        split_values = [v for k,v in split.items()]
        assert sum(split_values) == 1.0

        self.main_dir = main_dir
        self.split = split
        self.limit_files = limit_files
        self.transform = transform

        self.classes, self.class_to_idx = self._find_classes(self.main_dir)
        self.images, self.labels = self.make_dataset(
            directory=self.main_dir,
            class_to_idx=self.class_to_idx,
            mode = mode,
        )

    @staticmethod
    def _find_classes(directory):
        """
        Finds the class folders in a dataset
        :param directory: root directory of the dataset
        :returns: (classes, class_to_idx), where
          - classes is the list of all classes found
          - class_to_idx is a dict that maps class to label
        """
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def select_split(self, images, labels, mode):
        """
        Depending on the mode of the dataset, deterministically split it.
        
        :param images, a list containing paths to all images in the dataset
        :param labels, a list containing one label per image
        
        :returns (images, labels), where only the indices for the
            corresponding data split are selected.
        """
        fraction_train = self.split['train']
        fraction_val = self.split['val']
        num_samples = len(images)
        num_train = int(num_samples * fraction_train)
        num_valid = int(num_samples * fraction_val)
        
        np.random.seed(0)
        rand_perm = np.random.permutation(num_samples)
        
        if mode == 'train':
            idx = rand_perm[:num_train]
        elif mode == 'val':
            idx = rand_perm[num_train:num_train+num_valid]
        elif mode == 'test':
            idx = rand_perm[num_train+num_valid:]

        if self.limit_files:
            idx = idx[:self.limit_files]
        
        if isinstance(images, list): 
            return list(np.array(images)[idx]), list(np.array(labels)[idx])
        else: 
            return images[idx], list(np.array(labels)[idx])

    
    def make_dataset(self, directory, class_to_idx, mode):
        """
        Create the image dataset by preparaing a list of samples
        Images are sorted in an ascending order by class and file name
        :param directory: root directory of the dataset
        :param class_to_idx: A dict that maps classes to labels
        :returns: (images, labels) where:
            - images is a list containing paths to all images in the dataset, NOT the actual images
            - labels is a list containing one label per image
        """
        images, labels = [], []

        for target_class in sorted(class_to_idx.keys()):
            label = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            for root, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    if fname.endswith(".npy"):
                        path = os.path.join(root, fname)
                        images.append(path)
                        labels.append(label)

        images, labels = self.select_split(images, labels, mode)

        assert len(images) == len(labels)
        return images, labels

    def __len__(self):
        return len(self.images)

    @staticmethod
    def load_image_as_numpy(image_path):
        """Load image from image_path as numpy array"""
            #np.asarray(Image.open(image_path), dtype=float)
        return np.load(image_path)

    def __getitem__(self, index):

        imagen = self.load_image_as_numpy(self.images[index])
        if self.transform is not None:
            imagen = self.transform(imagen)
        label = int(self.labels[index])
        data_dict = {"image":imagen,"label":int(self.labels[index])}
        data_dict = [imagen, label] # Make it like a pytorch ImageFolderDataset output

        return data_dict