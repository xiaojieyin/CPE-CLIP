import math
import os

import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import PIL
from PIL import Image


class BaseJsonDataset(Dataset):
    def __init__(self, image_path, json_path, mode='train', n_shot=None, transform=None):
        self.transform = transform
        self.image_path = image_path
        self.split_json = json_path
        self.mode = mode
        self.image_list = []
        self.label_list = []
        with open(self.split_json) as fp:
            splits = json.load(fp)
            samples = splits[self.mode]
            for s in samples:
                self.image_list.append(s[0])
                self.label_list.append(s[1])

        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_path, self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label).long()


fewshot_datasets = ['dtd', 'oxford_flowers', 'food101', 'stanford_cars', 'sun397',
                    'fgvc_aircraft', 'oxford_pets', 'caltech101', 'ucf101', 'eurosat',
                    'caltech256', 'cub', 'birdsnap']


def build_fewshot_dataset(set_id, root, transform, mode='train', n_shot=None):
    path_dict = {
        "oxford_flowers": ["jpg", root + "/split_zhou_OxfordFlowers.json"],
        "food101": ["images", root + "/split_zhou_Food101.json"],
        "dtd": ["images", root + "/split_zhou_DescribableTextures.json"],
        "oxford_pets": ["images", root + "/split_zhou_OxfordPets.json"],
        "sun397": ["", root + "/split_zhou_SUN397.json"],
        "caltech101": ["101_ObjectCategories", root + "/split_zhou_Caltech101.json"],
        "ucf101": ["", root + "/split_zhou_UCF101.json"],
        "stanford_cars": ["", root + "/split_zhou_StanfordCars.json"],
        "eurosat": ["images", root + "/split_zhou_EuroSAT.json"],
        "cub": ["images", root + "/split_CUB.json"],
        "caltech256": ["256_ObjectCategories", root + "/split_Caltech256.json"],
        "birdsnap": ["images", root + "/split_Birdsnap.json"],
    }

    if set_id.lower() == 'fgvc_aircraft':
        return Aircraft(root, mode, n_shot, transform)
    path_suffix, json_path = path_dict[set_id.lower()]
    image_path = os.path.join(root, path_suffix)
    return BaseJsonDataset(image_path, json_path, mode, n_shot, transform)


class Aircraft(Dataset):
    """ FGVC Aircraft dataset """

    def __init__(self, root, mode='train', n_shot=None, transform=None):
        self.transform = transform
        self.path = root
        self.mode = mode

        self.cname = []
        with open(os.path.join(self.path, "variants.txt"), 'r') as fp:
            self.cname = [l.replace("\n", "") for l in fp.readlines()]

        self.image_list = []
        self.label_list = []
        with open(os.path.join(self.path, 'images_variant_{:s}.txt'.format(self.mode)), 'r') as fp:
            lines = [s.replace("\n", "") for s in fp.readlines()]
            for l in lines:
                ls = l.split(" ")
                img = ls[0]
                label = " ".join(ls[1:])
                self.image_list.append("{}.jpg".format(img))
                self.label_list.append(self.cname.index(label))

        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, 'images', self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label).long()


class ImageNet_LT(Dataset):
    """Dataset class for the Imagenet-LT dataset."""

    def __init__(self, image_dir, transform, mode):
        """Initialize and preprocess the Imagenet-LT dataset."""
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []
        self.train_txt = os.path.join(self.image_dir, "ImageNet_LT_train.txt")
        self.val_txt = os.path.join(self.image_dir, "ImageNet_LT_val.txt")
        self.test_txt = os.path.join(self.image_dir, "ImageNet_LT_test.txt")
        self.cls_num_list_train = [0] * 1000
        self.cls_num_list_val = [0] * 1000
        self.cls_num_list_test = [0] * 1000
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        elif mode == "val":
            self.num_images = len(self.val_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):

        train_file = open(self.train_txt, "r")
        val_file = open(self.val_txt, "r")
        test_file = open(self.test_txt, "r")

        for elem in train_file.readlines():
            filename = elem.split(' ')[0]
            label = int(elem.split(' ')[1])
            self.train_dataset.append([filename, label])
            self.cls_num_list_train[label] += 1

        for elem in val_file.readlines():
            filename = elem.split(' ')[0]
            label = int(elem.split(' ')[1])
            self.val_dataset.append([filename, label])
            self.cls_num_list_val[label] += 1

        for elem in test_file.readlines():
            filename = elem.split(' ')[0]
            label = int(elem.split(' ')[1])
            self.test_dataset.append([filename, label])
            self.cls_num_list_test[label] += 1

    def __getitem__(self, index):
        """Return one image and its corresponding label."""
        if self.mode == "train":
            dataset = self.train_dataset
        elif self.mode == "val":
            dataset = self.val_dataset
        else:
            dataset = self.test_dataset

        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename)).convert('RGB')

        return self.transform(image), label

    def __len__(self):
        """Return the number of images."""
        return self.num_images

    def get_cls_num_list(self):
        if self.mode == "train":
            return self.cls_num_list_train
        elif self.mode == "val":
            return self.cls_num_list_val
        else:
            return self.cls_num_list_test


class SUN_LT(Dataset):

    def __init__(self, root, mode, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        txt = os.path.join(root, f"sun397_{mode}_lt.txt")
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0].replace('/SUN397/', '')))
                self.labels.append(int(line.split()[1]) - 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label
