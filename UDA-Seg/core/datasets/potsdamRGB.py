import os
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image


class potsdamRGBDataSet(data.Dataset):
    def __init__(
            self,
            data_root,
            data_list,
            max_iters=None,
            num_classes=6,
            split="train",
            transform=None,
            ignore_label=255,
            debug=False,
    ):
        self.split = split
        self.NUM_CLASS = num_classes
        self.data_root = data_root
        self.data_list = []
        with open(data_list, "r") as handle:
            content = handle.readlines()

        for fname in content:
            name = fname.strip()
            self.data_list.append(
                {
                    "img": os.path.join(
                        self.data_root, "images/%s" % name
                    ),
                    "label": os.path.join(
                        self.data_root, "labels/%s" % name
                    ),
                    "name": name,
                }
            )

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))

        self.id_to_trainid = {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            5: 4,
            6: 5,
        }
        self.trainid2name = {
            0: "background",
            1: "impervious surfaces",
            2: "car",
            3: "tree",
            4: "low vegetation",
            5: "building",
        }

        self.transform = transform

        self.ignore_label = ignore_label
        self.debug = debug

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        if self.debug:
            index = 0
        datafiles = self.data_list[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = np.array(Image.open(datafiles["label"]), dtype=np.float32)
        name = datafiles["name"]

        # re-assign labels to match the format of Cityscapes
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        # for k in self.trainid2name.keys():
        #     label_copy[label == k] = k
        label = Image.fromarray(label_copy)

        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label, name
