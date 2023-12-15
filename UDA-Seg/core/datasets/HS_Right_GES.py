import os
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
from skimage import io


class HS_Right_GES_Dataset(data.Dataset):
    def __init__(
            self,
            data_root,
            data_list,
            max_iters=None,
            num_classes=2,
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
                        self.data_root, name
                    ),
                    "label":
                        self.data_root + "/none"
                    ,
                    "name": name,
                }
            )

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))

        self.id_to_trainid = {
            0: 0,
            255: 1,
        }
        self.trainid2name = {
            0: "others",
            1: "building",
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

        image = io.imread(datafiles["img"]).astype(np.float32)
        # image = io.imread(datafiles["img"]).transpose(2, 0, 1)
        # image = Image.open(datafiles["img"]).convert('RGB')
        # label = np.load(datafiles["label"]).astype(np.float32)
        label = np.zeros([650, 650]).astype(np.float32)
        # label = np.array(Image.open(datafiles["label"]), dtype=np.float32)
        name = datafiles["name"]

        label[label == 255] = 1
        # # re-assign labels to match the format of Cityscapes
        # label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
        # for k, v in self.id_to_trainid.items():
        #     label_copy[label == k] = v
        # # for k in self.trainid2name.keys():
        # #     label_copy[label == k] = k
        # label = Image.fromarray(label_copy)

        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label, name
