import argparse
import os
import datetime
import logging
import time
import math
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import re

import torch
import torch.nn.functional as F

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_model, build_adversarial_discriminator, build_feature_extractor, build_classifier
from core.solver import adjust_learning_rate
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU, get_color_pallete
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
from core.models import DN_transform, ConvDown, SELayer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn as nn


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def test(cfg, saveres):
    logger = logging.getLogger("FGDAL.t_SNE")
    logger.info("Start t_SNE")
    device = torch.device(cfg.MODEL.DEVICE)

    feature_extractor = build_feature_extractor(cfg)
    feature_extractor.to(device)

    classifier = build_classifier(cfg, 4)
    classifier.to(device)

    msf = 0

    if msf == 1:
        conv_down = ConvDown()
        conv_down.to(device)

    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        feature_extractor_weights = strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(feature_extractor_weights)
        classifier_weights = strip_prefix_if_present(checkpoint['classifier'], 'module.')
        classifier.load_state_dict(classifier_weights)
        if "ConvDown" in checkpoint and msf == 1:
            conv_down_weights = strip_prefix_if_present(checkpoint['ConvDown'], 'module.')
            conv_down.load_state_dict(conv_down_weights)

    feature_extractor.eval()
    classifier.eval()
    if msf == 1:
        conv_down.to(device)

    test_data = build_dataset(cfg, mode='test', is_source=False)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=4,
                                              pin_memory=True, sampler=None)

    num_per_cls = 1000
    cls_num = 6
    feature_D = 2048
    hidden_feature = np.zeros([num_per_cls * cls_num, feature_D])
    cls_number_list = np.zeros(num_per_cls * cls_num)
    with torch.no_grad():
        for cls in range(cls_num):
            k = 0
            is_full = 0
            for batch in tqdm(test_loader):
                x, y, name = batch
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True).long()

                if msf == 1:
                    temp_feature = conv_down(feature_extractor(x)[1])
                else:
                    temp_feature = feature_extractor(x)
                temp_feature = F.interpolate(temp_feature, size=y.shape[-2:], mode='bilinear', align_corners=True)
                temp_feature = torch.squeeze(temp_feature.flatten(start_dim=2), dim=0).transpose(1, 0)

                temp_label = torch.squeeze(y.flatten(), dim=0).cpu().numpy()

                permutation = list(np.random.permutation(len(temp_label)))
                temp_feature = temp_feature[permutation]
                temp_label = temp_label[permutation]

                q = 0
                for i in range(0, len(temp_label)):
                    if temp_label[i] == cls:
                        index = cls * num_per_cls + k
                        hidden_feature[index, :] = temp_feature[i, :].cpu().numpy()
                        cls_number_list[index] = temp_label[i]
                        k = k+1
                        q = q+1
                    if k == num_per_cls:
                        is_full = 1
                        break
                    if q == 5:
                        print("image" + str(i))
                        break
                if is_full == 1:
                    print(cls)
                    break
                else:
                    continue

            print("all ok")
            print(len(cls_number_list))
            print(hidden_feature.shape)
            print(hidden_feature[0].shape)
        print(cls_number_list)

        trainid2name = {
            0: "background",
            1: "impervious surfaces",
            2: "car",
            3: "tree",
            4: "low vegetation",
            5: "building",

        }

        tsne = TSNE(n_components=2, init='pca')
        X_embedded = tsne.fit_transform(hidden_feature)

        plt.figure(figsize=(8, 8))
        for cl in range(6):
            indices = np.where(cls_number_list == cl)
            indices = indices[0]
            print(indices.shape)
            plt.scatter(X_embedded[indices, 0], X_embedded[indices, 1], s=10, label=trainid2name[cl])
        plt.legend(bbox_to_anchor=(0.8, 0.8), loc=3, borderaxespad=0)
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Testing")
    parser.add_argument("-cfg",
                        "--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        )
    parser.add_argument('--saveres', action="store_true",
                        help='save the result')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("FGDAL", save_dir, 0)
    logger.info(cfg)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    test(cfg, args.saveres)


if __name__ == "__main__":
    main()

