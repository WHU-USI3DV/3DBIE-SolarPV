import argparse
import math
import os
import logging
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.configs import cfg
from core.datasets import build_dataset
from core.datasets.build import build_transform
from core.models import build_model, build_adversarial_discriminator, build_feature_extractor, build_classifier
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU, get_color_pallete
from core.utils.logger import setup_logger
import rasterio
from core.models import DN_transform, ConvDown, SELayer
from core.datasets import transform


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def inference(dnt, SE, conv_down, feature_extractor, classifier, image, label, flip=True):
    size = label.shape[-2:]
    if flip:
        image = torch.cat([image, torch.flip(image, [3])], 0)
    with torch.no_grad():
        fea = conv_down(feature_extractor(dnt(image))[1])
        output = classifier(SE(fea)*fea)
    output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]
    return output.unsqueeze(dim=0)


def multi_scale_inference(dnt, SE, conv_down, feature_extractor, classifier, image, label, scales=[0.7,1.0,1.3], flip=True):
    output = None
    size = image.shape[-2:]
    for s in scales:
        x = F.interpolate(image, size=(int(size[0]*s), int(size[1]*s)), mode='bilinear', align_corners=True)
        pred = inference(dnt, SE, conv_down, feature_extractor, classifier, x, label, flip=False)
        if output is None:
            output = pred
        else:
            output = output + pred
        if flip:
            x_flip = torch.flip(x, [3])
            pred = inference(dnt, SE, conv_down, feature_extractor, classifier, x_flip, label, flip=False)
            output = output + pred.flip(3)
    if flip:
        return output/len(scales)/2
    return output/len(scales)


def test(cfg, img_path):
    logger = logging.getLogger("FGDAL.tester")
    logger.info("Start testing")
    device = torch.device(cfg.MODEL.DEVICE)

    dnt = DN_transform()
    dnt.to(device)

    conv_down = ConvDown()
    conv_down.to(device)

    SE = SELayer(2048)
    SE.to(device)

    feature_extractor = build_feature_extractor(cfg)
    feature_extractor.to(device)

    classifier = build_classifier(cfg, 4)
    classifier.to(device)

    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        feature_extractor_weights = strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(feature_extractor_weights)
        classifier_weights = strip_prefix_if_present(checkpoint['classifier'], 'module.')
        classifier.load_state_dict(classifier_weights)
        if "DNT" in checkpoint:
            dnt_weights = strip_prefix_if_present(checkpoint['DNT'], 'module.')
            dnt.load_state_dict(dnt_weights)
        if "SE" in checkpoint:
            SE_weights = strip_prefix_if_present(checkpoint['SE'], 'module.')
            SE.load_state_dict(SE_weights)
        if "ConvDown" in checkpoint:
            conv_down_weights = strip_prefix_if_present(checkpoint['ConvDown'], 'module.')
            conv_down.load_state_dict(conv_down_weights)

    dnt.eval()
    conv_down.eval()
    SE.eval()
    feature_extractor.eval()
    classifier.eval()

    torch.cuda.empty_cache()  # TODO check if it helps

    save_as_tif = 1
    img = rasterio.open(img_path, 'r')
    mask = np.zeros([img.height, img.width], dtype=np.uint8)
    window_size = 1300
    trans = build_transform(cfg, 'test', False)

    num_row = math.ceil(img.height / window_size)
    num_col = math.ceil(img.width / window_size)
    for i in tqdm(range(0, num_row)):
        for j in tqdm(range(0, num_col)):
            if i == num_row-1 and j != num_col-1:
                window_subset = ((i * window_size, img.height), (j * window_size, (j + 1) * window_size))
                sub_height = img.height - i * window_size
                sub_width = window_size
            elif j == num_col-1 and i != num_row-1:
                window_subset = ((i * window_size, (i + 1) * window_size), (j * window_size, img.width))
                sub_height = window_size
                sub_width = img.width - j * window_size
            elif j == num_col-1 and i == num_row-1:
                window_subset = ((i * window_size, img.height), (j * window_size, img.width))
                sub_height = img.height - i * window_size
                sub_width = img.width - j * window_size
            else:
                window_subset = ((i * window_size, (i + 1) * window_size), (j * window_size, (j + 1) * window_size))
                sub_height = window_size
                sub_width = window_size
            subset = np.rollaxis(img.read(window=window_subset), 0, 3).astype(np.float32)
            label = np.zeros([sub_height, sub_width])
            subset, label = trans(subset, label)
            subset = subset.unsqueeze(0).cuda(non_blocking=True)
            label = label.cuda(non_blocking=True).long()
            # pred = inference(dnt, SE, conv_down, feature_extractor, classifier, subset, label, flip=False)
            pred = multi_scale_inference(dnt, SE, conv_down, feature_extractor, classifier, subset, label, flip=True)
            pred = pred.cpu().numpy().squeeze()
            pred = pred.argmax(0)
            if i == num_row-1 and j != num_col-1:
                mask[i * window_size: img.height, j * window_size: (j + 1) * window_size] = pred
            elif j == num_col-1 and i != num_row-1:
                mask[i * window_size: (i + 1) * window_size, j * window_size: img.width] = pred
            elif j == num_col-1 and i == num_row-1:
                mask[i * window_size: img.height, j * window_size: img.width] = pred
            else:
                mask[i * window_size: (i + 1) * window_size, j * window_size: (j + 1) * window_size] = pred
    if save_as_tif == 0:
        mask = get_color_pallete(mask, "buildings")
        mask_path = img_path.replace(".tif", "_mask.png")
        mask.save(mask_path)
    else:
        np.savez_compressed("hp_mask.npz", mask)

        mask = get_color_pallete(mask, "buildings")
        mask.save("hp_mask.png")
        mask_path = img_path.replace(".tif", "_mask.tif")
        profile = img.profile
        profile.update(dtype=rasterio.uint8, count=1, compress='lzw')
        with rasterio.open(mask_path, 'w', **profile) as dst:
            dst.write(mask.astype(rasterio.uint8), 1)
            dst.close()

    print("It has been done!")


def main():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Testing")
    parser.add_argument("-cfg",
                        "--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        )
    parser.add_argument('--img_path', help='the path to load the img')
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

    test(cfg, args.img_path)


if __name__ == "__main__":
    main()
