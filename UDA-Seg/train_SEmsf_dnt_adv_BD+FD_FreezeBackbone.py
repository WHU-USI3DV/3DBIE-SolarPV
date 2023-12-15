import argparse
import os
import datetime
import logging
import time
import math
import numpy as np
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.nn as nn

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_model, build_adversarial_discriminator, build_feature_extractor, build_classifier, \
    build_binary_discriminator
from core.solver import adjust_learning_rate, adjust_learning_rate_use_stair, adjust_learning_rate_use_stair2
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
from core.models.layers import FrozenBatchNorm2d
from core.models import DN_transform, ConvDown, SELayer
from PIL import Image

def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def soft_label_cross_entropy(pred, soft_label, pixel_weights=None):
    N, C, H, W = pred.shape
    loss = -soft_label.float() * F.log_softmax(pred, dim=1)
    if pixel_weights is None:
        return torch.mean(torch.sum(loss, dim=1))
    return torch.mean(pixel_weights * torch.sum(loss, dim=1))


def train(cfg, local_rank, distributed):
    logger = logging.getLogger("FGDAL.trainer")
    logger.info("Start training")

    device = torch.device(cfg.MODEL.DEVICE)

    conv_down = ConvDown()
    conv_down.to(device)

    SE = SELayer(2048)
    SE.to(device)

    dnt = DN_transform()
    dnt.to(device)

    feature_extractor = build_feature_extractor(cfg, multi_scale=True)
    feature_extractor.to(device)

    classifier = build_classifier(cfg, 4)
    classifier.to(device)

    model_D = build_adversarial_discriminator(cfg)
    model_D.to(device)

    model_binary_D = build_binary_discriminator(cfg)
    model_binary_D.to(device)

    if local_rank == 0:
        print(feature_extractor)
        print(model_D)

    batch_size = cfg.SOLVER.BATCH_SIZE // 2
    if distributed:
        pg1 = torch.distributed.new_group(range(torch.distributed.get_world_size()))

        batch_size = int(cfg.SOLVER.BATCH_SIZE / torch.distributed.get_world_size()) // 2
        if not cfg.MODEL.FREEZE_BN:
            feature_extractor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(feature_extractor)
        feature_extractor = torch.nn.parallel.DistributedDataParallel(
            feature_extractor, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, process_group=pg1
        )
        pg2 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        classifier = torch.nn.parallel.DistributedDataParallel(
            classifier, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, process_group=pg2
        )
        pg3 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        model_D = torch.nn.parallel.DistributedDataParallel(
            model_D, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, process_group=pg3
        )
        pg4 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        conv_down = torch.nn.parallel.DistributedDataParallel(
            conv_down, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, process_group=pg4
        )
        pg5 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        model_binary_D = torch.nn.parallel.DistributedDataParallel(
            model_binary_D, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, process_group=pg5
        )
        pg6 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        SE = torch.nn.parallel.DistributedDataParallel(
            SE, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, process_group=pg6
        )
        pg7 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        dnt = torch.nn.parallel.DistributedDataParallel(
            dnt, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, process_group=pg7
        )
        torch.autograd.set_detect_anomaly(True)
        torch.distributed.barrier()

    optimizer_conv_down = torch.optim.SGD(conv_down.parameters(), lr=cfg.SOLVER.BASE_LR * 10, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_conv_down.zero_grad()

    optimizer_SE = torch.optim.SGD(SE.parameters(), lr=cfg.SOLVER.BASE_LR * 10, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_SE.zero_grad()

    optimizer_dnt = torch.optim.SGD(dnt.parameters(), lr=cfg.SOLVER.BASE_LR * 10, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_dnt.zero_grad()

    optimizer_fea = torch.optim.SGD(feature_extractor.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_fea.zero_grad()

    optimizer_cls = torch.optim.SGD(classifier.parameters(), lr=cfg.SOLVER.BASE_LR * 10, momentum=cfg.SOLVER.MOMENTUM,weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_cls.zero_grad()

    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=cfg.SOLVER.BASE_LR_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()

    optimizer_binary_D = torch.optim.Adam(model_binary_D.parameters(), lr=cfg.SOLVER.BASE_LR_D, betas=(0.9, 0.99))
    optimizer_binary_D.zero_grad()

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = local_rank == 0

    start_epoch = 0
    iteration = 0

    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        model_weights = checkpoint['feature_extractor'] if distributed else strip_prefix_if_present(
            checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(model_weights)
        classifier_weights = checkpoint['classifier'] if distributed else strip_prefix_if_present(
            checkpoint['classifier'], 'module.')
        classifier.load_state_dict(classifier_weights)
        if "model_D" in checkpoint:
            logger.info("Loading model_D from {}".format(cfg.resume))
            model_D_weights = checkpoint['model_D'] if distributed else strip_prefix_if_present(checkpoint['model_D'], 'module.')
            model_D.load_state_dict(model_D_weights)


    # Freeze parameters in feature_extractor for init
    for child in feature_extractor.children():
        for param in child.parameters():
            param.requires_grad = False

    src_train_data = build_dataset(cfg, mode='train', is_source=True)
    tgt_train_data = build_dataset(cfg, mode='train', is_source=False)

    if distributed:
        src_train_sampler = torch.utils.data.distributed.DistributedSampler(src_train_data)
        tgt_train_sampler = torch.utils.data.distributed.DistributedSampler(tgt_train_data)
    else:
        src_train_sampler = None
        tgt_train_sampler = None

    src_train_loader = torch.utils.data.DataLoader(src_train_data, batch_size=batch_size,
                                                   shuffle=(src_train_sampler is None), num_workers=4, pin_memory=True,
                                                   sampler=src_train_sampler, drop_last=True)
    tgt_train_loader = torch.utils.data.DataLoader(tgt_train_data, batch_size=batch_size,
                                                   shuffle=(tgt_train_sampler is None), num_workers=4, pin_memory=True,
                                                   sampler=tgt_train_sampler, drop_last=True)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    bce_loss = torch.nn.BCEWithLogitsLoss()

    max_iters = cfg.SOLVER.MAX_ITER
    source_label = 0
    target_label = 1
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    dnt.train()
    SE.train()
    conv_down.train()
    feature_extractor.train()
    classifier.train()
    model_D.train()
    model_binary_D.train()
    start_training_time = time.time()
    end = time.time()
    iter_stage_1 = 5000
    iter_stage_2 = 20000

    for i, ((src_input, src_label, src_name), (tgt_input, _, _)) in enumerate(zip(src_train_loader, tgt_train_loader)):
        # torch.distributed.barrier()
        data_time = time.time() - end

        if iteration == iter_stage_2 + 1:
            for child in feature_extractor.children():
                for param in child.parameters():
                    param.requires_grad = True
            # model_binary_D.cpu()
            torch.cuda.empty_cache()
            # for child in classifier.children():
            #     for param in child.parameters():
            #         param.requires_grad = True

        if iteration > iter_stage_2:
            current_lr_dnt = adjust_learning_rate_use_stair(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR * 10,
                                                            iteration - iter_stage_2, max_iters, power=cfg.SOLVER.LR_POWER)
            current_lr = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR, iteration - iter_stage_2, max_iters,
                                              power=cfg.SOLVER.LR_POWER)
            current_lr_D = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR_D, iteration - iter_stage_2,
                                                max_iters, power=cfg.SOLVER.LR_POWER)
            for index in range(len(optimizer_fea.param_groups)):
                optimizer_fea.param_groups[index]['lr'] = current_lr
            for index in range(len(optimizer_cls.param_groups)):
                optimizer_cls.param_groups[index]['lr'] = current_lr * 10
            for index in range(len(optimizer_dnt.param_groups)):
                optimizer_dnt.param_groups[index]['lr'] = current_lr_dnt
            for index in range(len(optimizer_conv_down.param_groups)):
                optimizer_conv_down.param_groups[index]['lr'] = current_lr
            for index in range(len(optimizer_SE.param_groups)):
                optimizer_SE.param_groups[index]['lr'] = current_lr
            for index in range(len(optimizer_D.param_groups)):
                optimizer_D.param_groups[index]['lr'] = current_lr_D

        elif iter_stage_1 < iteration <= iter_stage_2:
            current_lr_dnt = adjust_learning_rate_use_stair2(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR * 10,
                                                            iteration - iter_stage_1, max_iters, power=cfg.SOLVER.LR_POWER)
            current_lr = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR, iteration - iter_stage_1, max_iters,
                                              power=cfg.SOLVER.LR_POWER)
            current_lr_D = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR_D, iteration - iter_stage_1,
                                                max_iters, power=cfg.SOLVER.LR_POWER)
            for index in range(len(optimizer_binary_D.param_groups)):
                optimizer_binary_D.param_groups[index]['lr'] = current_lr_D
            for index in range(len(optimizer_cls.param_groups)):
                optimizer_cls.param_groups[index]['lr'] = current_lr * 10
            for index in range(len(optimizer_dnt.param_groups)):
                optimizer_dnt.param_groups[index]['lr'] = current_lr_dnt
            for index in range(len(optimizer_conv_down.param_groups)):
                optimizer_conv_down.param_groups[index]['lr'] = current_lr
            for index in range(len(optimizer_SE.param_groups)):
                optimizer_SE.param_groups[index]['lr'] = current_lr
        else:
            current_lr = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR, iteration, max_iters,
                                              power=cfg.SOLVER.LR_POWER)
            current_lr_D = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR_D, iteration,
                                                max_iters, power=cfg.SOLVER.LR_POWER)
            for index in range(len(optimizer_binary_D.param_groups)):
                optimizer_binary_D.param_groups[index]['lr'] = current_lr_D
            for index in range(len(optimizer_cls.param_groups)):
                optimizer_cls.param_groups[index]['lr'] = current_lr * 10
            for index in range(len(optimizer_conv_down.param_groups)):
                optimizer_conv_down.param_groups[index]['lr'] = current_lr
            for index in range(len(optimizer_SE.param_groups)):
                optimizer_SE.param_groups[index]['lr'] = current_lr

        #   torch.distributed.barrier()

        for child in SE.children():
            for param in child.parameters():
                param.requires_grad = True

        optimizer_cls.zero_grad()
        optimizer_conv_down.zero_grad()
        optimizer_SE.zero_grad()
        if iteration > iter_stage_2:
            optimizer_fea.zero_grad()
            optimizer_D.zero_grad()
        else:
            optimizer_binary_D.zero_grad()
        if iteration > iter_stage_2:
            optimizer_dnt.zero_grad()
        src_input = src_input.cuda(non_blocking=True)
        src_label = src_label.cuda(non_blocking=True).long()
        tgt_input = tgt_input.cuda(non_blocking=True)

        src_size = src_input.shape[-2:]
        tgt_size = tgt_input.shape[-2:]

        src_fea = conv_down(feature_extractor(src_input)[1])
        src_fea = SE(src_fea) * src_fea
        src_pred = classifier(src_fea, src_size)
        temperature = 1.8
        src_pred = src_pred.div(temperature)
        loss_seg = criterion(src_pred, src_label)
        loss_seg.backward()

        # torch.distributed.barrier()

        # generate soft labels
        src_soft_label = F.softmax(src_pred, dim=1).detach()
        src_soft_label[src_soft_label > 0.9] = 0.9
        for child in SE.children():
            for param in child.parameters():
                param.requires_grad = False
        if iteration > iter_stage_2:
            tgt_fea = conv_down(feature_extractor(dnt(tgt_input))[1])
        else:
            tgt_fea = conv_down(feature_extractor(tgt_input)[1])
        tgt_fea = SE(tgt_fea) * tgt_fea
        tgt_pred = classifier(tgt_fea, tgt_size)
        tgt_pred = tgt_pred.div(temperature)
        tgt_soft_label = F.softmax(tgt_pred, dim=1).detach()
        tgt_soft_label[tgt_soft_label > 0.9] = 0.9

        if iteration <= iter_stage_2:
            tgt_D_pred = model_binary_D(tgt_fea)
            loss_adv_tgt = 0.001 * bce_loss(tgt_D_pred, torch.zeros_like(tgt_D_pred))
        else:
            tgt_D_pred = model_D(tgt_fea, tgt_size)
            loss_adv_tgt = 0.001 * soft_label_cross_entropy(tgt_D_pred, torch.cat(
                (tgt_soft_label, torch.zeros_like(tgt_soft_label)), dim=1))
        loss_adv_tgt.backward()

        optimizer_SE.step()
        optimizer_cls.step()
        optimizer_conv_down.step()
        if iteration > iter_stage_2:
            optimizer_fea.step()
            optimizer_D.zero_grad()
        else:
            optimizer_binary_D.zero_grad()
        if iteration > iter_stage_2:
            optimizer_dnt.step()
        # torch.distributed.barrier()

        if iteration <= iter_stage_2:
            src_D_pred = model_binary_D(src_fea.detach())
            loss_D_src = 0.5 * bce_loss(src_D_pred, torch.zeros_like(src_D_pred))
        else:
            src_D_pred = model_D(src_fea.detach(), src_size)
            loss_D_src = 0.5 * soft_label_cross_entropy(src_D_pred, torch.cat((src_soft_label, torch.zeros_like(src_soft_label)), dim=1))
        loss_D_src.backward()

        if iteration <= iter_stage_2:
            tgt_D_pred = model_binary_D(tgt_fea.detach())
            loss_D_tgt = 0.5 * bce_loss(tgt_D_pred, torch.ones_like(tgt_D_pred))
        else:
            tgt_D_pred = model_D(tgt_fea.detach(), tgt_size)
            loss_D_tgt = 0.5 * soft_label_cross_entropy(tgt_D_pred, torch.cat((torch.zeros_like(tgt_soft_label), tgt_soft_label), dim=1))
        loss_D_tgt.backward()

        # torch.distributed.barrier()

        if iteration <= iter_stage_2:
            optimizer_binary_D.step()
        else:
            optimizer_D.step()

        meters.update(loss_seg=loss_seg.item())
        meters.update(loss_adv_tgt=loss_adv_tgt.item())
        meters.update(loss_D=(loss_D_src.item() + loss_D_tgt.item()))
        meters.update(loss_D_src=loss_D_src.item())
        meters.update(loss_D_tgt=loss_D_tgt.item())

        iteration = iteration + 1

        n = src_input.size(0)

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (cfg.SOLVER.STOP_ITER - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iters:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr_src: {lr_src:.6f}",
                        "lr: {lr:.6f}",
                        "lr_dnt:{lr_dnt:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr_src=current_lr,
                    lr=optimizer_fea.param_groups[0]["lr"],
                    lr_dnt=optimizer_dnt.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        if (iteration == cfg.SOLVER.MAX_ITER or iteration % cfg.SOLVER.CHECKPOINT_PERIOD == 0) and save_to_disk:
            filename = os.path.join(output_dir, "model_iter{:06d}.pth".format(iteration))
            torch.save({'iteration': iteration,
                        'DNT': dnt.state_dict(),
                        'SE': SE.state_dict(), 'ConvDown': conv_down.state_dict(),
                        'feature_extractor': feature_extractor.state_dict(), 'classifier': classifier.state_dict(),
                        'optimizer_fea': optimizer_fea.state_dict(), 'optimizer_cls': optimizer_cls.state_dict()},
                       filename)

        if iteration == cfg.SOLVER.MAX_ITER:
            break
        if iteration == cfg.SOLVER.STOP_ITER:
            break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (cfg.SOLVER.MAX_ITER)
        )
    )

    return conv_down, feature_extractor, classifier


def run_test(cfg, conv_down, feature_extractor, classifier, local_rank, distributed):
    logger = logging.getLogger("FGDAL.tester")
    if local_rank == 0:
        logger.info('>>>>>>>>>>>>>>>> Start Testing >>>>>>>>>>>>>>>>')

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    if distributed:
        conv_down, feature_extractor, classifier = conv_down.module, feature_extractor.module, classifier.module
    torch.cuda.empty_cache()  # TODO check if it helps
    dataset_name = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        mkdir(output_folder)

    test_data = build_dataset(cfg, mode='test', is_source=False)
    if distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    else:
        test_sampler = None
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=4,
                                              pin_memory=True, sampler=test_sampler)
    conv_down.eval()
    feature_extractor.eval()
    classifier.eval()
    end = time.time()
    with torch.no_grad():
        for i, (x, y, _) in enumerate(test_loader):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True).long()

            size = y.shape[-2:]

            output = classifier(conv_down(feature_extractor(x)[1]), size)
            output = output.max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output, y, cfg.MODEL.NUM_CLASSES,
                                                                  cfg.INPUT.IGNORE_LABEL)
            if distributed:
                torch.distributed.all_reduce(intersection), torch.distributed.all_reduce(
                    union), torch.distributed.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)
            end = time.time()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if local_rank == 0:
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(cfg.MODEL.NUM_CLASSES):
            logger.info(
                'Class_{} {} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, test_data.trainid2name[i], iou_class[i],
                                                                         accuracy_class[i]))


def main():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Training")
    parser.add_argument("-cfg",
                        "--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("FGDAL", output_dir, args.local_rank)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    conv_down, fea, cls = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        run_test(cfg, conv_down, fea, cls, args.local_rank, args.distributed)


if __name__ == "__main__":
    main()
