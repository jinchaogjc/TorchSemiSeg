from __future__ import division
import os.path as osp
import os
import sys
import time
import argparse
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from config import config
from dataloader import get_train_loader
from network_isda import Network
from dataloader import VOC
from utils.init_func import init_weight, group_weight
from engine.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d
# from seg_opr.sync_bn import DataParallelModel, Reduce, BatchNorm2d
from tensorboardX import SummaryWriter
from ISDA import EstimatorCV, ISDALoss
import pdb

# from loss.criterion import CriterionDSN

try:
    from apex.parallel import DistributedDataParallel, SyncBatchNorm
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex .")

try:
    from azureml.core import Run

    azure = True
    run = Run.get_context()
except:
    azure = False
# print(f"azure: {azure}")

parser = argparse.ArgumentParser()

os.environ['MASTER_PORT'] = '169711'

if os.getenv('debug') is not None:
    is_debug = os.environ['debug']
else:
    is_debug = False

LAMBDA_0 = config.lambda0  # select from {1, 2.5, 5, 7.5, 10}
# NUM_STEPS = 40000
NUM_STEPS = config.num_step

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    # print("sys.argv", sys.argv)
    # print(args)

    cudnn.benchmark = True

    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader + unsupervised data loader
    train_loader, train_sampler = get_train_loader(engine, VOC,
                                                   train_source=config.train_source, unsupervised=False)
    unsupervised_train_loader, \
    unsupervised_train_sampler = get_train_loader(engine, VOC,
                                                  train_source=config.unsup_source, unsupervised=True)

    if engine.distributed and (engine.local_rank == 0):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        logger = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)

    # config network and criterion
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
    criterion_csst = nn.MSELoss(reduction='mean')
    # criterion_isda = CriterionDSN()

    if engine.distributed:
        BatchNorm2d = SyncBatchNorm
    # else:
    #     BatchNorm2d = SyncBatchNorm

    # define and init the model
    model = Network(config.num_classes, criterion=criterion,
                    pretrained_model=config.pretrained_model,
                    norm_layer=BatchNorm2d)

    init_weight(model.branch1.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')
    init_weight(model.branch2.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    # define the learning rate
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr * engine.world_size

    # define the two optimizers
    params_list_l = []
    params_list_l = group_weight(params_list_l, model.branch1.backbone,
                                 BatchNorm2d, base_lr)
    for module in model.branch1.business_layer:
        params_list_l = group_weight(params_list_l, module, BatchNorm2d,
                                     base_lr)  # head lr * 10

    optimizer_l = torch.optim.SGD(params_list_l,
                                  lr=base_lr,
                                  momentum=config.momentum,
                                  weight_decay=config.weight_decay)

    params_list_r = []
    params_list_r = group_weight(params_list_r, model.branch2.backbone,
                                 BatchNorm2d, base_lr)
    for module in model.branch2.business_layer:
        params_list_r = group_weight(params_list_r, module, BatchNorm2d,
                                     base_lr)  # head lr * 10

    optimizer_r = torch.optim.SGD(params_list_r,
                                  lr=base_lr,
                                  momentum=config.momentum,
                                  weight_decay=config.weight_decay)

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)
    NUM_STEPS = total_iteration

    if engine.distributed:
        print('distributed !!')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model)
        device = "cuda"
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DataParallelModel(model, device_ids=engine.devices)
        model.to(device)

    isda_augmentor_1 = ISDALoss(256, config.num_classes, device)
    # # print("isda_augmentor_1", isda_augmentor_1)
    isda_augmentor_2 = ISDALoss(256, config.num_classes, device)
    # # print("isda_augmentor_2", isda_augmentor_2)

    engine.register_state(dataloader=train_loader, model=model,
                          optimizer_l=optimizer_l, optimizer_r=optimizer_r)

    if engine.continue_state_object:
        engine.restore_checkpoint()

    model.train()
    global_iteration = 0
    # print('begin train1-l-isda1-4g-8b')
    print(
        f"exp_name:{config.exp_name}, exp_num:{config.exp_num}, labeled_ratio:{config.labeled_ratio}, nepochs:{config.nepochs} ")

    for epoch in range(engine.state.epoch, config.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'

        if is_debug:
            pbar = tqdm(range(500), file=sys.stdout, bar_format=bar_format)
        else:
            pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)

        dataloader = iter(train_loader)
        unsupervised_dataloader = iter(unsupervised_train_loader)

        sum_loss_sup = 0
        sum_loss_sup_r = 0
        sum_cps = 0

        ''' supervised part '''
        for idx in pbar:
            global_iteration += 1

            optimizer_l.zero_grad()
            optimizer_r.zero_grad()
            engine.update_iteration(epoch, idx)
            # if idx > 10:
            #     break
            start_time = time.time()

            minibatch = dataloader.next()
            unsup_minibatch = unsupervised_dataloader.next()
            imgs = minibatch['data']
            gts = minibatch['label']
            unsup_imgs = unsup_minibatch['data']
            imgs = imgs.cuda(non_blocking=True)
            unsup_imgs = unsup_imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)

            b, c, h, w = imgs.shape
            # print("imgs.shape:", imgs.shape, "b,c,h,w:", b, c, h, w)
            # _, pred_sup_l = model(imgs, step=1)
            # _, pred_unsup_l = model(unsup_imgs, step=1)
            # _, pred_sup_r = model(imgs, step=2)
            # _, pred_unsup_r = model(unsup_imgs, step=2)

            #### add isda loss for sup_label*****************************************************
            ratio = LAMBDA_0 * global_iteration / NUM_STEPS
            if config.exp_num == -1:
                pass

            elif config.exp_num == 1:
                x_sup_l, feature_x_sup_l, pred_sup_l = model(imgs, step=1, isda=True)
                x_sup_r, feature_x_sup_r, pred_sup_r = model(imgs, step=2, isda=True)
                # x_dsn_sup_r, feature_x_dsn_sup_r, feature_x_sup_r, pred_sup_r = \
                #     model(imgs, step=2, isda=True)

                # pdb.set_trace()
                x_isda_l = isda_augmentor_1(feature_x_sup_l, model.module.branch1.final_conv_1, x_sup_l, gts, ratio)
                # print(x_isda.shape)
                x_isda_r = isda_augmentor_2(feature_x_sup_r, model.module.branch2.final_conv_1, x_sup_r, gts, ratio)
                # # print(x_isda.shape)
                # # print(gts.shape)
                # loss_isda = criterion_isda([x_isda, x_dsn_isda], gts)
                _, h, w = gts.shape
                x_isda_pred_l = F.interpolate(input=x_isda_l, size=(h, w), mode='bilinear', align_corners=True)
                loss_isda = criterion(x_isda_pred_l, gts)
                # print("loss_isda", loss_isda)
                x_isda_pred_r = F.interpolate(input=x_isda_r, size=(h, w), mode='bilinear', align_corners=True)
                loss_isda += criterion(x_isda_pred_r, gts)

                x_sup_l, feature_x_sup_l, pred_sup_l = model(imgs, step=1, isda=True)
                x_sup_r, feature_x_sup_r, pred_sup_r = model(imgs, step=2, isda=True)
                _, max_l = torch.max(pred_sup_l, dim=1)
                _, max_r = torch.max(pred_sup_r, dim=1)
                max_l = max_l.long()
                max_r = max_r.long()
                x_isda_l = isda_augmentor_1(feature_x_sup_l, model.module.branch1.final_conv_1, x_sup_l, max_r, ratio)
                # print(x_isda.shape)
                x_isda_r = isda_augmentor_2(feature_x_sup_r, model.module.branch2.final_conv_1, x_sup_r, max_l, ratio)
                # # print(x_isda.shape)
                # # print(gts.shape)
                # loss_isda = criterion_isda([x_isda, x_dsn_isda], gts)

                x_isda_pred_l = F.interpolate(input=x_isda_l, size=(h, w), mode='bilinear', align_corners=True)
                loss_isda_labeled = criterion(x_isda_pred_l, max_r)
                # print("loss_isda", loss_isda)
                x_isda_pred_r = F.interpolate(input=x_isda_r, size=(h, w), mode='bilinear', align_corners=True)
                loss_isda_labeled += criterion(x_isda_pred_r, max_l)

                x_unsup_l, feature_x_unsup_l, pred_unsup_l = model(unsup_imgs, step=1, isda=True)
                x_unsup_r, feature_x_unsup_r, pred_unsup_r = model(unsup_imgs, step=2, isda=True)

                _, max_l = torch.max(pred_unsup_l, dim=1)
                _, max_r = torch.max(pred_unsup_r, dim=1)
                max_l = max_l.long()
                max_r = max_r.long()

                x_isda_unsup_l = isda_augmentor_1(feature_x_unsup_l, model.module.branch1.final_conv_1, x_unsup_l,
                                                  max_r, ratio)
                x_isda_unsup_r = isda_augmentor_2(feature_x_unsup_r, model.module.branch2.final_conv_1, x_unsup_r,
                                                  max_l, ratio)

                x_isda_pred_unsup_l = F.interpolate(input=x_isda_unsup_l, size=(h, w), mode='bilinear',
                                                    align_corners=True)
                loss_isda_unlabeled = criterion(x_isda_pred_unsup_l, max_l)

                x_isda_pred_unsup_r = F.interpolate(input=x_isda_unsup_r, size=(h, w), mode='bilinear',
                                                    align_corners=True)
                loss_isda_unlabeled += criterion(x_isda_pred_unsup_r, max_r)


            elif config.exp_num == 2:
                # # ******************************************************************************
                #
                # ratio = LAMBDA_0 * global_iteration / NUM_STEPS
                x_sup_l, feature_x_sup_l, pred_sup_l = model(imgs, step=1, isda=True)
                x_sup_r, feature_x_sup_r, pred_sup_r = model(imgs, step=2, isda=True)
                _, max_l = torch.max(pred_sup_l, dim=1)
                _, max_r = torch.max(pred_sup_r, dim=1)
                max_l = max_l.long()
                max_r = max_r.long()
                x_isda_l = isda_augmentor_1(feature_x_sup_l, model.module.branch1.final_conv_1, x_sup_l, max_r, ratio)
                # print(x_isda.shape)
                x_isda_r = isda_augmentor_2(feature_x_sup_r, model.module.branch2.final_conv_1, x_sup_r, max_l, ratio)
                # # print(x_isda.shape)
                # # print(gts.shape)
                # loss_isda = criterion_isda([x_isda, x_dsn_isda], gts)
                _, h, w = max_l.shape
                x_isda_pred_l = F.interpolate(input=x_isda_l, size=(h, w), mode='bilinear', align_corners=True)
                loss_isda_labeled = criterion(x_isda_pred_l, max_r)
                # print("loss_isda", loss_isda)
                x_isda_pred_r = F.interpolate(input=x_isda_r, size=(h, w), mode='bilinear', align_corners=True)
                loss_isda_labeled += criterion(x_isda_pred_r, max_l)

                # *******************************************
                x_unsup_l, feature_x_unsup_l, pred_unsup_l = model(unsup_imgs, step=1, isda=True)
                x_unsup_r, feature_x_unsup_r, pred_unsup_r = model(unsup_imgs, step=2, isda=True)

                _, max_l = torch.max(pred_unsup_l, dim=1)
                _, max_r = torch.max(pred_unsup_r, dim=1)
                max_l = max_l.long()
                max_r = max_r.long()

                x_isda_unsup_l = isda_augmentor_1(feature_x_unsup_l, model.module.branch1.final_conv_1, x_unsup_l,
                                                  max_r, ratio)
                x_isda_unsup_r = isda_augmentor_2(feature_x_unsup_r, model.module.branch2.final_conv_1, x_unsup_r,
                                                  max_l, ratio)

                _, h, w = max_l.shape
                x_isda_pred_unsup_l = F.interpolate(input=x_isda_unsup_l, size=(h, w), mode='bilinear',
                                                    align_corners=True)
                loss_isda_unlabeled = criterion(x_isda_pred_unsup_l, max_r)

                x_isda_pred_unsup_r = F.interpolate(input=x_isda_unsup_r, size=(h, w), mode='bilinear',
                                                    align_corners=True)
                loss_isda_unlabeled += criterion(x_isda_pred_unsup_r, max_l)

            elif config.exp_num == 3:
                # print("experiment number is 2")

                x_sup_l, feature_x_sup_l, pred_sup_l = model(imgs, step=1, isda=True)
                x_sup_r, feature_x_sup_r, pred_sup_r = model(imgs, step=2, isda=True)
                # x_dsn_sup_r, feature_x_dsn_sup_r, feature_x_sup_r, pred_sup_r = \
                #     model(imgs, step=2, isda=True)

                x_isda_l = isda_augmentor_1(feature_x_sup_l, model.module.branch1.final_conv_1, x_sup_l, gts, ratio)
                # print(x_isda.shape)
                x_isda_r = isda_augmentor_2(feature_x_sup_r, model.module.branch2.final_conv_1, x_sup_r, gts, ratio)
                # # print(x_isda.shape)
                # # print(gts.shape)
                # loss_isda = criterion_isda([x_isda, x_dsn_isda], gts)
                _, h, w = gts.shape
                x_isda_pred_l = F.interpolate(input=x_isda_l, size=(h, w), mode='bilinear', align_corners=True)
                loss_isda = criterion(x_isda_pred_l, gts)
                # print("loss_isda", loss_isda)
                x_isda_pred_r = F.interpolate(input=x_isda_r, size=(h, w), mode='bilinear', align_corners=True)
                loss_isda += criterion(x_isda_pred_r, gts)

                ####**add isda loss unsup label***************************************************
                #
                # _, pred_sup_l = model(imgs, step=1, isda=True)
                # _, pred_unsup_l = model(unsup_imgs, step=1, isda=True)
                x_unsup_l, feature_x_unsup_l, pred_unsup_l = model(unsup_imgs, step=1, isda=True)
                x_unsup_r, feature_x_unsup_r, pred_unsup_r = model(unsup_imgs, step=2, isda=True)

                # ratio = LAMBDA_0 * global_iteration / NUM_STEPS

                _, max_l = torch.max(pred_unsup_l, dim=1)
                _, max_r = torch.max(pred_unsup_r, dim=1)
                max_l = max_l.long()
                max_r = max_r.long()

                x_isda_unsup_l = isda_augmentor_1(feature_x_unsup_l, model.module.branch1.final_conv_1, x_unsup_l,
                                                  max_r, ratio)
                x_isda_unsup_r = isda_augmentor_2(feature_x_unsup_r, model.module.branch2.final_conv_1, x_unsup_r,
                                                  max_l, ratio)

                _, h, w = max_l.shape
                x_isda_pred_unsup_l = F.interpolate(input=x_isda_unsup_l, size=(h, w), mode='bilinear',
                                                    align_corners=True)
                loss_isda_unlabeled = criterion(x_isda_pred_unsup_l, max_r)

                x_isda_pred_unsup_r = F.interpolate(input=x_isda_unsup_r, size=(h, w), mode='bilinear',
                                                    align_corners=True)
                loss_isda_unlabeled += criterion(x_isda_pred_unsup_r, max_l)

                ####*****************************************************
            elif config.exp_num == 4:
                # print("experiment number is 4")

                x_sup_l, feature_x_sup_l, pred_sup_l = model(imgs, step=1, isda=True)
                x_sup_r, feature_x_sup_r, pred_sup_r = model(imgs, step=2, isda=True)
                # x_dsn_sup_r, feature_x_dsn_sup_r, feature_x_sup_r, pred_sup_r = \
                #     model(imgs, step=2, isda=True)

                x_isda_l = isda_augmentor_1(feature_x_sup_l, model.module.branch1.final_conv_1, x_sup_l, gts, ratio)
                # print(x_isda.shape)
                x_isda_r = isda_augmentor_2(feature_x_sup_r, model.module.branch2.final_conv_1, x_sup_r, gts, ratio)
                # # print(x_isda.shape)
                # # print(gts.shape)
                # loss_isda = criterion_isda([x_isda, x_dsn_isda], gts)
                _, h, w = gts.shape
                x_isda_pred_l = F.interpolate(input=x_isda_l, size=(h, w), mode='bilinear', align_corners=True)
                loss_isda = criterion(x_isda_pred_l, gts)
                # print("loss_isda", loss_isda)
                x_isda_pred_r = F.interpolate(input=x_isda_r, size=(h, w), mode='bilinear', align_corners=True)
                loss_isda += criterion(x_isda_pred_r, gts)

                # # ******************************************************************************
                #
                # ratio = LAMBDA_0 * global_iteration / NUM_STEPS
                # x_sup_l, feature_x_sup_l, pred_sup_l = model(imgs, step=1, isda=True)
                # x_sup_r, feature_x_sup_r, pred_sup_r = model(imgs, step=2, isda=True)
                _, max_l = torch.max(pred_sup_l, dim=1)
                _, max_r = torch.max(pred_sup_r, dim=1)
                max_l = max_l.long()
                max_r = max_r.long()
                x_isda_l = isda_augmentor_1(feature_x_sup_l, model.module.branch1.final_conv_1, x_sup_l, max_r, ratio)
                # print(x_isda.shape)
                x_isda_r = isda_augmentor_2(feature_x_sup_r, model.module.branch2.final_conv_1, x_sup_r, max_l, ratio)
                # # print(x_isda.shape)
                # # print(gts.shape)
                # loss_isda = criterion_isda([x_isda, x_dsn_isda], gts)
                _, h, w = max_l.shape
                x_isda_pred_l = F.interpolate(input=x_isda_l, size=(h, w), mode='bilinear', align_corners=True)
                loss_isda_labeled = criterion(x_isda_pred_l, max_r)
                # print("loss_isda", loss_isda)
                x_isda_pred_r = F.interpolate(input=x_isda_r, size=(h, w), mode='bilinear', align_corners=True)
                loss_isda_labeled += criterion(x_isda_pred_r, max_l)
                _, pred_unsup_l = model(unsup_imgs, step=1)
                _, pred_unsup_r = model(unsup_imgs, step=2)
                #
                # # ******************************************************************************


            elif config.exp_num == 5:
                # print("experiment number is 2")
                # _, pred_sup_l = model(imgs, step=1, isda=True)
                # _, pred_unsup_l = model(unsup_imgs, step=1, isda=True)
                # x_sup_l, x_dsn_sup_l, feature_x_dsn_sup_l, feature_x_sup_l, pred_sup_l = \
                #     model(imgs, step=1, isda=True)
                x_sup_l, feature_x_sup_l, pred_sup_l = model(imgs, step=1, isda=True)
                x_sup_r, feature_x_sup_r, pred_sup_r = model(imgs, step=2, isda=True)
                # x_dsn_sup_r, feature_x_dsn_sup_r, feature_x_sup_r, pred_sup_r = \
                #     model(imgs, step=2, isda=True)

                x_isda_l = isda_augmentor_1(feature_x_sup_l, model.module.branch1.final_conv_1, x_sup_l, gts, ratio)
                # print(x_isda.shape)
                x_isda_r = isda_augmentor_2(feature_x_sup_r, model.module.branch2.final_conv_1, x_sup_r, gts, ratio)
                # # print(x_isda.shape)

                # # print(gts.shape)
                # loss_isda = criterion_isda([x_isda, x_dsn_isda], gts)
                _, h, w = gts.shape
                x_isda_pred_l = F.interpolate(input=x_isda_l, size=(h, w), mode='bilinear', align_corners=True)
                loss_isda = criterion(x_isda_pred_l, gts)
                # print("loss_isda", loss_isda)
                x_isda_pred_r = F.interpolate(input=x_isda_r, size=(h, w), mode='bilinear', align_corners=True)
                loss_isda += criterion(x_isda_pred_r, gts)
                # print("loss_isda", loss_isda)
                # print(loss_isda)
                # input()
                #
                # reduce_loss = engine.all_reduce_tensor(loss)
                # print(reduce_loss)
                # ####*****************************************************
                _, pred_unsup_l = model(unsup_imgs, step=1)
                _, pred_unsup_r = model(unsup_imgs, step=2)

            elif config.exp_num == 6:
                # # ******************************************************************************
                #
                # ratio = LAMBDA_0 * global_iteration / NUM_STEPS
                x_sup_l, feature_x_sup_l, pred_sup_l = model(imgs, step=1, isda=True)
                x_sup_r, feature_x_sup_r, pred_sup_r = model(imgs, step=2, isda=True)
                _, max_l = torch.max(pred_sup_l, dim=1)
                _, max_r = torch.max(pred_sup_r, dim=1)
                max_l = max_l.long()
                max_r = max_r.long()
                x_isda_l = isda_augmentor_1(feature_x_sup_l, model.module.branch1.final_conv_1, x_sup_l, max_r, ratio)
                # print(x_isda.shape)
                x_isda_r = isda_augmentor_2(feature_x_sup_r, model.module.branch2.final_conv_1, x_sup_r, max_l, ratio)
                # # print(x_isda.shape)
                # # print(gts.shape)
                # loss_isda = criterion_isda([x_isda, x_dsn_isda], gts)
                _, h, w = max_l.shape
                x_isda_pred_l = F.interpolate(input=x_isda_l, size=(h, w), mode='bilinear', align_corners=True)
                loss_isda_labeled = criterion(x_isda_pred_l, max_r)
                # print("loss_isda", loss_isda)
                x_isda_pred_r = F.interpolate(input=x_isda_r, size=(h, w), mode='bilinear', align_corners=True)
                loss_isda_labeled += criterion(x_isda_pred_r, max_l)
                _, pred_unsup_l = model(unsup_imgs, step=1)
                _, pred_unsup_r = model(unsup_imgs, step=2)
                #
                # # ******************************************************************************

            elif config.exp_num == 7:
                # print("NUM_STEPS:")
                # print(NUM_STEPS)
                ####**add isda loss unsup label***************************************************
                #
                # _, pred_sup_l = model(imgs, step=1, isda=True)
                # _, pred_unsup_l = model(unsup_imgs, step=1, isda=True)
                x_unsup_l, feature_x_unsup_l, pred_unsup_l = model(unsup_imgs, step=1, isda=True)
                x_unsup_r, feature_x_unsup_r, pred_unsup_r = model(unsup_imgs, step=2, isda=True)

                # ratio = LAMBDA_0 * global_iteration / NUM_STEPS

                _, max_l = torch.max(pred_unsup_l, dim=1)
                _, max_r = torch.max(pred_unsup_r, dim=1)
                max_l = max_l.long()
                max_r = max_r.long()
                # print(max_r[0].tolist())
                # print(max_r.shape)
                # print(max_l.shape)
                # input()
                # pdb.set_trace()
                x_isda_unsup_l = isda_augmentor_1(feature_x_unsup_l, model.module.branch1.final_conv_1, x_unsup_l,
                                                  max_r, ratio)
                x_isda_unsup_r = isda_augmentor_2(feature_x_unsup_r, model.module.branch2.final_conv_1, x_unsup_r,
                                                  max_l, ratio)

                # x_isda_r = isda_augmentor_1(feature_x_unsup_r, model.module.branch2.final_conv_1, x_unsup_r, max_r, ratio)
                # x_dsn_isda_r = isda_augmentor_2(feature_x_dsn_unsup_r, model.module.branch2.final_conv_2, x_dsn_unsup_r, max_r, ratio)
                # print(x_isda.shape)
                # print(gts.shape)
                _, h, w = max_l.shape
                x_isda_pred_unsup_l = F.interpolate(input=x_isda_unsup_l, size=(h, w), mode='bilinear',
                                                    align_corners=True)
                loss_isda_unlabeled = criterion(x_isda_pred_unsup_l, max_r)

                x_isda_pred_unsup_r = F.interpolate(input=x_isda_unsup_r, size=(h, w), mode='bilinear',
                                                    align_corners=True)
                loss_isda_unlabeled += criterion(x_isda_pred_unsup_r, max_l)
                # loss_isda_r = criterion_isda([x_isda_r, x_dsn_isda_r], max_r)
                # print(loss_isda_l)
                # print(loss_isda_r)
                # input()
                #
                # reduce_loss = engine.all_reduce_tensor(loss)
                # print(reduce_loss)
                _, pred_sup_l = model(imgs, step=1)
                _, pred_sup_r = model(imgs, step=2)

                ####*****************************************************

            elif config.exp_num == 8:

                ####**add isda loss unsup label***************************************************
                #
                # _, pred_sup_l = model(imgs, step=1, isda=True)
                # _, pred_unsup_l = model(unsup_imgs, step=1, isda=True)
                x_unsup_l, feature_x_unsup_l, pred_unsup_l = model(unsup_imgs, step=1, isda=True)
                x_unsup_r, feature_x_unsup_r, pred_unsup_r = model(unsup_imgs, step=2, isda=True)

                # ratio = LAMBDA_0 * global_iteration / NUM_STEPS

                _, max_l = torch.max(pred_unsup_l, dim=1)
                _, max_r = torch.max(pred_unsup_r, dim=1)
                max_l = max_l.long()
                max_r = max_r.long()
                # print(max_r)
                # print(max_r.shape)
                # print(max_l.shape)
                # input()

                x_isda_unsup_l = isda_augmentor_1(feature_x_unsup_l, model.module.branch1.final_conv_1, x_unsup_l,
                                                  max_r, ratio)
                x_isda_unsup_r = isda_augmentor_2(feature_x_unsup_r, model.module.branch2.final_conv_1, x_unsup_r,
                                                  max_l, ratio)

                # x_isda_r = isda_augmentor_1(feature_x_unsup_r, model.module.branch2.final_conv_1, x_unsup_r, max_r, ratio)
                # x_dsn_isda_r = isda_augmentor_2(feature_x_dsn_unsup_r, model.module.branch2.final_conv_2, x_dsn_unsup_r, max_r, ratio)
                # print(x_isda.shape)
                # print(gts.shape)
                _, h, w = max_l.shape
                x_isda_pred_unsup_l = F.interpolate(input=x_isda_unsup_l, size=(h, w), mode='bilinear',
                                                    align_corners=True)
                loss_isda_unlabeled = criterion(x_isda_pred_unsup_l, max_l)

                x_isda_pred_unsup_r = F.interpolate(input=x_isda_unsup_r, size=(h, w), mode='bilinear',
                                                    align_corners=True)
                loss_isda_unlabeled += criterion(x_isda_pred_unsup_r, max_r)
                # loss_isda_r = criterion_isda([x_isda_r, x_dsn_isda_r], max_r)
                # print(loss_isda_l)
                # print(loss_isda_r)
                # input()
                #
                # reduce_loss = engine.all_reduce_tensor(loss)
                # print(reduce_loss)
                _, pred_sup_l = model(imgs, step=1)
                _, pred_sup_r = model(imgs, step=2)

                ####*****************************************************

            elif config.exp_num == 9:

                ####**add isda loss unsup label***************************************************
                #
                # _, pred_sup_l = model(imgs, step=1, isda=True)
                # _, pred_unsup_l = model(unsup_imgs, step=1, isda=True)
                x_unsup_l, feature_x_unsup_l, pred_unsup_l = model(unsup_imgs, step=1, isda=True)
                x_unsup_r, feature_x_unsup_r, pred_unsup_r = model(unsup_imgs, step=2, isda=True)
                # with torch.no_grad():
                #     pred_unsup_l.detach()
                #     pred_unsup_r.detach()
                # ratio = LAMBDA_0 * global_iteration / NUM_STEPS

                _, max_l = torch.max(pred_unsup_l, dim=1)
                _, max_r = torch.max(pred_unsup_r, dim=1)
                max_l = max_l.long()
                max_r = max_r.long()
                # print(max_r)
                # print(max_r.shape)
                # print(max_l.shape)
                # input()

                # x_isda_unsup_l = isda_augmentor_1(feature_x_unsup_l, model.module.branch1.final_conv_1, x_unsup_l,
                #                                   max_r, ratio)
                # x_isda_unsup_r = isda_augmentor_2(feature_x_unsup_r, model.module.branch2.final_conv_1, x_unsup_r,
                #                                   max_l, ratio)

                x_isda_unsup_l = isda_augmentor_1(feature_x_unsup_l, model.module.branch1.final_conv_1, x_unsup_l,
                                                  max_r, ratio)
                x_isda_unsup_r = isda_augmentor_2(feature_x_unsup_r, model.module.branch2.final_conv_1, x_unsup_r,
                                                  max_l, ratio)

                # x_isda_r = isda_augmentor_1(feature_x_unsup_r, model.module.branch2.final_conv_1, x_unsup_r, max_r, ratio)
                # x_dsn_isda_r = isda_augmentor_2(feature_x_dsn_unsup_r, model.module.branch2.final_conv_2, x_dsn_unsup_r, max_r, ratio)
                # print(x_isda.shape)
                # print(gts.shape)
                _, h, w = max_l.shape
                x_isda_pred_unsup_l = F.interpolate(input=x_isda_unsup_l, size=(h, w), mode='bilinear',
                                                    align_corners=True)
                # criterion_kl = nn.KLDivLoss(reduction="batchmean")
                criterion_kl = nn.KLDivLoss(reduction="sum")
                T = 2  # temperature
                s = torch.flatten(x_isda_pred_unsup_l, start_dim=2)
                t = torch.flatten(pred_unsup_l, start_dim=2)
                loss_isda_unlabeled = criterion_kl(F.log_softmax(s / T, dim=-1),
                                                   F.softmax(t / T, dim=-1)) / (s.numel() / s.shape[-1])
                # print(loss_isda_unlabeled)
                x_isda_pred_unsup_r = F.interpolate(input=x_isda_unsup_r, size=(h, w), mode='bilinear',
                                                    align_corners=True)
                s = torch.flatten(x_isda_pred_unsup_r, start_dim=2)
                t = torch.flatten(pred_unsup_r, start_dim=2)
                loss_isda_unlabeled += criterion_kl(F.log_softmax(s / T, dim=-1),
                                                    F.softmax(t / T, dim=-1)) / (s.numel() / s.shape[-1])
                beta = 3
                loss_isda_unlabeled = loss_isda_unlabeled * beta
                # print(loss_isda_unlabeled)
                # loss_isda_r = criterion_isda([x_isda_r, x_dsn_isda_r], max_r)
                # print(loss_isda_l)
                # print(loss_isda_r)
                # input()
                #
                # reduce_loss = engine.all_reduce_tensor(loss)
                # print(reduce_loss)
                _, pred_sup_l = model(imgs, step=1)
                _, pred_sup_r = model(imgs, step=2)

                ####*****************************************************

            elif config.exp_num == 10:

                ####**add isda loss unsup label***************************************************
                #
                # _, pred_sup_l = model(imgs, step=1, isda=True)
                # _, pred_unsup_l = model(unsup_imgs, step=1, isda=True)
                x_unsup_l, feature_x_unsup_l, pred_unsup_l = model(unsup_imgs, step=1, isda=True)
                x_unsup_r, feature_x_unsup_r, pred_unsup_r = model(unsup_imgs, step=2, isda=True)
                # with torch.no_grad():
                #     pred_unsup_l.detach()
                #     pred_unsup_r.detach()
                # ratio = LAMBDA_0 * global_iteration / NUM_STEPS

                _, max_l = torch.max(pred_unsup_l, dim=1)
                _, max_r = torch.max(pred_unsup_r, dim=1)
                max_l = max_l.long()
                max_r = max_r.long()
                # print(max_r)
                # print(max_r.shape)
                # print(max_l.shape)
                # input()

                # x_isda_unsup_l = isda_augmentor_1(feature_x_unsup_l, model.module.branch1.final_conv_1, x_unsup_l,
                #                                   max_r, ratio)
                # x_isda_unsup_r = isda_augmentor_2(feature_x_unsup_r, model.module.branch2.final_conv_1, x_unsup_r,
                #                                   max_l, ratio)

                x_isda_unsup_l = isda_augmentor_1(feature_x_unsup_l, model.module.branch1.final_conv_1, x_unsup_l,
                                                  max_l, ratio)
                x_isda_unsup_r = isda_augmentor_2(feature_x_unsup_r, model.module.branch2.final_conv_1, x_unsup_r,
                                                  max_r, ratio)

                # x_isda_r = isda_augmentor_1(feature_x_unsup_r, model.module.branch2.final_conv_1, x_unsup_r, max_r, ratio)
                # x_dsn_isda_r = isda_augmentor_2(feature_x_dsn_unsup_r, model.module.branch2.final_conv_2, x_dsn_unsup_r, max_r, ratio)
                # print(x_isda.shape)
                # print(gts.shape)ex
                _, h, w = max_l.shape
                x_isda_pred_unsup_l = F.interpolate(input=x_isda_unsup_l, size=(h, w), mode='bilinear',
                                                    align_corners=True)
                # criterion_kl = nn.KLDivLoss(reduction="batchmean")
                criterion_kl = nn.KLDivLoss(reduction="sum")
                T = 1  # temperature
                s = torch.flatten(x_isda_pred_unsup_l, start_dim=2)
                t = torch.flatten(pred_unsup_r, start_dim=2)
                loss_isda_unlabeled = criterion_kl(F.log_softmax(s / T, dim=-1),
                                                   F.softmax(t / T, dim=-1)) / (s.numel() / s.shape[-1])
                # print(loss_isda_unlabeled)
                x_isda_pred_unsup_r = F.interpolate(input=x_isda_unsup_r, size=(h, w), mode='bilinear',
                                                    align_corners=True)
                s = torch.flatten(x_isda_pred_unsup_r, start_dim=2)
                t = torch.flatten(pred_unsup_l, start_dim=2)
                loss_isda_unlabeled += criterion_kl(F.log_softmax(s / T, dim=-1),
                                                    F.softmax(t / T, dim=-1)) / (s.numel() / s.shape[-1])
                beta = 1
                loss_isda_unlabeled = loss_isda_unlabeled * beta
                # print(loss_isda_unlabeled)
                # loss_isda_r = criterion_isda([x_isda_r, x_dsn_isda_r], max_r)
                # print(loss_isda_l)
                # print(loss_isda_r)
                # input()
                #
                # reduce_loss = engine.all_reduce_tensor(loss)
                # print(reduce_loss)
                _, pred_sup_l = model(imgs, step=1)
                _, pred_sup_r = model(imgs, step=2)

                ####*****************************************************

            elif config.exp_num == 11:
                ####**add isda loss unsup label***************************************************
                #
                # _, pred_sup_l = model(imgs, step=1, isda=True)
                # _, pred_unsup_l = model(unsup_imgs, step=1, isda=True)
                x_unsup_l, feature_x_unsup_l, pred_unsup_l = model(unsup_imgs, step=1, isda=True)
                x_unsup_r, feature_x_unsup_r, pred_unsup_r = model(unsup_imgs, step=2, isda=True)

                # ratio = LAMBDA_0 * global_iteration / NUM_STEPS

                _, max_l = torch.max(pred_unsup_l, dim=1)
                _, max_r = torch.max(pred_unsup_r, dim=1)
                max_l = max_l.long()
                max_r = max_r.long()
                # print(max_r)
                # print(max_r.shape)
                # print(max_l.shape)
                # input()

                x_isda_unsup_l = isda_augmentor_1(feature_x_unsup_l, model.module.branch1.final_conv_1, x_unsup_l,
                                                  max_r, ratio)
                x_isda_unsup_r = isda_augmentor_2(feature_x_unsup_r, model.module.branch2.final_conv_1, x_unsup_r,
                                                  max_l, ratio)

                # x_isda_r = isda_augmentor_1(feature_x_unsup_r, model.module.branch2.final_conv_1, x_unsup_r, max_r, ratio)
                # x_dsn_isda_r = isda_augmentor_2(feature_x_dsn_unsup_r, model.module.branch2.final_conv_2, x_dsn_unsup_r, max_r, ratio)
                # print(x_isda.shape)
                # print(gts.shape)
                _, h, w = max_l.shape
                x_isda_pred_unsup_l = F.interpolate(input=x_isda_unsup_l, size=(h, w), mode='bilinear',
                                                    align_corners=True)
                loss_isda_unlabeled = criterion(x_isda_pred_unsup_l, max_r)

                x_isda_pred_unsup_r = F.interpolate(input=x_isda_unsup_r, size=(h, w), mode='bilinear',
                                                    align_corners=True)
                loss_isda_unlabeled += criterion(x_isda_pred_unsup_r, max_l)
                # loss_isda_r = criterion_isda([x_isda_r, x_dsn_isda_r], max_r)
                # print(loss_isda_l)
                # print(loss_isda_r)
                # input()
                #
                # reduce_loss = engine.all_reduce_tensor(loss)
                # print(reduce_loss)
                _, pred_sup_l = model(imgs, step=1)
                _, pred_sup_r = model(imgs, step=2)

            elif config.exp_num == 12:
                # print("NUM_STEPS:")
                # print(NUM_STEPS)
                ####**add isda loss unsup label***************************************************
                #
                # _, pred_sup_l = model(imgs, step=1, isda=True)
                # _, pred_unsup_l = model(unsup_imgs, step=1, isda=True)
                x_unsup_l, feature_x_unsup_l, pred_unsup_l = model(unsup_imgs, step=1, isda=True)
                x_unsup_r, feature_x_unsup_r, pred_unsup_r = model(unsup_imgs, step=2, isda=True)

                # ratio = LAMBDA_0 * global_iteration / NUM_STEPS

                _, max_l = torch.max(pred_unsup_l, dim=1)
                _, max_r = torch.max(pred_unsup_r, dim=1)
                max_l = max_l.long()
                max_r = max_r.long()
                # print(max_r[0].tolist())
                # print(max_r.shape)
                # print(max_l.shape)
                # input()
                # pdb.set_trace()
                x_isda_unsup_l = isda_augmentor_1(feature_x_unsup_l, model.module.branch1.final_conv_1, x_unsup_l,
                                                  max_r, ratio)
                x_isda_unsup_r = isda_augmentor_2(feature_x_unsup_r, model.module.branch2.final_conv_1, x_unsup_r,
                                                  max_l, ratio)

                # x_isda_r = isda_augmentor_1(feature_x_unsup_r, model.module.branch2.final_conv_1, x_unsup_r, max_r, ratio)
                # x_dsn_isda_r = isda_augmentor_2(feature_x_dsn_unsup_r, model.module.branch2.final_conv_2, x_dsn_unsup_r, max_r, ratio)
                # print(x_isda.shape)
                # print(gts.shape)
                _, h, w = max_l.shape
                x_isda_pred_unsup_l = F.interpolate(input=x_isda_unsup_l, size=(h, w), mode='bilinear',
                                                    align_corners=True)
                loss_isda_unlabeled = criterion(x_isda_pred_unsup_l, max_r)

                x_isda_pred_unsup_r = F.interpolate(input=x_isda_unsup_r, size=(h, w), mode='bilinear',
                                                    align_corners=True)
                loss_isda_unlabeled += criterion(x_isda_pred_unsup_r, max_l)
                # loss_isda_r = criterion_isda([x_isda_r, x_dsn_isda_r], max_r)
                # print(loss_isda_l)
                # print(loss_isda_r)
                # input()
                #
                # reduce_loss = engine.all_reduce_tensor(loss)
                # print(reduce_loss)
                _, pred_sup_l = model(imgs, step=1)
                _, pred_sup_r = model(imgs, step=2)

                ####*****************************************************

            else:
                _, pred_sup_l = model(imgs, step=1)
                _, pred_unsup_l = model(unsup_imgs, step=1)
                _, pred_sup_r = model(imgs, step=2)
                _, pred_unsup_r = model(unsup_imgs, step=2)

            if config.exp_num == 12:
                pass
                ### cps loss ###
                pred_l = torch.cat([pred_sup_l, pred_unsup_l], dim=0)
                pred_r = torch.cat([pred_sup_r, pred_unsup_r], dim=0)
                _, max_l = torch.max(pred_l, dim=1)
                _, max_r = torch.max(pred_r, dim=1)
                max_l = max_l.long()
                max_r = max_r.long()
                cps_loss = criterion(pred_l, max_r) + criterion(pred_r, max_l)
                cps_loss += loss_isda_unlabeled * config.scale
                # print("add loss_isda_unlabeled.")
                dist.all_reduce(cps_loss, dist.ReduceOp.SUM)
                cps_loss = cps_loss / engine.world_size
                cps_loss = cps_loss * config.cps_weight

                ### standard cross entropy loss ###
                loss_sup = criterion(pred_sup_l, gts)
                dist.all_reduce(loss_sup, dist.ReduceOp.SUM)
                loss_sup = loss_sup / engine.world_size

                loss_sup_r = criterion(pred_sup_r, gts)
                dist.all_reduce(loss_sup_r, dist.ReduceOp.SUM)
                loss_sup_r = loss_sup_r / engine.world_size

                unlabeled_loss = False

                current_idx = epoch * config.niters_per_epoch + idx
                lr = lr_policy.get_lr(current_idx)

                # reset the learning rate
                optimizer_l.param_groups[0]['lr'] = lr
                optimizer_l.param_groups[1]['lr'] = lr
                for i in range(2, len(optimizer_l.param_groups)):
                    optimizer_l.param_groups[i]['lr'] = lr
                optimizer_r.param_groups[0]['lr'] = lr
                optimizer_r.param_groups[1]['lr'] = lr
                for i in range(2, len(optimizer_r.param_groups)):
                    optimizer_r.param_groups[i]['lr'] = lr
            else:
                ### cps loss ###
                pred_l = torch.cat([pred_sup_l, pred_unsup_l], dim=0)
                pred_r = torch.cat([pred_sup_r, pred_unsup_r], dim=0)
                _, max_l = torch.max(pred_l, dim=1)
                _, max_r = torch.max(pred_r, dim=1)
                max_l = max_l.long()
                max_r = max_r.long()
                cps_loss = criterion(pred_l, max_r) + criterion(pred_r, max_l)
                dist.all_reduce(cps_loss, dist.ReduceOp.SUM)
                cps_loss = cps_loss / engine.world_size
                cps_loss = cps_loss * config.cps_weight

                ### standard cross entropy loss ###
                loss_sup = criterion(pred_sup_l, gts)
                dist.all_reduce(loss_sup, dist.ReduceOp.SUM)
                loss_sup = loss_sup / engine.world_size

                loss_sup_r = criterion(pred_sup_r, gts)
                dist.all_reduce(loss_sup_r, dist.ReduceOp.SUM)
                loss_sup_r = loss_sup_r / engine.world_size

                unlabeled_loss = False

                current_idx = epoch * config.niters_per_epoch + idx
                lr = lr_policy.get_lr(current_idx)

                # reset the learning rate
                optimizer_l.param_groups[0]['lr'] = lr
                optimizer_l.param_groups[1]['lr'] = lr
                for i in range(2, len(optimizer_l.param_groups)):
                    optimizer_l.param_groups[i]['lr'] = lr
                optimizer_r.param_groups[0]['lr'] = lr
                optimizer_r.param_groups[1]['lr'] = lr
                for i in range(2, len(optimizer_r.param_groups)):
                    optimizer_r.param_groups[i]['lr'] = lr

            if config.exp_num == -1:
                pass
            elif config.exp_num == 1:
                loss = loss_sup + loss_sup_r + cps_loss + config.scale * loss_isda + \
                       config.scale * loss_isda_labeled + config.scale * loss_isda_unlabeled
            elif config.exp_num == 2:
                loss = loss_sup + loss_sup_r + cps_loss + config.scale * loss_isda_labeled + config.scale * loss_isda_unlabeled
            elif config.exp_num == 3:
                loss = loss_sup + loss_sup_r + cps_loss + config.scale * loss_isda + config.scale * loss_isda_unlabeled
            elif config.exp_num == 4:
                loss = loss_sup + loss_sup_r + cps_loss + config.scale * loss_isda + config.scale * loss_isda_labeled
            elif config.exp_num == 5:
                loss = loss_sup + loss_sup_r + cps_loss + config.scale * loss_isda
            elif config.exp_num == 6:
                loss = loss_sup + loss_sup_r + cps_loss + config.scale * loss_isda_labeled
            elif config.exp_num == 7:
                loss = loss_sup + loss_sup_r + cps_loss + config.scale * loss_isda_unlabeled
            elif config.exp_num == 8:
                loss = loss_sup + loss_sup_r + cps_loss + config.scale * loss_isda_unlabeled
            elif config.exp_num == 9:
                loss = loss_sup + loss_sup_r + cps_loss + config.scale * loss_isda_unlabeled
            elif config.exp_num == 10:
                loss = loss_sup + loss_sup_r + cps_loss + config.scale * loss_isda_unlabeled
            elif config.exp_num == 11:
                if (epoch > (epoch // 4)):
                    loss = loss_sup + loss_sup_r + cps_loss + config.scale * loss_isda_unlabeled
                else:
                    loss = loss_sup + loss_sup_r + cps_loss + 1e-10 * loss_isda_unlabeled
            else:
                loss = loss_sup + loss_sup_r + cps_loss

            loss.backward()
            optimizer_l.step()
            optimizer_r.step()

            if config.exp_num == -1:
                pass
            elif config.exp_num == 1:
                print_str = 'E1poch{}/{}'.format(epoch, config.nepochs) \
                            + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                            + ' lr=%.2e' % lr \
                            + ' loss_sup=%.2f' % loss_sup.item() \
                            + ' loss_sup_r=%.2f' % loss_sup_r.item() \
                            + ' loss_cps=%.4f' % cps_loss.item() \
                            + ' loss_isda=%.4f' % loss_isda.item() \
                            + ' loss_isda_label=%.4f' % loss_isda_labeled.item() \
                            + ' loss_isda_unlabel=%.4f' % loss_isda_unlabeled.item()
            elif config.exp_num == 2:
                print_str = 'E2poch{}/{}'.format(epoch, config.nepochs) \
                            + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                            + ' lr=%.2e' % lr \
                            + ' loss_sup=%.2f' % loss_sup.item() \
                            + ' loss_sup_r=%.2f' % loss_sup_r.item() \
                            + ' loss_cps=%.4f' % cps_loss.item() \
                            + ' loss_isda_label=%.4f' % loss_isda_labeled.item() \
                            + ' loss_isda_unlabel=%.4f' % loss_isda_unlabeled.item()
            elif config.exp_num == 3:
                print_str = 'E3poch{}/{}'.format(epoch, config.nepochs) \
                            + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                            + ' lr=%.2e' % lr \
                            + ' loss_sup=%.2f' % loss_sup.item() \
                            + ' loss_sup_r=%.2f' % loss_sup_r.item() \
                            + ' loss_cps=%.4f' % cps_loss.item() \
                            + ' loss_isda=%.4f' % loss_isda.item() \
                            + ' loss_isda_unlabel=%.4f' % loss_isda_unlabeled.item()
            elif config.exp_num == 4:
                print_str = 'E4poch{}/{}'.format(epoch, config.nepochs) \
                            + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                            + ' lr=%.2e' % lr \
                            + ' loss_sup=%.2f' % loss_sup.item() \
                            + ' loss_sup_r=%.2f' % loss_sup_r.item() \
                            + ' loss_cps=%.4f' % cps_loss.item() \
                            + ' loss_isda=%.4f' % loss_isda.item() \
                            + ' loss_isda_label=%.4f' % loss_isda_labeled.item()
            elif config.exp_num == 5:
                print_str = 'E5poch{}/{}'.format(epoch, config.nepochs) \
                            + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                            + ' lr=%.2e' % lr \
                            + ' loss_sup=%.2f' % loss_sup.item() \
                            + ' loss_sup_r=%.2f' % loss_sup_r.item() \
                            + ' loss_cps=%.4f' % cps_loss.item() \
                            + ' loss_isda=%.4f' % loss_isda.item()

            elif config.exp_num == 6:
                print_str = 'E6poch{}/{}'.format(epoch, config.nepochs) \
                            + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                            + ' lr=%.2e' % lr \
                            + ' loss_sup=%.2f' % loss_sup.item() \
                            + ' loss_sup_r=%.2f' % loss_sup_r.item() \
                            + ' loss_cps=%.4f' % cps_loss.item()

            elif config.exp_num == 7:
                print_str = 'E7poch{}/{}'.format(epoch, config.nepochs) \
                            + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                            + ' lr=%.2e' % lr \
                            + ' loss_sup=%.2f' % loss_sup.item() \
                            + ' loss_sup_r=%.2f' % loss_sup_r.item() \
                            + ' loss_cps=%.4f' % cps_loss.item() \
                            + ' loss_isda_unlabel=%.4f' % loss_isda_unlabeled.item()

            elif config.exp_num == 8:
                print_str = 'E8poch{}/{}'.format(epoch, config.nepochs) \
                            + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                            + ' lr=%.2e' % lr \
                            + ' loss_sup=%.2f' % loss_sup.item() \
                            + ' loss_sup_r=%.2f' % loss_sup_r.item() \
                            + ' loss_cps=%.4f' % cps_loss.item() \
                            + ' loss_isda_unlabel=%.4f' % loss_isda_unlabeled.item()

            elif config.exp_num == 9:
                print_str = 'E9poch{}/{}'.format(epoch, config.nepochs) \
                            + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                            + ' lr=%.2e' % lr \
                            + ' loss_sup=%.2f' % loss_sup.item() \
                            + ' loss_sup_r=%.2f' % loss_sup_r.item() \
                            + ' loss_cps=%.4f' % cps_loss.item() \
                            + ' loss_isda_unlabel=%.4f' % loss_isda_unlabeled.item()

            elif config.exp_num == 10:
                print_str = 'E10poch{}/{}'.format(epoch, config.nepochs) \
                            + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                            + ' lr=%.2e' % lr \
                            + ' loss_sup=%.2f' % loss_sup.item() \
                            + ' loss_sup_r=%.2f' % loss_sup_r.item() \
                            + ' loss_cps=%.4f' % cps_loss.item() \
                            + ' loss_isda_unlabel=%.4f' % loss_isda_unlabeled.item()

            elif config.exp_num == 11:
                print_str = 'E11poch{}/{}'.format(epoch, config.nepochs) \
                            + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                            + ' lr=%.2e' % lr \
                            + ' loss_sup=%.2f' % loss_sup.item() \
                            + ' loss_sup_r=%.2f' % loss_sup_r.item() \
                            + ' loss_cps=%.4f' % cps_loss.item() \
                            + ' loss_isda_unlabel=%.4f' % loss_isda_unlabeled.item()

            else:
                print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                            + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                            + ' lr=%.2e' % lr \
                            + ' loss_sup=%.2f' % loss_sup.item() \
                            + ' loss_sup_r=%.2f' % loss_sup_r.item() \
                            + ' loss_cps=%.4f' % cps_loss.item()

            sum_loss_sup += loss_sup.item()
            sum_loss_sup_r += loss_sup_r.item()
            sum_cps += cps_loss.item()
            pbar.set_description(print_str, refresh=False)

            end_time = time.time()

        if engine.distributed and (engine.local_rank == 0):
            logger.add_scalar('train_loss_sup', sum_loss_sup / len(pbar), epoch)
            logger.add_scalar('train_loss_sup_r', sum_loss_sup_r / len(pbar), epoch)
            logger.add_scalar('train_loss_cps', sum_cps / len(pbar), epoch)

        if azure and engine.local_rank == 0:
            run.log(name='Supervised Training Loss', value=sum_loss_sup / len(pbar))
            run.log(name='Supervised Training Loss right', value=sum_loss_sup_r / len(pbar))
            run.log(name='Supervised Training Loss CPS', value=sum_cps / len(pbar))

        # if (epoch > config.nepochs // 2) and (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
        if (epoch > -1) and (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
