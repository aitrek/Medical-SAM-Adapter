import argparse
import os
import sys
import random
import time
from collections import OrderedDict
from datetime import datetime
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from skimage import io
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
# from tensorboardX import SummaryWriter
#from dataset import *
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

import cfg
args = cfg.parse_args()

import function
from conf import settings
#from models.discriminatorlayer import discriminator
from dataset import *
from utils import *

torch.set_default_dtype(torch.float32)


def main():
    wandb.init(project="SAM_Nuclei",
               name=f"Medical-SAM-Adapter_{datetime.now().strftime('%m-%d_%H-%M')}")

    args.dataset = "cesan"
    args.data_path = "/root/autodl-tmp/datasets/SAM_nuclei_preprocessed/ALL_Multi"
    args.gpu = True
    # args.data_path = "/Users/zhaojq/Datasets/ALL_Multi"
    # args.gpu = False
    args.sam_ckpt = "sam_vit_b_01ec64.pth"
    args.val_freq = 1000
    args.w = 16
    args.b = 16
    args.excluded = ["MoNuSeg2020"]
    args.test_sample_rate = 0.3
    args.weights = "White Blood Cell_MicroScope_sam_1024.pth"

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.gpu:
        GPUdevice = torch.device('cuda', args.gpu_device)
    else:
        GPUdevice = torch.device('cpu')

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
    if args.pretrain:
        weights = torch.load(args.pretrain)
        net.load_state_dict(weights,strict=False)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay

    '''load pretrained model'''
    if args.weights != 0:
        print(f'=> resuming from {args.weights}')
        assert os.path.exists(args.weights)
        checkpoint_file = os.path.join(args.weights)
        assert os.path.exists(checkpoint_file)
        loc = 'cuda:{}'.format(args.gpu_device) if args.gpu else "cpu"
        checkpoint = torch.load(checkpoint_file, map_location=loc)
        start_epoch = checkpoint.get('epoch', 0)
        best_tol = checkpoint.get('best_tol', 100)

        net.load_state_dict(checkpoint['state_dict'], strict=False)
        # optimizer.load_state_dict(checkpoint['optimizer'], strict=False)

        args.path_helper = checkpoint.get('path_helper', {})
        log_dir = os.path.join(os.path.dirname(__file__), "log")
        os.makedirs(log_dir, exist_ok=True)
        logger = create_logger(log_dir)
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    nice_train_loader, nice_test_loader = get_dataloader(args)
    print(f"len(train_loader): {len(nice_train_loader)}, len(test_loader): {len(nice_test_loader)}")

    '''checkpoint path and tensorboard'''
    # iter_per_epoch = len(Glaucoma_training_loader)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    # writer = SummaryWriter(log_dir=os.path.join(
    #         settings.LOG_DIR, args.net, settings.TIME_NOW))
    # input_tensor = torch.Tensor(args.b, 3, 256, 256).cuda(device = GPUdevice)
    # writer.add_graph(net, Variable(input_tensor, requires_grad=True))
    writer = None

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    '''begain training'''
    best_acc = 0.0
    best_tol = 1e4
    best_dice = 0.0

    global_vals = {"step": 0, "best_dice": 100}

    for epoch in range(settings.EPOCH):

        # if epoch and epoch < 5:
        #     if args.dataset != 'REFUGE':
        #         tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
        #         logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
        #     else:
        #         tol, (eiou_cup, eiou_disc, edice_cup, edice_disc) = function.validation_sam(args, nice_test_loader, epoch, net, writer, global_vals)
        #         logger.info(f'Total score: {tol}, IOU_CUP: {eiou_cup}, IOU_DISC: {eiou_disc}, DICE_CUP: {edice_cup}, DICE_DISC: {edice_disc} || @ epoch {epoch}.')

        net.train()
        time_start = time.time()
        function.train_sam(args, net, optimizer, nice_train_loader, nice_test_loader, epoch, writer, vis = args.vis, global_vals=global_vals)
        time_end = time.time()
        print('time_for_training ', time_end - time_start)

        # net.eval()
        # if epoch and epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:
        #     if args.dataset != 'REFUGE':
        #         tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, nice_test_loader, epoch, net, writer)
        #         logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
        #     else:
        #         tol, (eiou_cup, eiou_disc, edice_cup, edice_disc) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
        #         logger.info(f'Total score: {tol}, IOU_CUP: {eiou_cup}, IOU_DISC: {eiou_disc}, DICE_CUP: {edice_cup}, DICE_DISC: {edice_disc} || @ epoch {epoch}.')
        #
        #     if args.distributed != 'none':
        #         sd = net.module.state_dict()
        #     else:
        #         sd = net.state_dict()
        #
        #     if edice > best_dice:
        #         best_tol = tol
        #         is_best = True
        #
        #         save_checkpoint({
        #         'epoch': epoch + 1,
        #         'model': args.net,
        #         'state_dict': sd,
        #         'optimizer': optimizer.state_dict(),
        #         'best_tol': best_dice,
        #         'path_helper': args.path_helper,
        #     }, is_best, args.path_helper['ckpt_path'], filename="best_dice_checkpoint.pth")
        #     else:
        #         is_best = False

    # writer.close()
    wandb.finish()


if __name__ == '__main__':
    main()
