import os
import torch
import torchvision
import argparse
import torch_geometric.data
import math
import torch.nn as nn
from modules import *
import logging
logging.getLogger('PIL').setLevel(logging.WARNING)

from dataset import ImageAugDataset

import matplotlib.pyplot as plt
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import sys
sys.path.append('../code')
import config as config

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=os.path.join(config.BASE_DATA_DIR, config.REGION, 'MIX_IMG')) 
parser.add_argument('--bands', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=24)   #128
parser.add_argument('--projection_dim', type=int, default=128)
parser.add_argument('--resnet', type=str, default='resnet50')
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--world_size', type=int, default=1)
parser.add_argument('--total_epoch', type=int, default=120)
parser.add_argument('--output_path', type=str, default=os.path.join(config.BASE_DATA_DIR, config.REGION, f'img_encoder_mix_{config.YEAR}.pth'))
parser.add_argument('--schedule', default=[90, 110], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cos', type=bool, default=False)
parser.add_argument('--log', type=str, default=os.path.join(config.BASE_DATA_DIR, config.REGION, f'img_encoder_mix_{config.YEAR}.log'))
parser.add_argument('--loss', type=str, default=os.path.join(config.BASE_DATA_DIR, config.REGION, f'loss_mix_{config.YEAR}.jpg'))

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.total_epoch))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            if epoch >= milestone:
                lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler(args.log, mode='a'), logging.StreamHandler()])
    logger = logging.getLogger('ImageEncoder')
    logger.setLevel(logging.DEBUG)

    train_dataset = ImageAugDataset(path=args.data_path)
    
    train_loader = torch_geometric.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                                   num_workers=6)

    encoder = torchvision.models.vit_l_16(pretrained=False, num_classes=args.projection_dim)
    dim_mlp = encoder.heads[-1].weight.shape[1]
    img_encoder = ImageEncoder(encoder, args.projection_dim, dim_mlp).to(device)

    img_encoder.train()

    img_optimizer = torch.optim.Adam(img_encoder.parameters(), lr=args.lr)

    criterion = NT_Xent(args.batch_size, args.temperature, args.world_size)

    prefix = args.log.strip('log/').strip('.log')
    epoch_losses = []

    logger.info("-----------------------------------------")
    logger.info(f"Starting training with parameters:")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Bands: {args.bands}")
    logger.info(f"Projection dim: {args.projection_dim}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Total epochs: {args.total_epoch}")
    logger.info("-----------------------------------------")

    accumulation_steps = 5

    for epoch in range(1, args.total_epoch + 1):
        img_optimizer = adjust_learning_rate(img_optimizer, epoch, args)
        epoch_loss = 0
        img_optimizer.zero_grad()

        for step, data in enumerate(train_loader):

            img1, img2 = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)

            h_img1, h_img2, z_img1, z_img2 = img_encoder(img1, img2)
            
            loss = criterion(z_img1, z_img2)
            loss_accumulated = loss / accumulation_steps

            loss_accumulated.backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                
                torch.nn.utils.clip_grad_norm_(img_encoder.parameters(), max_norm=1.0)
                
                img_optimizer.step()
                img_optimizer.zero_grad()

            if not torch.isnan(loss):
                epoch_loss += loss.item()

            if step % 10 == 0:
                logger.debug(f'Epoch: {epoch:03d} | Step: {step:03d} | Loss: {loss.item():.6f}')

        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_losses.append(avg_epoch_loss)
        logger.info(f'Epoch: {epoch:03d} | Average Loss: {avg_epoch_loss:.6f}')

    torch.save(img_encoder.state_dict(), args.output_path)
    logger.info(f"Model saved to {args.output_path}")
    
    plt.figure()                   
    plt.plot(epoch_losses,'b',label = 'loss', linewidth=0.5)       
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xticks(ticks=np.arange(0, len(train_loader)*args.total_epoch, step=len(train_loader)*20), labels=np.arange(0, args.total_epoch, step=20))
    plt.legend()        
    plt.savefig(args.loss)
    logger.info(f"Loss plot saved to {args.loss}")
