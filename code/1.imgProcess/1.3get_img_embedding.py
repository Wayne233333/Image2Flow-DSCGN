import os
import numpy as np
import torch
import torchvision
import argparse
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.multiprocessing
from tqdm import tqdm
import pickle
import pandas as pd
from collections import OrderedDict
import logging
logging.getLogger('PIL').setLevel(logging.WARNING)

import sys
sys.path.append('..')
from modules import ImageEncoder
from dataset import ImageAugDataset
import config as config

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=os.path.join(config.DATA_DIR, config.REGION, config.YEAR))
parser.add_argument('--bands', type=int, default=0)
parser.add_argument('--projection_dim', type=int, default=128)
parser.add_argument('--model_path', type=str, default='./ckpt')
parser.add_argument('--log', type=str, default=f'./log/get_embedding_{config.REGION}_{config.YEAR}.log')
parser.add_argument('--output_path', type=str, default=os.path.join(config.DATA_DIR, "Vis", f"train_on_{config.REGION}_{config.YEAR}.csv"))
parser.add_argument('--ckpt', type=str, default=f'img_encoder_mix_{config.YEAR}.pth')

if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda")

    # Setup logging
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler(args.log, mode='a'), logging.StreamHandler()])
    logger = logging.getLogger('ImageEmbedding')
    logger.setLevel(logging.DEBUG)

    logger.info("-----------------------------------------")
    logger.info(f"Starting image embedding extraction with parameters:")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Bands: {args.bands}")
    logger.info(f"Projection dim: {args.projection_dim}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Checkpoint: {args.ckpt}")
    logger.info("-----------------------------------------")

    ckpts = [args.ckpt]
    for ckpt in ckpts:
        model_fp = os.path.join(args.model_path, ckpt)
        logger.info(f"Loading model from {model_fp}")
        
        # image encoder
        encoder = torchvision.models.vit_l_16(pretrained=False, num_classes=args.projection_dim)
        dim_mlp = encoder.heads[-1].weight.shape[1]
        img_encoder = ImageEncoder(encoder, args.projection_dim, dim_mlp).to(device)
 
        state_dict = torch.load(model_fp, map_location=device)
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            if k.startswith('.'):
                if "projector" in k:
                    name = k[1:]
                else:
                    name = "encoder" + k
            elif k.startswith('module.'):
                name = k[7:]
            else:
                name = k

            new_state_dict[name] = v

        msg = img_encoder.load_state_dict(new_state_dict, strict=True)
        print(f"Successfully loaded checkpoint from {model_fp}: {msg}")

        img_encoder = img_encoder.to(device)

        for name, param in img_encoder.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        
        # img_encoder = nn.DataParallel(img_encoder)
        img_encoder.eval()
        dict = {}
                    
        logger.info("Starting image embedding extraction...")
        for img_file in tqdm(os.listdir(args.data_path)):
            if args.bands == 6:
                img_input = np.load(os.path.join(args.data_path,img_file))
                img_input = np.transpose(img_input, (1,2,0))
            else:
                img_input = Image.open(os.path.join(args.data_path,img_file))

            ID = img_file.split("_")[-1].replace('.tif', '')

            resize = transforms.Resize((224, 224))
            toTensor = transforms.ToTensor()
            normalize = transforms.Normalize(
                                            mean = [0.6247, 0.4714, 0.3403],
                                            std = [0.1945, 0.1398, 0.1141]
                                        )
            img_input = resize(img_input)
            img_input = toTensor(img_input)
            img_input = normalize(img_input)
            img_input = img_input.unsqueeze(0).to(device)
            output = img_encoder.encoder(img_input)
            feat = list(output.cpu().detach().numpy().flatten().astype(float))
            dict[ID] = feat
            
        node_feats = pd.DataFrame(dict).T
        node_feats.index.name = 'geocode'
        node_feats.to_csv(args.output_path)
        logger.info(f"Embeddings saved to {args.output_path}")
