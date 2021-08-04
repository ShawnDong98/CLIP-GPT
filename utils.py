import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np


class CFG:
    Debug = True
    pretrained = True
    trainable = True

    # Train
    learning_rate = 3e-4
    batch_size = 32
    num_workers = 0

    num_epochs = 10

    # Model
    temperature = 1.0

    image_model = "resnet50"

    image_features_dim = 2048
    text_features_dim = 768
    proj_dim = 768

    embed_dim = 768
    n_head = 12
    block_size = 100
    num_layers = 12

    attn_pdrop = 0.1
    resid_pdrop = 0.1
    embd_pdrop = 0.1

    # Datasets
    train_root_dir = "../datasets/MM/2017_AI_Challenger/train/caption_train_images_20170902"
    train_filepath = "../datasets/MM/2017_AI_Challenger/train/train.csv"

    val_root_dir = "../datasets/MM/2017_AI_Challenger/val/caption_validation_images_20170910"
    val_filepath = "../datasets/MM/2017_AI_Challenger/val/val.csv"



    # save
    load_checkpoint_path = "best.pth.tar"
    save_checkpoint_path = "best.pth.tar"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.sum, self.avg, self.count = [0] * 3

    def update(self, value, count=1):
        self.count += count
        self.sum += value * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg}"
        return text

def save_checkpoint(checkpoint, save_checkpoint_path):
    print("=> Saving Checkpoint...")
    torch.save(checkpoint, save_checkpoint_path)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading Checkpoint...")
    model.load_state_dict(checkpoint['model_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
