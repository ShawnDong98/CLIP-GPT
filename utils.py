import torch
import torch.nn as nn
import torch.nn.functional as F


class CFG:
    Debug = False

    # Model
    image_model = "resnet50"

    
    embed_dim = 768
    n_head = 12
    block_size = 100,

    attn_pdrop = 0.1
    resid_pdrop = 0.1
    embd_pdrop = 0.1



    # Datasets
    train_root_dir = "../datasets/MM/2017_AI_Challenger/train/caption_train_images_20170902"
    train_filepath = "../datasets/MM/2017_AI_Challenger/train/train.csv"

    val_root_dir = "../datasets/MM/2017_AI_Challenger/val/ caption_validation_images_20170910"
    val_filepath = "../datasets/MM/2017_AI_Challenger/val/val.csv"

    max_length = 100