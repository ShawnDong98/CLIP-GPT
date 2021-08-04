import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


import pandas as pd
import itertools
from tqdm import tqdm

from utils import CFG, AvgMeter, load_checkpoint, save_checkpoint, set_seed
from mydataset import ImageCaptionDataset, MyTokenizer

from mymodel import CLIPModel

set_seed(42)

def train_epoch(
    model,
    train_loader,
    optimizer,
    epoch,
    writer, 
    device
):
    train_loss_meter = AvgMeter(name="train_loss_meter")
    train_step = 0
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {
            key: value.to(device)
            for key, value in batch.items() if key != "filename"
        }
        loss = model(batch['image'], batch['input_id'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("Training loss", loss.item(), global_step=(epoch * len(train_loader) + train_step))
        count = batch['image'].shape[0]
        train_loss_meter.update(loss.item(), count)
        train_step += 1

        tqdm_object.set_postfix(train_loss=train_loss_meter.avg)

    return train_loss_meter




def val_epoch(
    model,
    val_loader,
    epoch,
    writer,
    device
):
    val_loss_meter = AvgMeter(name="val_loss_meter")
    val_step = 0
    tqdm_object = tqdm(val_loader, total=len(val_loader))
    for batch in tqdm_object:
        batch = {
            key: value.to(device)
            for key, value in batch.items() if key != "filename"
        }

        loss = model(batch['image'], batch['input_id'])

        count = batch['image'].shape[0]
        val_loss_meter.update(loss.item(), count)
        writer.add_scalar("Validing loss ", loss.item(), global_step=(epoch * len(val_loader) + val_step))
        val_step += 1

        tqdm_object.set_postfix(val_loss=val_loss_meter.avg)
    
    return val_loss_meter




def main():
    tokenizer = MyTokenizer()
    tokenizer.load_vocab("vocab.json")

    # tensorboard
    writer = SummaryWriter("./run/flickr8k")

    model = CLIPModel(
        temperature = CFG.temperature,
        image_model = CFG.image_model, 
        pretrained = CFG.pretrained,
        trainable = CFG.trainable,
        vocab_size = tokenizer.get_vocab_size(),
        block_size = CFG.block_size,
        embed_dim = CFG.embed_dim,
        n_head = CFG.n_head,
        num_layers = CFG.num_layers,
        attn_pdrop = CFG.attn_pdrop,
        resid_pdrop = CFG.resid_pdrop,
        embd_pdrop = CFG.embd_pdrop,
        image_features_dim = CFG.image_features_dim,
        text_features_dim = CFG.text_features_dim,
        proj_dim = CFG.proj_dim,
    )
    config = resolve_data_config({}, model=model.ImageEncoder)
    transform = create_transform(**config)

    train_df = pd.read_csv(CFG.train_filepath)
    val_df = pd.read_csv(CFG.val_filepath)

    trainset = ImageCaptionDataset(
        root_dir = CFG.train_root_dir,
        filenames = train_df["image"].tolist()[:100] if CFG.Debug else train_df["image"].tolist(),
        captions = train_df["caption"].tolist()[:100] if CFG.Debug else train_df["caption"].tolist(),
        tokenizer = tokenizer,
        max_length = CFG.block_size,
        trans = transform,
    )

    valset = ImageCaptionDataset(
        root_dir = CFG.val_root_dir,
        filenames = val_df["image"].tolist()[:100] if CFG.Debug else val_df["image"].tolist(),
        captions = val_df["caption"].tolist()[:100] if CFG.Debug else val_df["caption"].tolist(),
        tokenizer = tokenizer,
        max_length = CFG.block_size,
        trans = transform,
    )

    optimizer = optim.AdamW(model.parameters(), lr=CFG.learning_rate)

    device = "cpu"
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        model = nn.DataParallel(model).to(device)


    best_loss = float("inf")
    for epoch in range(CFG.num_epochs):
        print(f"Epoch: {epoch + 1}")
        train_loader  = DataLoader(
            trainset,
            batch_size = CFG.batch_size,
            shuffle = True,
            pin_memory = True,
            drop_last = True,
            num_workers = CFG.num_workers, 
        )

        val_loader  = DataLoader(
            valset,
            batch_size = CFG.batch_size,
            shuffle = False,
            pin_memory = True,
            drop_last = False,
            num_workers = CFG.num_workers, 
        )

        train_loss = train_epoch(
            model = model,
            train_loader = train_loader,
            optimizer = optimizer,
            epoch = epoch,
            writer = writer,
            device = device,
        )

        with torch.no_grad():
            val_loss = val_epoch(
                model = model,
                val_loader = val_loader,
                epoch = epoch,
                writer = writer,
                device = device,
            )

        if val_loss.avg < best_loss:
            best_loss = val_loss.avg
            checkpoint = {
                "model_dict": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, CFG.save_checkpoint_path)


if __name__ == '__main__':
    main()