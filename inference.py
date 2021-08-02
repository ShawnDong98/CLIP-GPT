import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image

from tqdm import tqdm

from utils import CFG
from mydataset import ImageCaptionDataset, MyTokenizer
from mymodel import CLIPModel




def get_image_embeddings(
        model,
        loader, 
        device
    ):
    valid_image_embeddings = []
    tqdm_object = tqdm(loader, total=len(loader))
    for batch in tqdm_object:
        batch = {
            key : value.to(device)
            for key, value in batch.items() if key != "filename"
        }

        with torch.no_grad():
            features = model.ImageEncoder(batch['image'])
            image_embeddings = model.ImagePrpjectionHead(features)
            valid_image_embeddings.append(image_embeddings)

    valid_image_embeddings = torch.cat(valid_image_embeddings)
    print(valid_image_embeddings.shape)
    return valid_image_embeddings





def find_matches(
        model, 
        image_embeddings,
        query,
        filenames,
        tokenizer,
        device,
        n=9
    ):
    encode_query = torch.tensor(
        tokenizer(
            query,
            max_length = 100
        ), 
    ).unsqueeze(0).to(device)

    

    with torch.no_grad():
        text_features = model.TextEncoder(
            text = encode_query,
        )
        text_embeddings = model.TextProjectionHead(text_features)

    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T

    value, indices = torch.topk(dot_similarity.squeeze(0), n * 5)

    mathes = [filenames[idx] for idx in indices[::5]]

    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for match, ax in zip(mathes, axes.flatten()):
        image = Image.open(f"{CFG.val_root_dir}/{match}")
        ax.imshow(image)
        ax.axis("off")

    plt.show()
    plt.savefig("test.png")


if __name__ == "__main__":
    tokenizer = MyTokenizer()
    tokenizer.load_vocab("vocab.json")

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

    model.load_state_dict(torch.load("./best.pth.tar")['model_dict'])

    config = resolve_data_config({}, model=model.ImageEncoder)
    transform = create_transform(**config)


    val_df = pd.read_csv(CFG.val_filepath)

    valset = ImageCaptionDataset(
        root_dir = CFG.val_root_dir,
        filenames = val_df["image"].tolist()[:5000] if CFG.Debug else val_df["image"].tolist(),
        captions = val_df["caption"].tolist()[:5000] if CFG.Debug else val_df["caption"].tolist(),
        tokenizer = tokenizer,
        max_length = CFG.block_size,
        trans = transform,
    )

    device = "cpu"
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        model = nn.DataParallel(model).to(device)

    

    val_loader  = DataLoader(
            valset,
            batch_size = CFG.batch_size,
            shuffle = False,
            pin_memory = True,
            drop_last = False,
            num_workers = CFG.num_workers, 
        )

    model.eval()

    valid_image_embeddings = get_image_embeddings(
        model.module, 
        val_loader, 
        device=device,
    )

    # find_matches(
    #     model, 
    #     valid_image_embeddings,
    #     tokenizer = tokenizer,
    #     query="一只狗在草坪上。",
    #     filenames=val_df['image_name'].tolist(),
    #     device=CFG.device,
    #     n=9
    # )

    find_matches(
        model.module, 
        valid_image_embeddings,
        tokenizer = tokenizer,
        query="两个人在说话。",
        filenames=val_df['image'].tolist(),
        device=device,
        n=9
    )