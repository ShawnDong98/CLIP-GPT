from typing import Text
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from mydataset import MyTokenizer
from utils import CFG


class CasualSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_head: int,
        block_size: int,
        attn_pdrop: float,
        resid_pdrop: float
    ): 
        super(CasualSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.head_dim = self.embed_dim // self.n_head
        assert self.head_dim * self.n_head == self.embed_dim, "Embed_dim should be divisible by n_head!"
        self.block_size = block_size

        self.queries = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.keys = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.values = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.attn_drop = nn.Dropout(attn_pdrop)

        self.fc_out = nn.Linear(self.embed_dim, self.embed_dim)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.register_buffer("mask", torch.tril(torch.ones(self.block_size, self.block_size)).view(1, 1, self.block_size, self.block_size))

    def forward(self, x):
        B, T, C = x.shape

        queries = self.queries(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        keys = self.keys(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        values = self.values(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        energy = queries @ keys.transpose(-2, -1)
        energy = energy.masked_fill(self.mask[:, :, :T, :T]==0, -float("inf"))
        attn = torch.softmax(energy / (self.embed_dim ** (1/2)), dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ values
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.fc_out(out))

        return out


class Block(nn.Module):
    def __init__(
        self,
        embed_dim, 
        n_head,
        block_size,
        attn_pdrop, 
        resid_pdrop,
    ):
        super(Block, self).__init__()
        self.embed_dim = embed_dim
        self.n_head = n_head

        self.ln1 = nn.LayerNorm(self.embed_dim)

        self.attn = CasualSelfAttention(
            embed_dim = embed_dim,
            n_head = n_head,
            block_size = block_size,
            attn_pdrop = attn_pdrop,
            resid_pdrop = resid_pdrop
        )

        self.attn_drop = nn.Dropout(attn_pdrop)

        self.ln2 = nn.LayerNorm(self.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, 4 * self.embed_dim),
            nn.GELU(),
            nn.Linear(4 * self.embed_dim, self.embed_dim),
            nn.Dropout(resid_pdrop)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        embed_dim,
        n_head,
        num_layers,
        attn_pdrop,
        resid_pdrop,
        embd_pdrop,
    ):
        super(GPT, self).__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.num_layers = num_layers

        self.word_embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros((1, self.block_size, self.embed_dim)))
        self.embd_drop = nn.Dropout(embd_pdrop)

        self.blocks = nn.Sequential(
            *[
                Block(
                    embed_dim = embed_dim,
                    n_head = n_head,
                    block_size = block_size,
                    attn_pdrop = attn_pdrop,
                    resid_pdrop = resid_pdrop
                )
                for _ in range(self.num_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, self.vocab_size)
    
    def forward(self, x, target=None):
        B, T = x.shape
        word_embedding = self.word_embedding(x)
        pos_embedding = self.pos_embedding[:, :T, :]

        x = self.embd_drop(word_embedding + pos_embedding)
        last_hidden_state = self.blocks(x)
        logits = self.head(self.ln_f(last_hidden_state))

        loss = None

        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), target.view(-1)).mean()

        return logits, loss, last_hidden_state



class ImageEncoder(nn.Module):
    def __init__(
        self,
        image_model,
        pretrained,
        trainable
    ):
        super(ImageEncoder, self).__init__()
        self.model = timm.create_model(
            model_name = image_model,
            pretrained = pretrained,
            num_classes = 0,
            global_pool = "avg"
        )

        for p in self.model.parameters():
            p.requires_grad = trainable

        print(self.model)

    def forward(self, image):
        features = self.model(image)

        return features


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        embed_dim,
        n_head,
        num_layers,
        attn_pdrop,
        resid_pdrop,
        embd_pdrop
    ):
        super(TextEncoder, self).__init__()
        self.model = GPT(
            vocab_size = vocab_size,
            block_size = block_size,
            embed_dim = embed_dim,
            n_head = n_head,
            num_layers = num_layers,
            attn_pdrop = attn_pdrop,
            resid_pdrop = resid_pdrop,
            embd_pdrop = embd_pdrop
        )
    
    def forward(self, text):
        trg_token_idx = torch.nonzero(torch.where(text == 3, torch.tensor(3).to(text.device), torch.tensor(0).to(text.device)))[:, 1]
        logits, loss, last_hidden_state = self.model(text)

        out = torch.stack([last_hidden_state[i, idx, :] for i, idx in enumerate(trg_token_idx)])
        return out


class ProjectionHead(nn.Module):
    def __init__(
        self,
        features_dim,
        proj_dim,
        resid_pdrop
    ):
        super(ProjectionHead, self).__init__()
        self.features_dim = features_dim
        self.proj_dim = proj_dim

        self.projection = nn.Linear(self.features_dim, self.proj_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(self.proj_dim, self.proj_dim)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.ln = nn.LayerNorm(self.proj_dim)

    def forward(self, features):
        
        projected = self.projection(features)
        projected = self.gelu(projected)

        x = self.fc(projected)
        x = self.resid_drop(x)
        x = self.ln(x + projected)

        return x

        
def cross_entropy(preds, targets, reduction="none"):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature,
        image_model,
        pretrained,
        trainable,
        vocab_size,
        block_size,
        embed_dim,
        n_head,
        num_layers,
        attn_pdrop,
        resid_pdrop,
        embd_pdrop,
        image_features_dim,
        text_features_dim,
        proj_dim,
    ):
        super(CLIPModel, self).__init__()
        self.temperature = temperature
        self.ImageEncoder = ImageEncoder(
            image_model = image_model,
            pretrained = pretrained,
            trainable = trainable,
        )

        self.TextEncoder = TextEncoder(
            vocab_size = vocab_size,
            block_size = block_size,
            embed_dim = embed_dim,
            n_head = n_head,
            num_layers = num_layers,
            attn_pdrop = attn_pdrop,
            resid_pdrop = resid_pdrop,
            embd_pdrop = embd_pdrop,
        )

        self.ImagePrpjectionHead = ProjectionHead(
            features_dim = image_features_dim,
            proj_dim = proj_dim,
            resid_pdrop = resid_pdrop
        )

        self.TextProjectionHead = ProjectionHead(
            features_dim = text_features_dim,
            proj_dim = proj_dim,
            resid_pdrop = resid_pdrop
        )

    def forward(self, image, text):
        image_features = self.ImageEncoder(image)
        text_features = self.TextEncoder(text)

        image_embeddings = self.ImagePrpjectionHead(image_features)
        text_embeddings = self.TextProjectionHead(text_features)


        logits = (text_embeddings @ image_embeddings.T) / self.temperature

        image_similarity = image_embeddings @ image_embeddings.T
        text_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax((image_similarity + text_similarity) / 2 * self.temperature, dim = -1)

        text_loss = cross_entropy(logits, targets, reduction="none")
        image_loss = cross_entropy(logits.T, targets.T, reduction="none")

        loss = (text_loss + image_loss) / 2.0

        return loss.mean()





if __name__ == '__main__':
    # model = ImageEncoder(
    #     image_model = CFG.image_model, 
    #     pretrained = CFG.pretrained,
    #     trainable = CFG.trainable,
    # )

    # tokenizer = MyTokenizer()
    # tokenizer.load_vocab("vocab.json")


    # model = CLIPModel(
    #     temperature = CFG.temperature,
    #     image_model = CFG.image_model, 
    #     pretrained = CFG.pretrained,
    #     trainable = CFG.trainable,
    #     vocab_size = tokenizer.get_vocab_size(),
    #     block_size = CFG.block_size,
    #     embed_dim = CFG.embed_dim,
    #     n_head = CFG.n_head,
    #     num_layers = CFG.num_layers,
    #     attn_pdrop = CFG.attn_pdrop,
    #     resid_pdrop = CFG.resid_pdrop,
    #     embd_pdrop = CFG.embd_pdrop,
    #     image_features_dim = CFG.image_features_dim,
    #     text_features_dim = CFG.text_features_dim,
    #     proj_dim = CFG.proj_dim,
    # )

    t = torch.randn((3, 4, 5, 5))
    print(t)

    index = torch.tensor(
        [
            [0, 1],
            [1, 2],
            [2, 3]
        ]
    )

    # res = torch.stack([t[i, idx, :, :] for i, idx in enumerate(index)])
    # for r in res:
    #     print(r.shape)
    print(t[index].shape)