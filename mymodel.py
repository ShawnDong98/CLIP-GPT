import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class CasualSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_head: int,
        block_size: int,
        attn_pdrop: float,
        resid_pdrop: float
    ): 
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

        queries = self.queries(x).view(B, T, C // self.n_head, self.head_dim).transpose(1, 2)
        keys = self.keys(x).view(B, T, C // self.n_head, self.head_dim).transpose(1, 2)
        values = self.values(x).view(B, T, C // self.n_head, self.head_dim).transpose(1, 2)

        energy = self.queries @ self.keys.transpose(-2, -1)
        energy = energy.masked_fill(self.mask[:, :, :T, :T], -float("inf"))
        attn = torch.softmax(energy / (self.embed_dim ** (1/2)), dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ values
        out = self.resid_drop(self.fc_out(out))
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return out


class Block(nn.Module):
    def __init__(
        self,
        embed_dim, 
        n_head,
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
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.num_layers = num_layers

        self.word_embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.block_size, self.embed_dim))
        self.embd_drop = nn.Dropout(embd_pdrop)

        self.blocks = nn.Sequential(
            *[
                Block(
                    embed_dim = embed_dim,
                    n_head = n_head,
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
        x = self.block(x)
        logits = self.head(self.ln_f(x))

        loss = None

        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), target.view(-1)).mean()

        return logits, loss



class ImageEncoder(nn.Module):
    def __init__()