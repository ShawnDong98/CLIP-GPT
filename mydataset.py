import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import os
from PIL import Image
import pandas as pd

from utils import CFG



class MyTokenizer:
    def __init__(self):
        pass


    def __call__(
        self, 
        sentences,
        max_length,
        padding = True,
        truncation = True,
        return_tokens = False
    ):
        list_sentences = list(sentences)
        list_ids = [self.stoi.get(token, self.stoi['[UNK]']) for token in list_sentences]

        

        if truncation:
            if len(list_ids) > max_length - 2:
                list_ids = list_ids[:max_length-2]

        

        list_ids = self.PostProcess(list_ids)

        

        if padding:
            if len(list_ids) < max_length:
                list_ids.extend([self.stoi['[PAD]']] * (max_length - len(list_ids)))

        if return_tokens:
            list_tokens = self.decode(list_ids)
            return list_ids, list_tokens

        return list_ids

    def PostProcess(self, list_ids):
        post_list_ids = [self.stoi['[CLS]']]
        post_list_ids.extend(list_ids)
        post_list_ids.append(self.stoi['[SEP]'])

        return post_list_ids            


    


    def decode(self, ids):
        return [self.itos[id_] for id_ in ids]


    def load_vocab(self, vocab_path):
        import json
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.stoi = json.load(f)

        self.itos = { i: ch for ch, i in self.stoi.items()}

    def save_vocab(self, vocab_path):
        import json
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.stoi, f, ensure_ascii=False)


class ImageCaptionDataset(Dataset):
    def __init__(
        self, 
        root_dir, 
        filenames,
        captions,
        tokenizer,
        max_length,
        trans = None
    ):
        self.root_dir = root_dir
        self.filenames = filenames
        self.captions = captions
        self.max_length = max_length
        self.trans = trans
        
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        batch = {
            "input_id": torch.tensor(
                self.tokenizer(
                    self.captions[idx],
                    max_length=self.max_length,
                )
            ),
            "filename": self.filenames[idx],
        }

        image = Image.open(os.path.join(self.root_dir, self.filenames[idx]))

        if self.trans:
            image = self.trans(image)

        batch['image'] = image

        return batch

    def __len__(self):
        return len(self.captions)





if __name__ == '__main__':
    tokenizer = MyTokenizer()
    tokenizer.load_vocab("vocab.json")

    text = "吴亦凡你好， 我是李雪琴！"
    ids, tokens = tokenizer(
        text, 
        max_length = CFG.max_length,
        return_tokens = True
    )

    print(ids)
    print(tokens)

    train_df = pd.read_csv(CFG.train_filepath)


    dataset = ImageCaptionDataset(
        root_dir = CFG.train_root_dir,
        filenames = train_df['image'].tolist(),
        captions = train_df['caption'].tolist(),
        tokenizer = tokenizer,
        max_length = CFG.max_length
    )

    print(dataset[0])
