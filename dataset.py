import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ChemDataset(Dataset):
    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, src_lang, tgt_format, seq_len):
        super().__init__()
        
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_format = tgt_format
        self.seq_len = seq_len
        
        self.sos = torch.tensor([tokenizer_src.encode("<s>", add_special_tokens=False)], dtype=torch.int64)
        self.eos = torch.tensor([tokenizer_src.encode("</s>", add_special_tokens=False)], dtype=torch.int64)
        self.pad = torch.tensor([tokenizer_src.encode("<pad>", add_special_tokens=False)], dtype=torch.int64)
        self.unk = torch.tensor([tokenizer_src.encode("<unk>", add_special_tokens=False)], dtype=torch.int64)
        self.mask = torch.tensor([tokenizer_src.encode("<mask>", add_special_tokens=False)], dtype=torch.int64)
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        src_text = row[self.src_lang]
        tgt_text = row[self.tgt_format]

        encoder_input_ids = self.tokenizer_src(src_text, padding='max_length', truncation=True, max_length=self.seq_len, return_tensors='pt')['input_ids']
        target = self.tokenizer_tgt(tgt_text, padding='max_length', truncation=True, max_length=(self.seq_len + 1), return_tensors='pt')['input_ids']


        # Create decoder inputs by shifting target sequence to the right
        decoder_input_ids = target[:, :-1].clone()
        
        # Shift target to the left to exclude start-of-sequence token
        target = target[:, 1:]
        
        
        assert decoder_input_ids.size(0) == self.seq_len
        assert encoder_input_ids.size(0) == self.seq_len
        assert target.size(0) == self.seq_len

        return {
            'encoder_input': encoder_input_ids.squeeze(),
            'decoder_input': decoder_input_ids.squeeze(),
            'encoder_mask': (encoder_input_ids != self.pad).unsqueeze(0).unsqueeze(0).int(),
            'decoder_mask': (decoder_input_ids != self.pad).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input_ids.size(0)),
            'label': target.squeeze(),
            'src_text': src_text,
            'tgt_text': tgt_text,
        }
        
def causal_mask(length):
    mask = torch.ones(length, length)
    mask = torch.triu(mask)
    return mask.unsqueeze(0).int() == 0