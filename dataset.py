import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class ChemDataset(Dataset):
    def __init__(self, dataframe, tokenizer_src, tokenizer_tgt, src_lang, tgt_format, seq_len):
        super().__init__()
        
        self.dataframe = dataframe
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_format = tgt_format
        self.seq_len = seq_len
        
        self.sos = tokenizer_src.encode("<s>", add_special_tokens=False)[0]
        self.eos = tokenizer_src.encode("</s>", add_special_tokens=False)[0]
        self.pad = tokenizer_src.encode("<pad>", add_special_tokens=False)[0]
        self.unk = tokenizer_src.encode("<unk>", add_special_tokens=False)[0]
        self.mask = tokenizer_src.encode("<mask>", add_special_tokens=False)[0]
        
        self.dataframe['encoder_input'] = self.dataframe[src_lang].apply(lambda x: tokenizer_src.encode(x, padding='max_length', truncation=True, max_length=self.seq_len, return_tensors='pt'))
        self.dataframe['decoder_input'] = self.dataframe[tgt_format].apply(lambda x: tokenizer_tgt.encode(x, padding='max_length', truncation=True, max_length=(self.seq_len + 1), return_tensors='pt'))
        self.dataframe['label'] = self.dataframe['decoder_input'].apply(lambda x: x[:,1:])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        encoder_input_ids = row['encoder_input'][0]
        decoder_input_ids = row['decoder_input'][0,:-1]
        target = row['label'][0]
        
        # print(f"shape: {decoder_input_ids.shape}")
        # print(f"2 shape: {causal_mask(decoder_input_ids.size(1)).shape}")
        
        encoder_mask = (encoder_input_ids != self.pad).unsqueeze(0).unsqueeze(0).int()
        decoder_mask = (decoder_input_ids != self.pad).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input_ids.size(0))
        
        assert decoder_input_ids.size(0) == self.seq_len, f"{decoder_input_ids.size(0)} != {self.seq_len}"
        assert encoder_input_ids.size(0) == self.seq_len, f"{encoder_input_ids.size(0)} != {self.seq_len}"
        assert target.size(0) == self.seq_len, f"{target.size(0)} != {self.seq_len}"

        return {
            'encoder_input': encoder_input_ids,
            'decoder_input': decoder_input_ids,
            'encoder_mask': encoder_mask,
            'decoder_mask': decoder_mask,
            'label': target,
            'src_text': row[self.src_lang],
            'tgt_text': row[self.tgt_format],
        }
        
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
