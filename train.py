import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

from dataset import ChemDataset, causal_mask
import CGPT_tokenizer
from config import get_config, get_weights_file_path, latest_weights_file_path

from pathlib import Path
import pandas as pd
from CGPT_utils import *
import os
import warnings
from tqdm import tqdm

import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from transformers import RobertaTokenizerFast


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.encode("<s>", add_special_tokens=False)
    eos_idx = tokenizer_tgt.encode("</s>", add_special_tokens=False)

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def get_ds(config):
    
    chem_vocab_size = 500
 
    
    
    chem_tokenizer_dir = "chem_tokenizer"
    text_tokenizer_dir = "text_tokenizer"

    # Check if the directory exists
    if os.path.exists(chem_tokenizer_dir) and os.path.exists(text_tokenizer_dir):
        try:
            # Load the tokenizers
            chem_tokenizer = RobertaTokenizerFast.from_pretrained(chem_tokenizer_dir)
            text_tokenizer = RobertaTokenizerFast.from_pretrained(text_tokenizer_dir)
            print("Tokenizers loaded successfully.")
        except Exception as e:
            print(f"Error loading tokenizers: {e}")
    else:
        print("Tokenizers directory not found.")
        chem_tokenizer = CGPT_tokenizer.make_custum_tokenizer(csv_path=config["SMILES dataset"], column="SMILES", vocab_size=chem_vocab_size)
        text_tokenizer = CGPT_tokenizer.make_default_tokenizer()
        text_tokenizer.save_pretrained("chem_tokenizer")
        chem_tokenizer.save_pretrained("text_tokenizer")
    
    

    # data = pd.read_csv(config["SMILES dataset"])
    
    # train_ds_size = int(0.9*len(data))
    # validation_ds_size = len(data) - train_ds_size
    # train_ds_raw, val_ds_raw = random_split(data, [train_ds_size, validation_ds_size])
    # train_df = pd.DataFrame(list(train_ds_raw))
    # val_df = pd.DataFrame(list(val_ds_raw))
    
    # validation_ds_size = int(0 * len(data))
    # val_df = data.iloc[-validation_ds_size:]
    # train_df = data.iloc[:-validation_ds_size]
    
    train_df = pd.read_csv(config["SMILES dataset"])
    val_df = pd.read_csv(config["validation dataset"])
    
    train_ds = ChemDataset(train_df, text_tokenizer, chem_tokenizer, config['src_lang'], config['tgt_format'], config['seq_len'])
    validation_ds = ChemDataset(val_df, text_tokenizer, chem_tokenizer, config['src_lang'], config['tgt_format'], config['seq_len'])
    
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(validation_ds, batch_size=1, shuffle=True)
    
    return train_dataloader, val_dataloader, text_tokenizer, chem_tokenizer


def get_model(config, vocab_src_len, vocab_tgt_len):
    if config["decoder only"]:
        model = build_decoder_only_transformer(vocab_tgt_len, config['seq_len'], config['d_model'])
    else:
        model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["Description"][0]
            target_text = batch["SMILES"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()
    
def train_model(config):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
        
    print(f"src and tgt vocab sizes: {tokenizer_src.vocab_size, tokenizer_tgt.vocab_size}")
    
    model = get_model(config, tokenizer_src.vocab_size, tokenizer_tgt.vocab_size).to(device)
    
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.encode('<pad>', add_special_tokens = False)[0], label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)
            
            # print(f"shapes: {encoder_input.shape, decoder_input.shape, encoder_mask.shape}")

            # print(f"input: {encoder_input}")
            # Run the tensors through the encoder, decoder and the projection layer
            if config["decoder only"]:
                decoder_output = model.decode(decoder_input, decoder_mask)
            else:
                encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
                
            
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.vocab_size), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        # run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()