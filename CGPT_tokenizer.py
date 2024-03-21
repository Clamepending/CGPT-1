from tokenizers import ByteLevelBPETokenizer
import pandas as pd
from transformers import RobertaTokenizerFast
import os

def make_custum_tokenizer(csv_path: str = "./data/test_dataset.csv", column: str = "SMILES", vocab_size: int = 265):
    # Check if tokenizer file exists
    if os.path.exists("./tokenizer/vocab.json"):
        # If tokenizer file exists, load tokenizer from pretrained
        print("found tokenizer already.")
    else:
        # Read your CSV file into a DataFrame
        df = pd.read_csv(csv_path)

        # Extract the text from the desired column
        text_column = df[column].tolist()

        # Initialize ByteLevelBPETokenizer
        tokenizer = ByteLevelBPETokenizer()

        # Train the tokenizer
        tokenizer.train_from_iterator(text_column, vocab_size=vocab_size, min_frequency=2, special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])

        # Save the trained tokenizer
        tokenizer.save_model("tokenizer")

    # Load the trained tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained("tokenizer")
        
    return tokenizer

def make_default_tokenizer():
    tokenizer = RobertaTokenizerFast.from_pretrained("FacebookAI/roberta-base")
    return tokenizer