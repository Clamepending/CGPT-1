from pathlib import Path
def get_config():
    return {
        "batch_size" : 8,
        "num_epochs" : 5,
        "lr": 10**-4,
        "seq_len": 80,
        "d_model" : 128,
        "src_lang" : "ChEMBL_ID",
        "tgt_format" : "SMILES",
        "model_folder" : "weights",
        "model_basename": "tdmodel_",
        "preload" : None,
        "tokenizer_file" : "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "SMILES dataset" : './data/train_dataset.csv',
        "validation dataset" : './data/test_dataset.csv',
        "decoder only" : True,
    }
    
    
def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)


def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])