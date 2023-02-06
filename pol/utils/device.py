import torch

def transfer_data_batch(data_batch, device):
    if isinstance(data_batch, list) or isinstance(data_batch, tuple):
        data_batch = [data.to(device) for data in data_batch]
    else:
        data_batch = {k: data.to(device) for k, data in data_batch.items()}
    return data_batch
