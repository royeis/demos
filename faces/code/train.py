DATA_PATH = '/User/demos/demos/faces/dataset/'
ARTIFACTS_PATH = '/User/demos/demos/faces/artifacts/'
MODELS_PATH = '/User/demos/demos/faces/models.py'

import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch
import horovod.torch as hvd
import importlib.util
import os
from pickle import dump
import pandas as pd
import v3io_frames as v3f
from mlrun import get_or_create_ctx


def train(context, processed_data, model_name='model.bst', cuda=False, horovod=False):
    
    hvd.init()

    if cuda:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            context.logger.info(f"Running on cuda device: {device}")
        else:
            device = torch.device("cpu")
            context.logger.info("Requested running on cuda but no cuda device available.\nRunning on cpu")
    else:
        device = torch.device("cpu")
    
    context.logger.info(f'device: {device}')
    context.logger.info('Client')
    client = v3f.Client('framesd:8081', container="users")
    with open(processed_data.url, 'r') as f:                      
        t = f.read()

    context.logger.info('Loading dataset')
    data_df = client.read(backend="kv", table=t, reset_index=False, filter='label != -1')
    X = data_df[['c'+str(i).zfill(3) for i in range(128)]].values
    y = data_df['label'].values

    n_classes = len(set(y))

    X = torch.as_tensor(X, device=device)
    y = torch.tensor(y, device=device).reshape(-1, 1)
    
    input_dim = 128
    hidden_dim = 64
    output_dim = n_classes

    context.logger.info('Preparing model architecture')
    spec = importlib.util.spec_from_file_location('models', MODELS_PATH)
    models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models)

    model = models.FeedForwardNeuralNetModel(input_dim, hidden_dim, output_dim)
    model.to(device)
    model = model.double()
    
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.05
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dataset = data.TensorDataset(X, y)
    train_loader = data.DataLoader(dataset)
    
    
    if cuda and horovod:
        context.logger.info('preparing for horovod distributed training')
        torch.cuda.set_device(hvd.local_rank())
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
        train_sampler = data.distributed.DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
        train_loader = data.DataLoader(dataset, sampler=train_sampler)
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    
    context.logger.info('Starting training process')
    for epoch in range(20):
        for features, target in train_loader:
            optimizer.zero_grad()
            out = model(features)
            loss = criterion(out, target[0])
            loss.backward()
            optimizer.step()
            
    
    if not horovod or hvd.rank() == 0:
        context.logger.info('Save model')
        dump(model._modules, open(model_name, 'wb'))
        context.log_artifact('model', src_path=model_name, target_path=model_name, labels={'framework': 'Pytorch-FeedForwardNN'})
        os.remove(model_name)

if __name__ == "__main__":
    context = get_or_create_ctx('train')
    processed_data = context.get_input('processed_data', None)
    cuda = context.get_input('cuda', False)
    horovod = context.get_input('horovod', False)
    context.logger.info('initiating training function')
    train(context, processed_data=processed_data, cuda=cuda, horovod=horovod)