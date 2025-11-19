import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

def continue_training(checkpoint_path, generator: DDP, discriminator: DDP, optimizer_d: optim.Optimizer, optimizer_g: optim.Optimizer) -> int:
    """load the latest checkpoints and optimizers"""
    generator_dict = {}
    discriminator_dict = {}
    optimizer_d_dict = {}
    optimizer_g_dict = {}
    
    # globt all the checkpoints in the directory
    for file in os.listdir(checkpoint_path):
        if file.endswith(".pt"):
            name, epoch_str = file.rsplit('_', 1)
            epoch = int(epoch_str.split('.')[0])
            
            if name.startswith("generator"):
                generator_dict[epoch] = file
            elif name.startswith("discriminator"):
                discriminator_dict[epoch] = file
            elif name.startswith("optimizerd"):
                optimizer_d_dict[epoch] = file
            elif name.startswith("optimizerg"):
                optimizer_g_dict[epoch] = file
    
    # get the largest epoch
    common_epochs = set(generator_dict.keys()) & set(discriminator_dict.keys()) & set(optimizer_d_dict.keys()) & set(optimizer_g_dict.keys())
    if common_epochs:
        max_epoch = max(common_epochs)
        generator_path = os.path.join(checkpoint_path, generator_dict[max_epoch])
        discriminator_path = os.path.join(checkpoint_path, discriminator_dict[max_epoch])
        optimizer_d_path = os.path.join(checkpoint_path, optimizer_d_dict[max_epoch])
        optimizer_g_path = os.path.join(checkpoint_path, optimizer_g_dict[max_epoch])
        
        # load model and optimizer
        generator.module.load_state_dict(torch.load(generator_path, map_location='cpu'))
        discriminator.module.load_state_dict(torch.load(discriminator_path, map_location='cpu'))
        optimizer_d.load_state_dict(torch.load(optimizer_d_path, map_location='cpu'))
        optimizer_g.load_state_dict(torch.load(optimizer_g_path, map_location='cpu'))
        
        print(f'resume model and optimizer from {max_epoch} epoch')
        return max_epoch + 1
    
    else:
        if generator_dict:
            max_epoch = max(generator_dict.keys())
            generator_path = os.path.join(checkpoint_path, generator_dict[max_epoch])
            generator.module.load_state_dict(torch.load(generator_path, map_location='cpu'))

            if max_epoch in discriminator_dict:
                discriminator_path = os.path.join(checkpoint_path, discriminator_dict[max_epoch])
                discriminator.module.load_state_dict(torch.load(discriminator_path, map_location='cpu'))

            print('pretrained model loaded')
        return 0

def get_model_statedict(model: DDP):
    """Remove unwanted key in torch.compile"""
    state_dict = model.module.state_dict()
    unwanted_prefix = '_orig_mod.' 
    for k,v in list(state_dict.items()):
        if unwanted_prefix in k: 
            state_dict[k.replace(unwanted_prefix, '')] = state_dict.pop(k) 
    return state_dict