import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from models.model import DCAEWrapper, compute_dispersive_loss
from models.discriminator import AEDiscriminator, discriminator_loss, generator_loss, feature_loss
from dataset import AEDataset, AEDatasetCollate
from utils.scheduler import get_cosine_schedule_with_warmup
from utils.load import continue_training, get_model_statedict
from utils.muon import get_muon_optimizer

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

from dataclasses import dataclass
            
@dataclass
class TrainConfig:
    train_dataset_path: str = 'filelists/filelist.txt'
    model_config_path: str = 'models/default_config.json'
    vocoder_config_path: str = 'models/bigvgan_v2_22khz_80band_256x/config.json'
    output_channels: int = 80 # output channels of mel spectrogram

    batch_size: int = 4
    gen_learning_rate: float = 1e-4
    disc_learning_rate: float = 1e-4
    max_grad_norm: float = 500.0
    recon_loss_factor: float = 5
    disp_loss_factor: float = 0.25
    max_steps: int = 10000000000000000

    model_save_path: str = './checkpoints'
    log_dir: str = './runs'
    log_interval: int = 32
    save_interval: int = 1000

    warmup_steps: int = 0
    num_workers: int = 4
    segment_samples: int = 256 * 16 * 60

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '6666'
    dist.init_process_group("gloo" if os.name == "nt" else "nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def _init_config(train_config: TrainConfig):
    if not os.path.exists(train_config.model_save_path):
        print(f'Creating {train_config.model_save_path}')
        os.makedirs(train_config.model_save_path, exist_ok=True)

def train(rank, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    train_config = TrainConfig()

    _init_config(train_config)

    generator = DCAEWrapper(train_config.model_config_path).to(rank)
    discriminator = AEDiscriminator(train_config.output_channels).to(rank)
    loss_fn = nn.L1Loss().to(rank)

    if rank == 0:
        generator.print_num_params()

    generator = DDP(generator, device_ids=[rank])
    discriminator = DDP(discriminator, device_ids=[rank])

    verbose = False
    train_dataset = AEDataset(train_config.train_dataset_path, train_config.vocoder_config_path,
                                train_config.segment_samples, verbose)
    train_sampler = DistributedSampler(train_dataset,
                                        num_replicas=dist.get_world_size(),
                                        rank=dist.get_rank(),
                                        shuffle=True, drop_last=True)
    train_collate_fn = AEDatasetCollate(verbose)
    train_dataloader = DataLoader(train_dataset, num_workers=train_config.num_workers, pin_memory=True,
                                  collate_fn=train_collate_fn, persistent_workers=True,
                                  batch_size=train_config.batch_size, prefetch_factor=8, drop_last=True)

    if rank == 0:
        writer = SummaryWriter(train_config.log_dir)

    optimizer_g = get_muon_optimizer([generator], lr=train_config.gen_learning_rate)
    optimizer_d = get_muon_optimizer([discriminator], lr=train_config.disc_learning_rate)
    scheduler_g = get_cosine_schedule_with_warmup(optimizer_g, num_warmup_steps=int(train_config.warmup_steps), num_training_steps=train_config.max_steps)
    scheduler_d = get_cosine_schedule_with_warmup(optimizer_d, num_warmup_steps=int(train_config.warmup_steps), num_training_steps=train_config.max_steps)
    

    # load latest checkpoints if possible
    current_steps = continue_training(train_config.model_save_path, generator, discriminator, optimizer_d, optimizer_g)

    # backward of compiled weight_norm conv2d is broken in pytorch now (version 2.9)
    # generator.module.forward = torch.compile(generator.module.forward, dynamic=False)
    # discriminator.module.forward = torch.compile(discriminator.module.forward, dynamic=False)

    generator.train()
    discriminator.train()

    stop = False
    epoch = 0

    while not stop: # loop over the train_dataset multiple times
        train_sampler.set_epoch(epoch)
        epoch += 1
        dataloader = train_dataloader
        if rank == 0:
            # progress_bar = tqdm()
            dataloader = tqdm(dataloader, desc=f'Epoch {epoch}', dynamic_ncols=True)

        for batch_idx, datas in enumerate(dataloader):
            datas = [data.to(rank, non_blocking=True) for data in datas]
            features, = datas

            features_recon, latent = generator(features)
            optimizer_d.zero_grad()
            
            # discriminator
            y_df_hat_r, y_df_hat_g, _, _ = discriminator(features, features_recon.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
            
            loss_disc_all = loss_disc_f
            loss_disc_all.backward()

            grad_norm_discriminator = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), train_config.max_grad_norm)
            optimizer_d.step()
            scheduler_d.step()

            # generator
            optimizer_g.zero_grad()
            loss_recon = loss_fn(features_recon, features)
            loss_disp = compute_dispersive_loss(latent)

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = discriminator(features, features_recon)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)

            loss_gen_all = loss_gen_f + loss_fm_f + loss_recon * train_config.recon_loss_factor + loss_disp * train_config.disp_loss_factor
            loss_gen_all.backward()

            grad_norm_g = torch.nn.utils.clip_grad_norm_(generator.parameters(), train_config.max_grad_norm)
            optimizer_g.step()
            scheduler_g.step()

            if rank == 0:
                # progress_bar.update(1)
                current_steps += 1

            if rank == 0 and batch_idx % train_config.log_interval == 0:
                steps = current_steps
                writer.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                writer.add_scalar("training/fm_loss_discriminator", loss_fm_f.item(), steps)
                writer.add_scalar("training/gen_loss_discriminator", loss_gen_f.item(), steps)
                writer.add_scalar("training/disc_loss_discriminator", loss_disc_f.item(), steps)
                writer.add_scalar("training/recon_loss", loss_recon.item(), steps)
                writer.add_scalar("training/disp_loss", loss_disp.item(), steps)
                writer.add_scalar("grad_norm/grad_norm_discriminator", grad_norm_discriminator, steps)
                writer.add_scalar("grad_norm/grad_norm_g", grad_norm_g, steps)
                writer.add_scalar("learning_rate/learning_rate_d", scheduler_d.get_last_lr()[0], steps)
                writer.add_scalar("learning_rate/learning_rate_g", scheduler_g.get_last_lr()[0], steps)

            if rank == 0 and current_steps % train_config.save_interval == 0:
                torch.save(get_model_statedict(generator), os.path.join(train_config.model_save_path, f'generator_{current_steps}.pt'))
                torch.save(get_model_statedict(discriminator), os.path.join(train_config.model_save_path, f'discriminator_{current_steps}.pt'))
                torch.save(optimizer_d.state_dict(), os.path.join(train_config.model_save_path, f'optimizerd_{current_steps}.pt'))
                torch.save(optimizer_g.state_dict(), os.path.join(train_config.model_save_path, f'optimizerg_{current_steps}.pt'))
                print(f"Step {current_steps}, Loss {loss_recon.item()}")

            if current_steps > train_config.max_steps:
                stop = True
                break

    cleanup()

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)
