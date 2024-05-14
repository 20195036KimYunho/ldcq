import os
import sys

curr_folder = os.path.abspath(__file__)
parent_folder = os.path.dirname(os.path.dirname(curr_folder))
sys.path.append(parent_folder)

from models.diffusion_models import (
    Model_mlp,
    Model_cnn_mlp,
    Model_Cond_Diffusion,
)
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pickle
import gym
import wandb
from argparse import ArgumentParser


# import d4rl


class PriorDataset(Dataset):
    def __init__(
        self, dataset_dir, filename, train_or_test, test_prop, sample_z=False
    ):
        # just load it all into RAM
        self.state_all = np.load(os.path.join(dataset_dir, filename + "_states.npy"), allow_pickle=True)
        self.latent_all = np.load(os.path.join(dataset_dir, filename + "_latents.npy"), allow_pickle=True)
        if sample_z:
            self.latent_all_std = np.load(os.path.join(dataset_dir, filename + "_latents_std.npy"), allow_pickle=True)

        self.state_mean = self.state_all.mean(axis=0)
        self.state_std = self.state_all.std(axis=0)
        # self.state_all = (self.state_all - self.state_mean) / self.state_std

        self.latent_mean = self.latent_all.mean(axis=0)
        self.latent_std = self.latent_all.std(axis=0)
        # self.latent_all = (self.latent_all - self.latent_mean) / self.latent_std
        self.sample_z = sample_z
        n_train = int(self.state_all.shape[0] * (1 - test_prop))
        if train_or_test == "train":
            self.state_all = self.state_all[:n_train]
            self.latent_all = self.latent_all[:n_train]
            if sample_z:
                self.latent_all_std = self.latent_all_std[:n_train]
        elif train_or_test == "test":
            self.state_all = self.state_all[n_train:]
            self.latent_all = self.latent_all[n_train:]
            if sample_z:
                self.latent_all_std = self.latent_all_std[n_train:]
        else:
            raise NotImplementedError

    def __len__(self):
        return self.state_all.shape[0]

    def __getitem__(self, index):
        state = self.state_all[index]
        latent = self.latent_all[index]
        if self.sample_z:
            latent_std = self.latent_all_std[index]
            latent = np.random.normal(latent, latent_std)
            # latent = (latent - self.latent_mean) / self.latent_std
        # else:
        #    latent = (latent - self.latent_mean) / self.latent_std
        return (state, latent)


def train(args):
    # experiment = Experiment(api_key = '', project_name = '')
    # experiment.log_parameters({'lrate':args.lrate,
    #                         'batch_size':args.batch_size,
    #                         'net_type':args.net_type,
    #                         'sample_z':args.sample_z,
    #                         'diffusion_steps':args.diffusion_steps,
    #                         'skill_model_filename':args.skill_model_filename,
    #                         'normalize_latent':args.normalize_latent,
    #                         'schedule': args.schedule})

    # get datasets set up
    wandb.init(project=args.env,
               name='train_diffusion/'+args.skill_model_filename,
               config={'lrate': args.lrate,
                       'batch_size': args.batch_size,
                       'net_type': args.net_type,
                       'sample_z': args.sample_z,
                       'diffusion_steps': args.diffusion_steps,
                       'skill_model_filename': args.skill_model_filename,
                       'normalize_latent': args.normalize_latent,
                       'schedule': args.schedule,
                        'use_contrastive': args.use_contrastive,
                        'num_categorical_interval': args.num_categorical_interval,
                        'contrastive_ratio': args.contrastive_ratio
                       }
               )
    torch_data_train = PriorDataset(
        args.dataset_dir, args.skill_model_filename[:-4], train_or_test="train", test_prop=args.test_split, sample_z=args.sample_z
    )
    dataload_train = DataLoader(
        torch_data_train, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    if args.test_split > 0.0:
        torch_data_test = PriorDataset(
            args.dataset_dir, args.skill_model_filename[:-4], train_or_test="test", test_prop=args.test_split, sample_z=args.sample_z
        )
        dataload_test = DataLoader(
            torch_data_test, batch_size=args.batch_size, shuffle=False, num_workers=4
        )

    x_shape = torch_data_train.state_all.shape[1]
    y_dim = torch_data_train.latent_all.shape[1]

    # create model
    nn_model = Model_mlp(
        x_shape, args.n_hidden, y_dim, embed_dim=128, net_type=args.net_type
    ).to(args.device)
    model = Model_Cond_Diffusion(
        nn_model,
        betas=(1e-4, 0.02),
        n_T=args.diffusion_steps,
        device=args.device,
        x_dim=x_shape,
        y_dim=y_dim,
        drop_prob=args.drop_prob,
        guide_w=0.0,
        normalize_latent=args.normalize_latent,
        schedule=args.schedule,
    ).to(args.device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lrate)
    best_test_loss = 10000000
    best_train_loss = 10000000

    for ep in tqdm(range(args.n_epoch), desc="Epoch"):
        model.train()

        # lrate decay
        # optim.param_groups[0]["lr"] = args.lrate * ((np.cos((ep / args.n_epoch) * np.pi) + 1) / 2)
        optim.param_groups[0]["lr"] = args.lrate * ((np.cos((ep / 75) * np.pi) + 1))

        # train loop
        model.train()
        # pbar = tqdm(dataload_train)
        pbar=dataload_train
        loss_ep, n_batch = 0, 0

        for x_batch, y_batch in pbar:
            x_batch = x_batch.type(torch.FloatTensor).to(args.device)
            y_batch = y_batch.type(torch.FloatTensor).to(args.device)
            loss = model.loss_on_batch(x_batch, y_batch, args.predict_noise)
            optim.zero_grad()
            loss.backward()
            loss_ep += loss.detach().item()
            n_batch += 1
            # pbar.set_description(f"train loss: {loss_ep/n_batch:.4f}")
            optim.step()
            wandb.log({"train_diffusion/train_loss": loss_ep/n_batch, "epoch": ep})

        # torch.save(nn_model, os.path.join(args.checkpoint_dir, args.skill_model_filename[:-4] + '_diffusion_prior.pt'))

        # test loop
        if args.test_split > 0.0:
            model.eval()
            # pbar = tqdm(dataload_test)
            pbar= dataload_test
            loss_ep, n_batch = 0, 0

            with torch.no_grad():
                for x_batch, y_batch in pbar:
                    x_batch = x_batch.type(torch.FloatTensor).to(args.device)
                    y_batch = y_batch.type(torch.FloatTensor).to(args.device)
                    loss = model.loss_on_batch(x_batch, y_batch, args.predict_noise)
                    loss_ep += loss.detach().item()
                    n_batch += 1
                    # pbar.set_description(f"test loss: {loss_ep/n_batch:.4f}")
                    wandb.log({"train_diffusion/test_loss": loss_ep/n_batch, "epoch": ep})

            if loss_ep < best_test_loss:
                best_test_loss = loss_ep
                torch.save(nn_model, os.path.join(args.checkpoint_dir, args.skill_model_filename[:-4] + '_diffusion_prior_best.pt'))
        
        else:
            if ep % 50 == 0 or ep == args.n_epoch-1:
                checkpoint_path = os.path.join(args.checkpoint_dir, args.skill_model_filename[:-4]+'_'+str(ep)+'_'+'.pt')
                torch.save(nn_model, checkpoint_path)
                
            elif loss_ep < best_train_loss:
                best_train_loss = loss_ep
                torch.save(nn_model, os.path.join(args.checkpoint_dir, args.skill_model_filename[:-4] + '_diffusion_prior_best.pt'))
                
if __name__ == "__main__":
    parser = ArgumentParser()

    # #####해놓은 것들이 argument 잘못넣으면 안 돌아가는 것들, 돌리기 전 꼭 확인할 것
    parser.add_argument('--env', type=str, default='antmaze-large-diverse-v2')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--lrate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--net_type', type=str, default='unet')
    parser.add_argument('--n_hidden', type=int, default=512)
    parser.add_argument('--test_split', type=float, default=0.2)
    parser.add_argument('--sample_z', type=int, default=0)

    parser.add_argument('--checkpoint_dir', type=str, default=parent_folder+'/checkpoints/')
    parser.add_argument('--dataset_dir', type=str, default=parent_folder+'/data/')
    parser.add_argument('--skill_model_filename', type=str)
    parser.add_argument('--append_goals', type=int, default=0)

    parser.add_argument('--drop_prob', type=float, default=0.0)
    parser.add_argument('--diffusion_steps', type=int, default=100)
    parser.add_argument('--cfg_weight', type=float, default=0.0)
    parser.add_argument('--predict_noise', type=int, default=0)
    parser.add_argument('--normalize_latent', type=int, default=0)
    parser.add_argument('--schedule', type=str, default='linear')
    
    parser.add_argument('--num_categorical_interval', type=int, default=10)
    parser.add_argument('--use_contrastive', type=int, default=0)
    parser.add_argument('--contrastive_ratio', type=float, default=1.0)
    args = parser.parse_args()

    train(args)
