
import sys
import os

curr_folder = os.path.abspath(__file__)
parent_folder = os.path.dirname(os.path.dirname(curr_folder))
sys.path.append(parent_folder)

import pickle
import argparse

import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from models.skill_model import SkillModel
import h5py

from utils.utils import get_dataset

def train(model, optimizer, train_state_decoder,epoch):
    losses = []

    for batch_id, data in enumerate(train_loader):
        data = data.cuda()
        states = data[:, :, :model.state_dim]
        actions = data[:, :, model.state_dim:]
        if train_state_decoder:
            loss_tot, s_T_loss, a_loss, kl_loss, diffusion_loss,contrastive_loss = model.get_losses(states, actions, train_state_decoder)
        else:
            loss_tot, a_loss, kl_loss, diffusion_loss,contrastive_loss = model.get_losses(states, actions, train_state_decoder)
        model.zero_grad()
        loss_tot.backward()
        optimizer.step()
        # log losses
        wandb.log({"train_skill/loss": loss_tot.item()})
        wandb.log({"train_skill/a_loss": a_loss.item()})
        wandb.log({"train_skill/s_T_loss": s_T_loss.item() if train_state_decoder else None})
        wandb.log({"train_skill/kl_loss": kl_loss.item()})
        wandb.log({"train_skill/diffusion_loss": diffusion_loss.item() if train_diffusion_prior else diffusion_loss})
        wandb.log({"train_skill/contrastive_loss": contrastive_loss.item() if use_contrastive else contrastive_loss})

        losses.append(loss_tot.item())

    return np.mean(losses)


def test(model, test_state_decoder):
    losses = []
    s_T_losses = []
    a_losses = []
    kl_losses = []
    s_T_ents = []
    diffusion_losses = []

    with torch.no_grad():
        for batch_id, data in enumerate(test_loader):
            data = data.cuda()
            states = data[:, :, :model.state_dim]
            actions = data[:, :, model.state_dim:]
            if test_state_decoder:
                loss_tot, s_T_loss, a_loss, kl_loss, diffusion_loss,contrastive_loss = model.get_losses(states, actions, test_state_decoder)
                s_T_losses.append(s_T_loss.item())
            else:
                loss_tot, a_loss, kl_loss, diffusion_loss,contrastive_loss = model.get_losses(states, actions, test_state_decoder)
            # log losses
            losses.append(loss_tot.item())
            a_losses.append(a_loss.item())
            kl_losses.append(kl_loss.item())
            contrastive_loss.append(contrastive_loss.item() if use_contrastive else contrastive_loss)
            diffusion_losses.append(diffusion_loss.item() if train_diffusion_prior else diffusion_loss)

    if train_diffusion_prior:
        return np.mean(losses), np.mean(s_T_losses), np.mean(a_losses), np.mean(kl_losses), np.mean(diffusion_losses), np.mean(contrastive_loss)
    return np.mean(losses), np.mean(s_T_losses), np.mean(a_losses), np.mean(kl_losses), None, np.mean(contrastive_loss)


parser = argparse.ArgumentParser()
# #####해놓은 것들이 argument 잘못넣으면 안 돌아가는 것들, 돌리기 전 꼭 확인할 것
parser.add_argument('--env', type=str, default='antmaze-large-diverse-v2')
parser.add_argument('--beta', type=float, default=0.05)
parser.add_argument('--conditional_prior', type=int, default=0)
parser.add_argument('--train_diffusion_prior', type=int, default=1)
parser.add_argument('--z_dim', type=int, default=16)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--policy_decoder_type', type=str, default='autoregressive')
parser.add_argument('--state_decoder_type', type=str, default='mlp')
parser.add_argument('--a_dist', type=str, default='normal') 
parser.add_argument('--horizon', type=int, default=30)
parser.add_argument('--separate_test_trajectories', type=int, default=0)
parser.add_argument('--test_split', type=float, default=0.2)
parser.add_argument('--get_rewards', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=50000)
parser.add_argument('--start_training_state_decoder_after', type=int, default=0)
parser.add_argument('--normalize_latent', type=int, default=0)
parser.add_argument('--append_goals', type=int, default=0)

parser.add_argument('--checkpoint_dir', type=str, default=parent_folder+'/checkpoints/')
parser.add_argument('--dataset_dir', type=str, default=parent_folder+'/data/')

parser.add_argument('--num_categorical_interval', type=int, default=10)
parser.add_argument('--use_contrastive', type=int, default=0)
parser.add_argument('--contrastive_ratio', type=float, default=1.0)

args = parser.parse_args()

batch_size = 128  # default 128

h_dim = 256
z_dim = args.z_dim
lr = args.lr  # 5e-5
wd = 0.0
H = args.horizon
stride = 1
n_epochs = args.num_epochs
test_split = args.test_split
a_dist = args.a_dist  # 'normal' # 'tanh_normal' or 'softmax'
encoder_type = 'gru'  # 'transformer' #'state_sequence'
state_decoder_type = args.state_decoder_type
policy_decoder_type = args.policy_decoder_type
load_from_checkpoint = False
per_element_sigma = True
start_training_state_decoder_after = args.start_training_state_decoder_after
train_diffusion_prior = args.train_diffusion_prior 
checkpoint_dir =args.checkpoint_dir
dataset_dir = args.dataset_dir

beta = args.beta  # 1.0 # 0.1, 0.01, 0.001
conditional_prior = args.conditional_prior  # True

num_categorical_interval = args.num_categorical_interval
use_contrastive = args.use_contrastive
contrastive_ratio= args.contrastive_ratio
env_name = args.env

dataset_file = os.path.join(dataset_dir,env_name+'.pkl')
with open(dataset_file, "rb") as f:
    dataset = pickle.load(f)

states = dataset['observations']  # [:10000]
# next_states = dataset['next_observations']
actions = dataset['actions']  # [:10000]

# 데이터셋의 형태
print("State:", states.shape)
print("action:", actions.shape)
N = states.shape[0]  # N: 전체 데이터셋 갯수

state_dim = states.shape[1] + args.append_goals * 2  # goal의 (x,y)추가
a_dim = actions.shape[1]

N_train = int((1-test_split)*N)
N_test = N - N_train

dataset = get_dataset(env_name, H, stride, test_split, get_rewards=args.get_rewards, separate_test_trajectories=args.separate_test_trajectories, append_goals=args.append_goals)

obs_chunks_train = dataset['observations_train']
action_chunks_train = dataset['actions_train']
# train chunk의 형태
print("Train_Observations:", obs_chunks_train.shape)  # chunk size x T(H) x s_dim
print("Train_Action:", action_chunks_train.shape)

if test_split > 0.0:
    obs_chunks_test = dataset['observations_test']
    action_chunks_test = dataset['actions_test']

filename = env_name+'_H_'+str(H)+'_adist_'+a_dist+'_use_contrastive_'+str(use_contrastive)+'_num_categorical_interval_'+str(num_categorical_interval)+'_contrastive_ratio_'+str(contrastive_ratio)+'_getrewards_'+str(args.get_rewards)+'_appendgoals_'+str(args.append_goals)

# filename = 'skill_model_'+env_name+'_encoderType('+encoder_type+')_state_dec_'+str(state_decoder_type)+'_policy_dec_'+str(policy_decoder_type)+'_H_'+str(H)+'_b_'+str(beta)+'_conditionalp_'+str(conditional_prior)+'_zdim_'+str(z_dim) + \
#     '_adist_'+a_dist+'_testSplit_'+str(test_split)+'_separatetest_'+str(args.separate_test_trajectories)+'_getrewards_'+str(args.get_rewards)+'_appendgoals_'+str(args.append_goals)
# experiment = Experiment(api_key='', project_name='')
# experiment.add_tag('noisy2')

model = SkillModel(state_dim, 
                   a_dim, 
                   z_dim, 
                   h_dim, 
                   horizon=H, 
                   a_dist=a_dist, 
                   beta=beta, 
                   fixed_sig=None, 
                   encoder_type=encoder_type, 
                   state_decoder_type=state_decoder_type, policy_decoder_type=policy_decoder_type,
                   per_element_sigma=per_element_sigma, conditional_prior=conditional_prior, train_diffusion_prior=train_diffusion_prior, 
                   normalize_latent=args.normalize_latent,num_categocical_interval=num_categorical_interval,use_contrastive=use_contrastive,
                   contrastive_ratio=contrastive_ratio
                   ).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

# 사용하지 않는 코드
# if load_from_checkpoint:
# 	PATH = os.path.join(checkpoint_dir,filename+'_best_sT.pth')
# 	checkpoint = torch.load(PATH)
# 	model.load_state_dict(checkpoint['model_state_dict'])
# 	E_optimizer.load_state_dict(checkpoint['E_optimizer_state_dict'])
# 	M_optimizer.load_state_dict(checkpoint['M_optimizer_state_dict'])

wandb.init(
    project=env_name,
    name='train_skill/'+filename,
    config={'lr': lr,
            'h_dim': h_dim,
            'z_dim': z_dim,
            'H': H,
            'a_dim': a_dim,
            'state_dim': state_dim,
            'l2_reg': wd,
            'beta': beta,
            'env_name': env_name,
            'a_dist': a_dist,
            'filename': filename,
            'encoder_type': encoder_type,
            'state_decoder_type': state_decoder_type,
            'policy_decoder_type': policy_decoder_type,
            'per_element_sigma': per_element_sigma,
            'conditional_prior': conditional_prior,
            'train_diffusion_prior': train_diffusion_prior,
            'test_split': test_split,
            'separate_test_trajectories': args.separate_test_trajectories,
            'get_rewards': args.get_rewards,
            'normalize_latent': args.normalize_latent,
            'append_goals': args.append_goals,
            'use_contrastive': use_contrastive,
            'num_categorical_interval': num_categorical_interval,
            'contrastive_ratio': contrastive_ratio
            }
)

inputs_train = torch.cat([obs_chunks_train, action_chunks_train], dim=-1)
if test_split > 0.0:
    inputs_test = torch.cat([obs_chunks_test,  action_chunks_test], dim=-1)

train_loader = DataLoader(
    inputs_train,
    batch_size=batch_size,
    num_workers=4,
    shuffle=True)
if test_split > 0.0:
    test_loader = DataLoader(
        inputs_test,
        batch_size=batch_size,
        num_workers=4)

min_test_loss = 10**10
min_test_s_T_loss = 10**10
min_test_a_loss = 10**10
min_train_loss = 10**10

for i in range(n_epochs):
    if test_split > 0.0:
        test_loss, test_s_T_loss, test_a_loss, test_kl_loss, test_diffusion_loss,test_contrastive_loss = test(model, test_state_decoder=i > start_training_state_decoder_after)

        print("--------TEST---------")

        print('test_loss: ', test_loss)
        print('test_s_T_loss: ', test_s_T_loss)
        print('test_a_loss: ', test_a_loss)
        print('test_kl_loss: ', test_kl_loss)
        
        if test_diffusion_loss is not None:
            print('test_diffusion_loss ', test_diffusion_loss)
    
        if use_contrastive:
            print('test_kl_loss: ', test_kl_loss)
            
        print(i)
        wandb.log({"train_skill/test_loss": test_loss.item()})
        wandb.log({"train_skill/test_s_T_loss": test_s_T_loss.item()})
        wandb.log({"train_skill/test_a_loss": test_a_loss.item()})
        wandb.log({"train_skill/test_kl_loss": test_kl_loss.item()})
        wandb.log({"train_skill/test_contrastive_loss": test_contrastive_loss.item()})
        if test_diffusion_loss is not None:
            wandb.log({"train_skill/test_diffusion_loss": test_diffusion_loss.item()})

        if test_loss < min_test_loss:
            min_test_loss = test_loss
            checkpoint_path = os.path.join(checkpoint_dir, filename+'_best.pth')
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
        if test_s_T_loss < min_test_s_T_loss:
            min_test_s_T_loss = test_s_T_loss

            checkpoint_path = os.path.join(checkpoint_dir, filename+'_best_sT.pth')
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
        if test_a_loss < min_test_a_loss:
            min_test_a_loss = test_a_loss

            checkpoint_path = os.path.join(checkpoint_dir, filename+'_best_a.pth')
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)

    loss = train(model, optimizer, train_state_decoder=i > start_training_state_decoder_after,epoch=i)
    
    if i % 20 == 0 or i == n_epochs-1:
        checkpoint_path = os.path.join(checkpoint_dir, filename+'_'+str(i)+'_'+'.pth')
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)

    elif loss < min_train_loss:
        min_train_loss = loss
        checkpoint_path = os.path.join(checkpoint_dir, filename+'_best.pth')
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
         
    print("--------TRAIN---------")

    print('Loss: ', loss)
    print(i)
    wandb.log({"train_skill/mean train loss": loss.item(), "epoch": i})

