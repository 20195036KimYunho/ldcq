# Latent Diffusion Constrained Q-Learning (LDCQ)

Training and visualizing of diffusion models from [Reasoning with Latent Diffusion in Offline Reinforcement Learning (NeurIPS 2023)](https://arxiv.org/abs/2309.06599).

## Installation

### Using CUDA=12.1, pytorch=2.1.2 torchvision=0.16.2. Install cuDNN according to your own environment

```
sudo apt-get update
sudo apt-get upgrade

conda create -n ldcq python=3.8
conda activate ldcq

conda install pytorch==2.1.2 torchvision==0.16.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Install mujoco key & engine

```
sudo apt-get update
sudo apt-get upgrade

sudo apt-get install liblzma-dev libffi-dev mpich unzip libosmesa6-dev libgl1-mesa-glx libglfw3 libosmesa6-dev
sudo apt-get install -y build-essential
sudo apt-get install freeglut3-dev libglu1-mesa-dev mesa-common-dev

wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xvf mujoco210-linux-x86_64.tar.gz
mkdir .mujoco
mv mujoco210 .mujoco

wget https://www.roboti.us/file/mjkey.txt
mv mjkey.txt .mujoco

wget https://www.roboti.us/download/mjpro131_linux.zip
unzip mjpro131_linux.zip
mv mjpro131 .mujoco
```

### Insert PATH in ~/.bashrc file

```
vim ~/.bashrc

### ====== 아래 내용 파일 내용에 추가 ====== ###
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/[유저이름]/.mujoco/mujoco210/bin
export MUJOCO_PY_MJKEY_PATH=~/.mujoco/mjkey.txt
export MUJOCO_PY_MJPRO_PATH=~/.mujoco/mjpro131
### ======================================== ###

source ~/.bashrc
conda activate ldcq
```

### Install requirements.txt

```
pip install --upgrade pip
pip install -r requirement.txt
```

## Requirements (pip)

```
tqdm
matplotlib
comet_ml
ipdb

mujoco-py==2.1.2.14
gym==0.12.1
PyOpenGL==3.1.1a1
patchelf
```

additionally,
```
pip install "cython<3"
pip install "protobuf<3.20"
```

## Dataset

### D4RL

When you want to use D4RL dataset used in original LDCQ, it is recommended to install d4rl using the method from the original d4rl github https://github.com/Farama-Foundation/D4RL :

```
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
```

when using `pip install d4rl`, there may be some issues about mjrl, environment.

```
추가하기!
```

## Training

Training Code for halfcheetah-expert Environment

1. Training skill with:

```
python ./train_skills.py --env halfcheetah-expert-v2 --num_epochs 3 --start_training_state_decoder_after 100 --state_decoder_type "none"
```

2. Collect data to train diffusion model with:

```
python ./collect_diffusion_data.py \
    --env halfcheetah-expert-v2 \
    --skill_model_filename "skill_model_halfcheetah-expert-v2_encoderType(gru)_state_dec_none_policy_dec_autoregressive_H_30_b_0.05_conditionalp_1_zdim_16_adist_normal_testSplit_0.2_separatetest_0_getrewards_1_appendgoals_0_best.pth" \
    --state_decoder_type "none"
```

3. Training diffusion model with:

```
python ./train_diffusion.py \
    --env halfcheetah-expert-v2 \
    --skill_model_filename "skill_model_halfcheetah-expert-v2_encoderType(gru)_state_dec_none_policy_dec_autoregressive_H_30_b_0.05_conditionalp_1_zdim_16_adist_normal_testSplit_0.2_separatetest_0_getrewards_1_appendgoals_0_best.pth" \
    --n_epoch 5 \
    --diffusion_steps 100
```

4. Collect data to train offline Q-learning with:

```
python ./collect_offline_q_learning_dataset.py \
    --env halfcheetah-expert-v2 \
    --skill_model_filename "skill_model_halfcheetah-expert-v2_encoderType(gru)_state_dec_none_policy_dec_autoregressive_H_30_b_0.05_conditionalp_1_zdim_16_adist_normal_testSplit_0.2_separatetest_0_getrewards_1_appendgoals_0_best.pth" \
    --num_diffusion_samples 100 \
    --num_prior_samples 100 \
    --diffusion_steps 100 \
    --state_decoder_type "none"
```

5. Training Q-network with:

```
python ./train_q_net.py \
    --env halfcheetah-expert-v2 \
    --skill_model_filename "skill_model_halfcheetah-expert-v2_encoderType(gru)_state_dec_none_policy_dec_autoregressive_H_30_b_0.05_conditionalp_1_zdim_16_adist_normal_testSplit_0.2_separatetest_0_getrewards_1_appendgoals_0_best.pth" \
    --total_prior_samples 100 \
    --n_epoch 5 \
    --diffusion_steps 100
```

## Test

```
python ./plan_skills_diffusion_franka.py \
    --env halfcheetah-expert-v2 \
    --skill_model_filename "skill_model_halfcheetah-expert-v2_encoderType(gru)_state_dec_none_policy_dec_autoregressive_H_30_b_0.05_conditionalp_1_zdim_16_adist_normal_testSplit_0.2_separatetest_0_getrewards_1_appendgoals_0_best.pth" \
    --state_decoder_type none \
    --q_checkpoint_steps 18
```

## Reference

```
@inproceedings{ldcq,
  title = {Reasoning with Latent Diffusion in Offline Reinforcement Learning},
  author = {Siddarth Venkatraman, Shivesh Khaitan, Ravi Tej Akella, John Dolan, Jeff Schneider, Glen Berseth},
  booktitle = {Conference on Neural Information Processing Systems},
  year = {2023},
}
```
