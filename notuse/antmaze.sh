#!/bin/bash

set -ex

python ./train_skills.py \
    --env antmaze-medium-diverse-v2 \
    --num_epochs 2 \
    --start_training_state_decoder_after 100 \
    --state_decoder_type "none" 

python ./collect_diffusion_data.py \
    --env antmaze-medium-diverse-v2  \
    --state_decoder_type "none" \
    --skill_model_filename "skill_model_antmaze-medium-diverse-v2_encoderType(gru)_state_dec_none_policy_dec_autoregressive_H_30_b_0.05_conditionalp_1_zdim_16_adist_normal_testSplit_0.2_separatetest_0_getrewards_1_appendgoals_0_best.pth"

python ./train_diffusion.py \
    --env antmaze-medium-diverse-v2  \
    --n_epoch 2 \
    --diffusion_steps 100 \
    --skill_model_filename "skill_model_antmaze-medium-diverse-v2_encoderType(gru)_state_dec_none_policy_dec_autoregressive_H_30_b_0.05_conditionalp_1_zdim_16_adist_normal_testSplit_0.2_separatetest_0_getrewards_1_appendgoals_0_best.pth"

python ./collect_offline_q_learning_dataset.py \
    --env antmaze-medium-diverse-v2  \
    --num_diffusion_samples 10 \
    --num_prior_samples 10 \
    --diffusion_steps 100 \
    --state_decoder_type "none" \
    --skill_model_filename "skill_model_antmaze-medium-diverse-v2_encoderType(gru)_state_dec_none_policy_dec_autoregressive_H_30_b_0.05_conditionalp_1_zdim_16_adist_normal_testSplit_0.2_separatetest_0_getrewards_1_appendgoals_0_best.pth"


python ./train_q_net.py \
    --env antmaze-medium-diverse-v2  \
    --total_prior_samples 10 \
    --n_epoch 2 \
    --diffusion_steps 100 \
    --skill_model_filename "skill_model_antmaze-medium-diverse-v2_encoderType(gru)_state_dec_none_policy_dec_autoregressive_H_30_b_0.05_conditionalp_1_zdim_16_adist_normal_testSplit_0.2_separatetest_0_getrewards_1_appendgoals_0_best.pth"


