#!/bin/bash

echo "===== train skills -- ldcq ====="
python ./train_skills.py --env halfcheetah-expert-v2 --num_epochs 3 --start_training_state_decoder_after 100 --state_decoder_type "none"

echo "===== collect diffusion data -- ldcq ====="
python ./collect_diffusion_data.py --env halfcheetah-expert-v2 --skill_model_filename "skill_model_halfcheetah-expert-v2_encoderType(gru)_state_dec_none_policy_dec_autoregressive_H_30_b_0.05_conditionalp_1_zdim_16_adist_normal_testSplit_0.2_separatetest_0_getrewards_1_appendgoals_0_best.pth" --state_decoder_type "none"

echo "===== train diffusion --ldcq ====="
python ./train_diffusion.py --env halfcheetah-expert-v2 --skill_model_filename "skill_model_halfcheetah-expert-v2_encoderType(gru)_state_dec_none_policy_dec_autoregressive_H_30_b_0.05_conditionalp_1_zdim_16_adist_normal_testSplit_0.2_separatetest_0_getrewards_1_appendgoals_0_best.pth" --n_epoch 3 --diffusion_steps 100

echo "===== collect offline q learning dataset --ldcq ====="
python ./collect_offline_q_learning_dataset.py --env halfcheetah-expert-v2 --skill_model_filename "skill_model_halfcheetah-expert-v2_encoderType(gru)_state_dec_none_policy_dec_autoregressive_H_30_b_0.05_conditionalp_1_zdim_16_adist_normal_testSplit_0.2_separatetest_0_getrewards_1_appendgoals_0_best.pth" --num_diffusion_samples 10 --num_prior_samples 10 --diffusion_steps 100 --state_decoder_type "none" 

echo "===== train q net --ldcq ====="
python ./train_q_net.py --env halfcheetah-expert-v2 --skill_model_filename "skill_model_halfcheetah-expert-v2_encoderType(gru)_state_dec_none_policy_dec_autoregressive_H_30_b_0.05_conditionalp_1_zdim_16_adist_normal_testSplit_0.2_separatetest_0_getrewards_1_appendgoals_0_best.pth" --total_prior_samples 10 --n_epoch 3 --diffusion_steps 100