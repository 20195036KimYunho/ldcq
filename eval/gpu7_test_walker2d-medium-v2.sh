CUDA_VISIBLE_DEVICES=7 python ./plan_skills_diffusion_franka.py \
    --env walker2d-medium-v2 \
    --device 'cuda'\
    --num_evals 100\
    --num_parallel_envs 1\
    --checkpoint_dir '/workspace/ldcq/ldcq/checkpoints'\
    --q_checkpoint_dir '/workspace/ldcq/ldcq/q_checkpoints'\
    --q_checkpoint_steps 98\
    --dataset_dir '/workspace/ldcq/ldcq/data'\
    --skill_model_filename "skill_model_walker2d-medium-v2_encoderType(gru)_state_dec_mlp_policy_dec_autoregressive_H_10_b_0.1_conditionalp_1_zdim_16_adist_normal_testSplit_0.0_separatetest_0_getrewards_1_appendgoals_0_best_diffusion_prior_best.pt" \
    --append_goals 0\
    --policy q\
    --num_diffusion_samples 10\
    --diffusion_steps 200\
    --cfg_weight 0.0\
    --extra_steps 5\
    --predict_noise 0\
    --exec_horizon 20\
    --beta 1.0\
    --a_dist normal\
    --encoder_type gru\
    --state_decoder_type none \
    --policy_decoder_type autoregressive\
    --per_element_sigma 1\
    --conditional_prior 1\
    --h_dim 256\
    --z_dim 16\
    --horizon 20\
    --render 0\
