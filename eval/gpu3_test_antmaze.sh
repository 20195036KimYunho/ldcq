CUDA_VISIBLE_DEVICES=2 python ./plan_skills_diffusion.py \
    --env antmaze-medium-diverse-v2\
    --device 'cuda'\
    --num_evals 100\
    --num_parallel_envs 1\
    --checkpoint_dir '/workspace/ldcq/ldcq/checkpoints/antmaze-medium-20-naive'\
    --q_checkpoint_dir '/workspace/ldcq/ldcq/q_checkpoints/antmaze-medium-20-naive'\
    --q_checkpoint_steps 25\
    --dataset_dir '/workspace/ldcq/ldcq/data'\
    --skill_model_filename "skill_model_antmaze-medium-diverse-v2_encoderType(gru)_state_dec_none_policy_dec_autoregressive_H_20_b_0.1_conditionalp_1_zdim_16_adist_normal_testSplit_0.0_separatetest_0_getrewards_1_appendgoals_0_best.pth" \
    --append_goals 0\
    --policy q\
    --num_diffusion_samples 500\
    --diffusion_steps 100\
    --diffusion_checkpoint best\
    --cfg_weight 0.0\
    --planning_depth 2\
    --extra_steps 10\
    --predict_noise 0\
    --exec_horizon 10\
    --beta 0.1\
    --a_dist normal\
    --encoder_type gru\
    --state_decoder_type none \
    --policy_decoder_type autoregressive\
    --per_element_sigma 1\
    --conditional_prior 1\
    --h_dim 256\
    --z_dim 16\
    --horizon 10\
    --render 0\
    --visualize 0