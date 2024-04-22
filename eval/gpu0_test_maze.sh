CUDA_VISIBLE_DEVICES=0 python ./plan_skills_diffusion_maze2d.py \
    --env maze2d-large-v1 \
    --device 'cuda'\
    --num_evals 100\
    --num_parallel_envs 1\
    --checkpoint_dir \
    --q_checkpoint_dir \
    --q_checkpoint_steps 0\
    --dataset_dir \
    --skill_model_filename "skill_model_halfcheetah-expert-v2_encoderType(gru)_state_dec_none_policy_dec_autoregressive_H_30_b_0.05_conditionalp_1_zdim_16_adist_normal_testSplit_0.2_separatetest_0_getrewards_1_appendgoals_0_best.pth" \
    --append_goals 0\
    --policy q\
    --num_diffusion_samples 10\
    --difusion_steps 200\
    --cfg_weight 0.0\
    --planning_depth 5\
    --extra_steps 5\
    --predict_noise 0\
    --exec_horizon 10\
    --beta 1.0\
    --a_dist normal\
    --encoder_type gru\
    --state_decoder_type none \
    --policy_decoder_type autoregressive\
    --per_element_sigma 1\
    --conditional_prior 1\
    --h_dim 256\
    --z_dim 16\
    --horizon 30\
    --render 1\
    --visualize 0\