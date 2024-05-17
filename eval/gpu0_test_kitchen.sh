CUDA_VISIBLE_DEVICES=0 python ./plan_skills_diffusion_franka.py \
    --env kitchen-complete-v0\
    --device 'cuda'\
    --num_evals 100\
    --num_parallel_envs 1\
    --checkpoint_dir '/workspace/ldcq/ldcq/checkpoints/contrastive-kitchen-complete'\
    --q_checkpoint_dir '/workspace/ldcq/ldcq/q_checkpoints/contrastive-kitchen-complete'\
    --q_checkpoint_steps 250\
    --dataset_dir '/workspace/ldcq/ldcq/data'\
    --skill_model_filename "kitchen-complete-v0_H_20_adist_softmax_use_contrastive_1_num_categorical_interval_10_contrastive_ratio_0.5_margin_0.05_scale_10_getrewards_1_appendgoals_0_best.pth" \
    --append_goals 0\
    --policy prior\
    --num_diffusion_samples 300\
    --diffusion_steps 100\
    --cfg_weight 0.0\
    --extra_steps 5\
    --predict_noise 0\
    --exec_horizon 20\
    --beta 0.1\
    --a_dist softmax\
    --encoder_type gru\
    --state_decoder_type mlp \
    --policy_decoder_type autoregressive\
    --per_element_sigma 1\
    --conditional_prior 1\
    --h_dim 256\
    --z_dim 16\
    --horizon 20\
    --render 0\
    --diffusion_checkpoint 400
