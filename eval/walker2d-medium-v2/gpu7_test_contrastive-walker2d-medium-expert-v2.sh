CUDA_VISIBLE_DEVICES=7 python ./plan_skills_diffusion_franka.py \
    --env walker2d-medium-expert-v2 \
    --device 'cuda'\
    --num_evals 100\
    --num_parallel_envs 1\
    --checkpoint_dir '/workspace/ldcq/ldcq/checkpoints/contrastive-walker2d-medium-expert-v2-10'\
    --q_checkpoint_dir '/workspace/ldcq/ldcq/q_checkpoints'\
    --q_checkpoint_steps 194\
    --dataset_dir '/workspace/ldcq/ldcq/data/contrastive-walker2d-medium-expert-v2-10' \
    --skill_model_filename "walker2d-medium-expert-v2_H_10_adist_softmax_use_contrastive_1_num_categorical_interval_10_contrastive_ratio_0.1_getrewards_1_appendgoals_0_best.pth" \
    --append_goals 0\
    --policy diffusion_prior\
    --num_diffusion_samples 10\
    --diffusion_steps 200\
    --cfg_weight 0.0\
    --extra_steps 5\
    --predict_noise 0\
    --exec_horizon 20\
    --beta 1.0\
    --a_dist softmax\
    --encoder_type gru\
    --state_decoder_type mlp\
    --policy_decoder_type autoregressive\
    --per_element_sigma 1\
    --conditional_prior 1\
    --h_dim 256\
    --z_dim 16\
    --horizon 20\
    --render 0\


# prior / diffusion_prior
