CUDA_VISIBLE_DEVICES=5 python ./train_skills.py \
    --env walker2d-medium-expert-v2\
    --beta 0.1\
    --conditional_prior 1\
    --z_dim 16\
    --lr 5e-5\
    --policy_decoder_type autoregressive\
    --state_decoder_type none\
    --a_dist normal\
    --horizon 10\
    --separate_test_trajectories 0\
    --test_split 0.1\
    --get_rewards 1\
    --num_epochs 100 \
    --start_training_state_decoder_after 1000 \
    --normalize_latent 0\
    --append_goals 0


# horizon 10 