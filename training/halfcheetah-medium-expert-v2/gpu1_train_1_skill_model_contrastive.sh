CUDA_VISIBLE_DEVICES=1 python ./train_skills.py \
    --env maze2d-large-v1\
    --train_diffusion_prior 0\
    --z_dim 16\
    --lr 5e-5\
    --separate_test_trajectories 0\
    --test_split 0.0\
    --get_rewards 1\
    --num_epochs 100 \
    --start_training_state_decoder_after 101 \
    --normalize_latent 0\
    --policy_decoder_type autoregressive\
    --state_decoder_type mlp\
    --a_dist softmax\
    --beta 0.1\
    --conditional_prior 1\
    --checkpoint_dir '/workspace/ldcq/ldcq/checkpoints/contrastive-maze2d-large-30'\
    --dataset_dir '/workspace/ldcq/ldcq/data/contrastive-maze2d-large-30' \
    --use_contrastive 1 \
    --contrastive_ratio 0.05 \
    --num_categorical_interval 10 \
    --append_goals 0 \
    --horizon 20

