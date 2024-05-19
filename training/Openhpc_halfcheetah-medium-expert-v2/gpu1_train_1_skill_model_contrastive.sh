CUDA_VISIBLE_DEVICES=1 python ./train_skills.py \
    --env halfcheetah-medium-expert-v2\
    --train_diffusion_prior 0 \
    --z_dim 16 \
    --lr 5e-5 \
    --separate_test_trajectories 0 \
    --test_split 0.0 \
    --get_rewards 1 \
    --num_epochs 200 \
    --start_training_state_decoder_after 201 \
    --normalize_latent 0 \
    --policy_decoder_type autoregressive \
    --state_decoder_type mlp \
    --a_dist softmax \
    --beta 0.01 \
    --conditional_prior 1 \
    --checkpoint_dir '/home/jovyan/beomi/jaehyun/ldcq/checkpoints/gpu1_halfcheetah-medium-expert'\
    --dataset_dir '/home/jovyan/beomi/jaehyun/ldcq/data/gpu1_halfcheetah-medium-expert' \
    --use_contrastive 1 \
    --contrastive_ratio 0.4 \
    --num_categorical_interval 10 \
    --append_goals 0 \
    --horizon 10 \
    --margin 0.4 \
    --scale 30
