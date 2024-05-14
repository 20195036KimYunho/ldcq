CUDA_VISIBLE_DEVICES=0 python ./collect_diffusion_data.py \
    --env halfcheetah-medium-expert-v2 \
    --device 'cuda' \
    --skill_model_filename 'halfcheetah-medium-expert-v2_H_20_adist_softmax_use_contrastive_1num_categorical_interval10contrastive_ratio0.1_getrewards_1_appendgoals_0_best.pth' \
    --batch_size 1024 \
    --append_goals 0 \
    --save_z_dist 0 \
    --get_rewards 1 \
    --horizon 20 \
    --stride 1 \
    --beta 0.1 \
    --a_dist softmax\
    --encoder_type gru\
    --state_decoder_type mlp\
    --policy_decoder_type autoregressive\
    --per_element_sigma 1\
    --conditional_prior 1\
    --h_dim 256\
    --z_dim 16\
    --checkpoint_dir '/workspace/ldcq/ldcq/checkpoints/contrastive-halfcheetah-medium-expert-20'\
    --dataset_dir '/workspace/ldcq/ldcq/data/contrastive-halfcheetah-medium-expert-20' \
    --use_contrastive 1 \
    --contrastive_ratio 0.1 \
    --num_categorical_interval 10 \
    --append_goals 0 \
    --horizon 20
