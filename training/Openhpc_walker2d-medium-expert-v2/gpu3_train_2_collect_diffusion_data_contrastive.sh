CUDA_VISIBLE_DEVICES=3 python ./collect_diffusion_data.py \
    --env walker2d-medium-expert-v2 \
    --device 'cuda' \
    --skill_model_filename 'halfcheetah-medium-expert-v2_H_20_adist_softmax_use_contrastive_1_num_categorical_interval_10_contrastive_ratio_0.05_getrewards_1_appendgoals_0_best.pth'\
    --batch_size 1024 \
    --append_goals 0 \
    --save_z_dist 0 \
    --get_rewards 1 \
    --stride 1 \
    --a_dist softmax \
    --encoder_type gru \
    --state_decoder_type mlp \
    --policy_decoder_type autoregressive \
    --per_element_sigma 1 \
    --conditional_prior 1 \
    --h_dim 256 \
    --z_dim 16 \
    --beta 0.01 \
    --checkpoint_dir '/home/jovyan/beomi/jaehyun/ldcq/checkpoints/gpu3_walker2d-medium-expert' \
    --dataset_dir '/home/jovyan/beomi/jaehyun/ldcq/data/gpu3_walker2d-medium-expert' \
    --use_contrastive 1 \
    --contrastive_ratio 0.4 \
    --num_categorical_interval 10 \
    --append_goals 0 \
    --horizon 5 \
    --margin 0.4 \
    --scale 30