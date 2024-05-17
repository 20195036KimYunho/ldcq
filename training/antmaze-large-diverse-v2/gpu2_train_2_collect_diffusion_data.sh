CUDA_VISIBLE_DEVICES=0 python ./collect_diffusion_data.py \
    --env antmaze-large-diverse-v2 \
    --device 'cuda' \
    --checkpoint_dir '/home/jovyan/ldcq/checkpoints/antmaze-large-30-gc' \
    --dataset_dir '/home/jovyan/ldcq/data/antmaze-large-30-skill-full'\
    --skill_model_filename 'skill_model_antmaze-large-diverse-v2_encoderType(gru)_state_dec_mlp_policy_dec_autoregressive_H_30_b_0.1_conditionalp_1_zdim_16_adist_normal_testSplit_0.0_separatetest_0_getrewards_1_appendgoals_1_60_.pth' \
    --batch_size 1024 \
    --append_goals 1 \
    --save_z_dist 0 \
    --get_rewards 1 \
    --horizon 30 \
    --stride 1 \
    --beta 0.1 \
    --a_dist normal\
    --encoder_type gru\
    --state_decoder_type mlp\
    --policy_decoder_type autoregressive\
    --per_element_sigma 1\
    --conditional_prior 1\
    --train_diffusion_prior 0\
    --h_dim 256\
    --z_dim 16
