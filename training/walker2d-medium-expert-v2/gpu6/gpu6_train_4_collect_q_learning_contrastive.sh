CUDA_VISIBLE_DEVICES=6 python ./collect_offline_q_learning_dataset.py \
    --env walker2d-medium-expert-v2  \
    --device 'cuda' \
    --checkpoint_dir '/workspace/ldcq/ldcq/checkpoints/contrastive-walker2d-medium-expert-v2-10'\
    --dataset_dir '/workspace/ldcq/ldcq/data/contrastive-walker2d-medium-expert-v2-10' \
    --skill_model_filename 'walker2d-medium-expert-v2_H_10_adist_softmax_use_contrastive_1_num_categorical_interval_10_contrastive_ratio_0.05_getrewards_1_appendgoals_0_best.pth' \
    --batch_size 128\
    --append_goals 0 \
    --save_z_dist 1 \
    --cum_rewards 0 \
    --do_diffusion 1 \
    --num_diffusion_samples 300 \
    --num_prior_samples 300 \
    --diffusion_steps 100 \
    --cfg_weight 0.0 \
    --extra_steps 5\
    --predict_noise 0 \
    --gamma 0.995\
    --horizon 10\
    --stride 1\
    --beta 0.3\
    --a_dist softmax\
    --encoder_type "gru"\
    --state_decoder_type mlp\
    --policy_decoder_type "autoregressive"\
    --per_element_sigma 1\
    --conditional_prior 1\
    --h_dim 256\
    --z_dim 16\
    --diffusion_checkpoint 300\
    --use_contrastive 1\
    --contrastive_ratio 0.05 \
    --num_categorical_interval 10 \
    --append_goals 0   
