CUDA_VISIBLE_DEVICES=0 python ../collect_offline_q_learning_dataset.py \
    --env antmaze-medium-diverse-v2  \
    --device 'cuda' \
    --checkpoint_dir '/home/sundong/ldcq/ldcq/checkpoints' \
    --skill_model_filename "skill_model_halfcheetah-expert-v2_encoderType(gru)_state_dec_none_policy_dec_autoregressive_H_30_b_0.05_conditionalp_1_zdim_16_adist_normal_testSplit_0.2_separatetest_0_getrewards_1_appendgoals_0_best.pth" \
    --batch_size 128\
    --append_goals 0 \
    --save_z_dist 1 \
    --cum_rewards 0 \
    --do_diffusion 1 \
    --num_diffusion_samples 500 \
    --num_prior_samples 500 \
    --diffusion_steps 200 \
    --cfg_weight 0.0 \
    --extra_steps 5\
    --predict_noise 0 \
    --gamma 0.995\
    --horizon 30\
    --stride 1\
    --beta 0.3\
    --a_dist "normal"\
    --encoder_type "gru"\
    --state_decoder_type "none"\
    --policy_decoder_type "autoregressive"\
    --per_element_sigma 1\
    --conditional_prior 1\
    --h_dim 256\
    --z_dim 16
