CUDA_VISIBLE_DEVICES=6 python ./train_diffusion.py \
    --env halfcheetah-medium-v2  \
    --device 'cuda' \
    --n_epoch 450 \
    --lrate 1e-4 \
    --batch_size 128 \
    --net_type 'unet' \
    --n_hidden 512 \
    --test_split 0.0 \
    --sample_z 0 \
    --checkpoint_dir '/workspace/ldcq/ldcq/checkpoints' \
    --dataset_dir '/workspace/ldcq/ldcq/data' \
    --skill_model_filename "skill_model_halfcheetah-medium-v2_encoderType(gru)_state_dec_none_policy_dec_autoregressive_H_20_b_0.1_conditionalp_1_zdim_16_adist_normal_testSplit_0.0_separatetest_0_getrewards_1_appendgoals_0_best.pth" \
    --append_goals 0 \
    --drop_prob 0.1 \
    --diffusion_steps 100 \
    --cfg_weight 0.0 \
    --predict_noise 0 \
    --normalize_latent 0 \
    --schedule 'linear'
    
