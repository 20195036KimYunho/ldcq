CUDA_VISIBLE_DEVICES=0 python ./train_diffusion.py \
    --env antmaze-large-diverse-v2_1 \
    --device 'cuda' \
    --n_epoch 450 \
    --lrate 1e-4 \
    --batch_size 128 \
    --net_type 'unet' \
    --n_hidden 512 \
    --test_split 0.0 \
    --sample_z 0 \
    --checkpoint_dir '/home/jovyan/ldcq/checkpoints/antmaze-medium-30-naive-1' \
    --dataset_dir '/home/jovyan/ldcq/data/test'\
    --skill_model_filename 'skill_model_antmaze-large-diverse-v2_encoderType(gru)_state_dec_mlp_policy_dec_autoregressive_H_30_b_0.1_conditionalp_1_zdim_16_adist_normal_testSplit_0.0_separatetest_0_getrewards_1_appendgoals_1_60_.pth' \
    --append_goals 1 \
    --drop_prob 0.1 \
    --diffusion_steps 100 \
    --cfg_weight 0.0 \
    --predict_noise 0 \
    --normalize_latent 0 \
    --schedule 'linear'
    