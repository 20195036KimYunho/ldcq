CUDA_VISIBLE_DEVICES=2 python ./train_q_net.py \
    --env antmaze-large-diverse-v2 \
    --device 'cuda' \
    --n_epoch 100\
    --lr 5e-4\
    --batch_size 128\
    --net_type 'unet'\
    --n_hidden 512\
    --test_split 0.0\
    --sample_z 1\
    --per_buffer 1\
    --sample_max_latents 1\
    --total_prior_samples 500\
    --gamma 0.995\
    --alpha 0.7\
    --horizon 20\
    --checkpoint_dir '/workspace/ldcq/ldcq/checkpoints/antmaze-large-20-naive'\
    --q_checkpoint_dir '/workspace/ldcq/ldcq/q_checkpoints/antmaze-large-20-naive'\
    --dataset_dir '/workspace/ldcq/ldcq/data'\
    --skill_model_filename 'skill_model_antmaze-large-diverse-v2_encoderType(gru)_state_dec_mlp_policy_dec_autoregressive_H_20_b_0.1_conditionalp_1_zdim_16_adist_normal_testSplit_0.0_separatetest_0_getrewards_1_appendgoals_0_best.pth' \
    --do_diffusion 1 \
    --drop_prob 0.0 \
    --diffusion_steps 100\
    --cfg_weight 0.0\
    --predict_noise 0
