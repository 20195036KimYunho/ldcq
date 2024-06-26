CUDA_VISIBLE_DEVICES=0 python ./train_q_net.py \
    --env maze2d-large-v1\
    --device 'cuda' \
    --n_epoch 100\
    --lr 5e-4\
    --batch_size 128\
    --net_type 'unet'\
    --n_hidden 512\
    --test_split 0.2\
    --sample_z 0\
    --per_buffer 1\
    --sample_max_latents 1\
    --total_prior_samples 300\
    --gamma 0.995\
    --alpha 0.7\
    --checkpoint_dir '/workspace/ldcq/ldcq/checkpoints/maze2d-large-30-naive'\
    --q_checkpoint_dir '/workspace/ldcq/ldcq/q_checkpoints/maze2d-large-30-naive'\
    --dataset_dir '/workspace/ldcq/ldcq/data/maze2d-large'\
    --skill_model_filename 'skill_model_maze2d-large-v1_encoderType(gru)_state_dec_none_policy_dec_autoregressive_H_30_b_0.1_conditionalp_1_zdim_16_adist_normal_testSplit_0.0_separatetest_0_getrewards_1_appendgoals_0_best.pth' \
    --do_diffusion 1 \
    --drop_prob 0.0 \
    --diffusion_steps 100\
    --cfg_weight 0.0\
    --predict_noise 0
