CUDA_VISIBLE_DEVICES=4 python ./train_q_net.py \
    --env walker2d-medium-expert-v2 \
    --device 'cuda' \
    --n_epoch 300\
    --lr 5e-4\
    --batch_size 128\
    --net_type 'unet'\
    --n_hidden 512\
    --test_split 0.1\
    --sample_z 0\
    --per_buffer 1\
    --sample_max_latents 1\
    --total_prior_samples 500\
    --gamma 0.995\
    --alpha 0.7\
    --checkpoint_dir '/workspace/ldcq/ldcq/checkpoints'\
    --dataset_dir '/workspace/ldcq/ldcq/data'\
    --skill_model_filename "skill_model_halfcheetah-expert-v2_encoderType(gru)_state_dec_mlp_policy_dec_autoregressive_H_10_b_0.05_conditionalp_1_zdim_16_adist_normal_testSplit_0.1_separatetest_0_getrewards_1_appendgoals_0_best.pth" \
    --do_diffusion 1 \
    --drop_prob 0.0 \
    --diffusion_steps 200\
    --cfg_weight 0.0\
    --predict_noise 0
