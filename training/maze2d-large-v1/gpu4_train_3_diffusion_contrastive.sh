CUDA_VISIBLE_DEVICES=4 python ./train_diffusion.py \
    --env maze2d-large-v1 \
    --device 'cuda' \
    --n_epoch 450 \
    --lrate 1e-4 \
    --batch_size 128 \
    --net_type 'unet' \
    --n_hidden 512 \
    --test_split 0.0 \
    --sample_z 0 \
    --checkpoint_dir '/workspace/ldcq/ldcq/checkpoints/contrastive-maze2d-large-30' \
    --dataset_dir '/workspace/ldcq/ldcq/data/contrastive-maze2d-large-30' \
    --skill_model_filename 'maze2d-large-v1_H_30_adist_softmax_use_contrastive_1_num_categorical_interval_10_contrastive_ratio_0.1_getrewards_1_appendgoals_0_best.pth'\
    --append_goals 0 \
    --drop_prob 0.1 \
    --diffusion_steps 100 \
    --cfg_weight 0.0 \
    --predict_noise 0 \
    --normalize_latent 0 \
    --schedule 'linear'\
    --use_contrastive 1 \
    --contrastive_ratio 0.1 \
    --num_categorical_interval 10 \
    --append_goals 0     
    

