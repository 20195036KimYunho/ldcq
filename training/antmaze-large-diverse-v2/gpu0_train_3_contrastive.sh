CUDA_VISIBLE_DEVICES=0 python ./train_diffusion.py \
    --env antmaze-large-diverse-v2 \
    --device 'cuda' \
    --n_epoch 450 \
    --lrate 1e-4 \
    --batch_size 128 \
    --net_type 'unet' \
    --n_hidden 512 \
    --test_split 0.0 \
    --sample_z 0 \
    --checkpoint_dir '/home/jovyan/ldcq/checkpoints/contrastive-antmaze-large-20'\
    --dataset_dir '/home/jovyan/ldcq/data/contrastive-antmaze-large-20' \
    --skill_model_filename 'antmaze-large-diverse-v2_H_20_adist_softmax_use_contrastive_0_num_categorical_interval_10_contrastive_ratio_0.0_getrewards_1_appendgoals_0_best.pth'\
    --append_goals 0 \
    --drop_prob 0.1 \
    --diffusion_steps 100 \
    --cfg_weight 0.0 \
    --predict_noise 0 \
    --normalize_latent 0 \
    --schedule 'linear'\
    --use_contrastive 0 \
    --contrastive_ratio 0.00 \
    --num_categorical_interval 10 \
    --append_goals 0     
    
