CUDA_VISIBLE_DEVICES=2 python ./train_diffusion.py \
    --env walker2d-medium-expert-v2 \
    --device 'cuda' \
    --n_epoch 450 \
    --lrate 1e-4 \
    --batch_size 128 \
    --net_type 'unet' \
    --n_hidden 512 \
    --test_split 0.0 \
    --sample_z 0 \
    --checkpoint_dir '/home/jovyan/beomi/jaehyun/ldcq/checkpoints/gpu2_walker2d-medium-expert' \
    --dataset_dir '/home/jovyan/beomi/jaehyun/ldcq/data/gpu2_walker2d-medium-expert' \
    --skill_model_filename 'halfcheetah-medium-expert-v2_H_20_adist_softmax_use_contrastive_1_num_categorical_interval_10_contrastive_ratio_0.05_getrewards_1_appendgoals_0_best.pth'\
    --append_goals 0 \
    --drop_prob 0.1 \
    --diffusion_steps 100 \
    --cfg_weight 0.0 \
    --predict_noise 0 \
    --normalize_latent 0 \
    --schedule 'linear' \
    --num_categorical_interval 10 \
    --append_goals 0