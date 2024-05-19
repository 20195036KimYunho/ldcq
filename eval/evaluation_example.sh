CUDA_VISIBLE_DEVICES=3 python ./plan_skills_diffusion_franka.py \
    --env halfcheetah-expert-v2 \
    --device 'cuda'\
    --num_evals 100\
    --num_parallel_envs 1\
    --checkpoint_dir '/workspace/ldcq/ldcq/checkpoints/contrastive-halfcheetah-medium-expert-20'\
    --q_checkpoint_dir '/workspace/ldcq/ldcq/q_checkpoints/halfcheetah-medium-expert-20-none-200'\
    --q_checkpoint_steps 250\
    --dataset_dir '/workspace/ldcq/ldcq/data'\
    --skill_model_filename "halfcheetah-medium-expert-v2_H_20_adist_softmax_use_contrastive_1num_categorical_interval10contrastive_ratio0.1_getrewards_1_appendgoals_0_best.pth" \
    --append_goals 0\
    --policy diffusion_prior\
    --num_diffusion_samples 300\
    --diffusion_steps 100\
    --cfg_weight 0.0\
    --extra_steps 5\
    --predict_noise 0\
    --exec_horizon 20\
    --beta 0.1\
    --a_dist softmax\
    --encoder_type gru\
    --state_decoder_type mlp \
    --policy_decoder_type autoregressive\
    --per_element_sigma 1\
    --conditional_prior 1\
    --h_dim 256\
    --z_dim 16\
    --horizon 20\
    --render 0\
    --diffusion_checkpoint 100\
    --use_contrastive 1 \
    --contrastive_ratio 0.1 \
    --num_categorical_interval 10 \
    --margin 0.0\
    --scale 30

:<<"OPTIONS"
pyhton 파일 이름도 맞출것
-locomotion : plan_skills_diffusion_franka.py
-antmaze : plan_skills_diffusion.py
-maze2d : plan_skills_diffusion_maze2d.py
OPTIONS