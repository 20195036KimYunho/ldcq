<1단계 - train_skills>
nohup ./Openhpc_walker2d-medium-v2/gpu6_train_1_skill_model_contrastive.sh > ./log/walker2d-medium_gpu6_05.19/gpu6_1.log 2>&1 &
nohup ./Openhpc_walker2d-medium-v2/gpu7_train_1_skill_model_contrastive.sh > ./log/walker2d-medium_gpu7_05.19/gpu7_1.log 2>&1 &

<2단계 - collect_diffusion_data>
nohup ./Openhpc_walker2d-medium-v2/gpu6_train_2_collect_diffusion_data_contrastive.sh > ./log/walker2d-medium_gpu6_05.19/gpu6_2.log 2>&1 &
nohup ./Openhpc_walker2d-medium-v2/gpu7_train_2_collect_diffusion_data_contrastive.sh > ./log/walker2d-medium_gpu7_05.19/gpu7_2.log 2>&1 &

<3단계 - train_diffusion>
nohup ./Openhpc_walker2d-medium-v2/gpu6_train_3_diffusion_contrastive.sh > ./log/walker2d_gpu6_05.19/gpu6_3.log 2>&1 &
nohup ./Openhpc_walker2d-medium-v2/gpu7_train_3_diffusion_contrastive.sh > ./log/walker2d_gpu7_05.19/gpu7_3.log 2>&1 &

<4단계 - collect_offline_q_learning_dataset>
nohup ./Openhpc4_gpu0_train_4_collect_q_learning.sh > ./log/gpu0_05.11/gpu0_4.log 2>&1 &
nohup ./Openhpc4_gpu1_train_4_collect_q_learning.sh > ./log/gpu1_05.11/gpu1_4.log 2>&1 &

<5단계 - train_q_net>
nohup ./Openhpc4_gpu0_train_5_q_learning.sh > ./log/gpu0_05.11/gpu0_5.log 2>&1 &
nohup ./Openhpc4_gpu1_train_5_q_learning.sh > ./log/gpu1_05.11/gpu1_5.log 2>&1 &


ps -ef | grep train_skills.py
ps -ef | grep train_diffusion.py
ps aux | grep python