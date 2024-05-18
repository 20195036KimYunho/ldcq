<1단계 - train_skills>
nohup ./Openhpc_walker2d-medium-expert-v2/gpu0_train_1_skill_model_contrastive.sh > ./log/walker2d_gpu0_05.17/gpu0_1.log 2>&1 &
nohup ./Openhpc_walker2d-medium-expert-v2/gpu1_train_1_skill_model_contrastive.sh > ./log/walker2d_gpu1_05.17/gpu1_1.log 2>&1 &
nohup ./Openhpc_walker2d-medium-expert-v2/gpu2_train_1_skill_model_contrastive.sh > ./log/walker2d_gpu2_05.17/gpu2_1.log 2>&1 &
nohup ./Openhpc_walker2d-medium-expert-v2/gpu3_train_1_skill_model_contrastive.sh > ./log/walker2d_gpu3_05.17/gpu3_1.log 2>&1 &

<2단계 - collect_diffusion_data>
nohup ./Openhpc_walker2d-medium-expert-v2/gpu0_train_2_collect_diffusion_data_contrastive.sh > ./log/walker2d_gpu0_05.17/gpu0_2.log 2>&1 &
nohup ./Openhpc_walker2d-medium-expert-v2/gpu1_train_2_collect_diffusion_data_contrastive.sh > ./log/walker2d_gpu1_05.17/gpu1_2.log 2>&1 &
nohup ./Openhpc_walker2d-medium-expert-v2/gpu2_train_2_collect_diffusion_data_contrastive.sh > ./log/walker2d_gpu2_05.17/gpu2_2.log 2>&1 &
nohup ./Openhpc_walker2d-medium-expert-v2/gpu3_train_2_collect_diffusion_data_contrastive.sh > ./log/walker2d_gpu3_05.17/gpu3_2.log 2>&1 &

<3단계 - train_diffusion>
nohup ./Openhpc4_gpu0_train_3_diffusion.sh > ./log/gpu0_05.11/gpu0_3.log 2>&1 &
nohup ./Openhpc4_gpu1_train_3_diffusion.sh > ./log/gpu1_05.11/gpu1_3.log 2>&1 &

<4단계 - collect_offline_q_learning_dataset>
nohup ./Openhpc4_gpu0_train_4_collect_q_learning.sh > ./log/gpu0_05.11/gpu0_4.log 2>&1 &
nohup ./Openhpc4_gpu1_train_4_collect_q_learning.sh > ./log/gpu1_05.11/gpu1_4.log 2>&1 &

<5단계 - train_q_net>
nohup ./Openhpc4_gpu0_train_5_q_learning.sh > ./log/gpu0_05.11/gpu0_5.log 2>&1 &
nohup ./Openhpc4_gpu1_train_5_q_learning.sh > ./log/gpu1_05.11/gpu1_5.log 2>&1 &


ps -ef | grep train_skills.py
ps -ef | grep train_diffusion.py
ps aux | grep python