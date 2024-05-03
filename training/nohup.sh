사용법
log 폴더 밑에 gpu 번호_ 날짜로 폴더 생성 후 로그파일의 날짜 변경 후 라인 복붙으로 실행
gpu별 shell 파일 생성하고 각 environment에 맞게 수정 후 실행
각 gpu 번호마다 하나씩 environment 할당 -> notion 표에 몇번 gpu인지 확실히 할것.
'watch -d -n 1 nvidia-smi'로 spu 사용량 확인하고 사용할 것 이미 무언가 돌아가고 있다면 꼭 확인해볼 것

<1단계 - train_skills>
nohup maze2d-large-v1/gpu0_train_1_skill_model.sh > ./log/gpu0_04.24/gpu0_1_04.24.log 2>&1 &
nohup antmaze-medium-diverse-v2/gpu1_train_1_skill_model.sh > ./log/gpu1_04.24/gpu1_1_04.24.log 2>&1 &
nohup antmaze-large-diverse-v2/gpu2_train_1_skill_model.sh > ./log/gpu2_04.24/gpu2_1_04.24.log 2>&1 &
nohup halfcheetah-medium-expert-v2/gpu3_train_1_skill_model.sh > ./log/gpu3_04.24/gpu3_1_04.24.log 2>&1 &
nohup ./gpu4_train_1_skill_model.sh > ./log/gpu4_04.11/gpu4_1_04.11.log 2>&1 &
nohup ./gpu5_train_1_skill_model.sh > ./log/gpu5_04.11/gpu5_1_04.11.log 2>&1 &
nohup ./gpu6_train_1_skill_model.sh > ./log/gpu6_04.11/gpu6_1_04.11.log 2>&1 &
nohup ./gpu7_train_1_skill_model.sh > ./log/gpu7_04.11/gpu7_1_04.11.log 2>&1 &

<2단계 - collect_diffusion_data>
nohup maze2d-large-v1/gpu0_train_2_collect_diffusion_data.sh > ./log/gpu0_04.25/gpu0_2_04.25.log 2>&1 &
nohup antmaze-medium-diverse-v2/gpu1_train_2_collect_diffusion_data.sh > ./log/gpu1_04.25/gpu1_2_04.25.log 2>&1 &
nohup antmaze-large-diverse-v2/gpu2_train_2_collect_diffusion_data.sh > ./log/gpu2_04.25/gpu2_2_04.25.log 2>&1 &
nohup halfcheetah-medium-expert-v2/gpu3_train_2_collect_diffusion_data.sh > ./log/gpu3_04.25/gpu3_2_04.25.log 2>&1 &
nohup ./gpu4_train_2_collect_diffusion_data.sh > ./log/gpu4_04.11/gpu4_2_04.11.log 2>&1 &
nohup ./gpu5_train_2_collect_diffusion_data.sh > ./log/gpu5_04.11/gpu5_2_04.11.log 2>&1 &
nohup ./gpu6_train_2_collect_diffusion_data.sh > ./log/gpu6_04.11/gpu6_2_04.11.log 2>&1 &
nohup ./gpu7_train_2_collect_diffusion_data.sh > ./log/gpu7_04.11/gpu7_2_04.11.log 2>&1 &

<3단계 - train_diffusion>
nohup maze2d-large-v1/gpu0_train_3_diffusion.sh > ./log/gpu0_04.25/gpu0_3_04.25.log 2>&1 &
nohup antmaze-medium-diverse-v2/gpu1_train_3_diffusion.sh > ./log/gpu1_04.25/gpu1_3_04.25.log 2>&1 &
nohup antmaze-large-diverse-v2/gpu2_train_3_diffusion.sh > ./log/gpu2_04.25/gpu2_3_04.25.log 2>&1 &
nohup halfcheetah-medium-expert-v2/gpu3_train_3_diffusion.sh > ./log/gpu3_04.25/gpu3_3_04.25.log 2>&1 &
nohup ./gpu4_train_3_diffusion.sh > ./log/gpu4_04.24/gpu4_3_04.11.log 2>&1 &
nohup ./gpu5_train_3_diffusion.sh > ./log/gpu5_04.24/gpu5_3_04.11.log 2>&1 &
nohup ./gpu6_train_3_diffusion.sh > ./log/gpu6_04.24/gpu6_3_04.11.log 2>&1 &
nohup ./gpu7_train_3_diffusion.sh > ./log/gpu7_04.24/gpu7_3_04.11.log 2>&1 &

<4단계 - collect_offline_q_learning_dataset>
nohup maze2d-large-v1/gpu0_train_4_collect_q_learning.sh > ./log/gpu0_04.24/gpu0_4_04.24.log 2>&1 &
nohup antmaze-medium-diverse-v2/gpu1_train_4_collect_q_learning.sh > ./log/gpu1_04.29/gpu1_4_04.29.log 2>&1 &
nohup antmaze-large-diverse-v2/gpu2_train_4_collect_q_learning.sh > ./log/gpu2_04.27/gpu2_4_04.27.log 2>&1 &
nohup halfcheetah-medium-expert-v2/gpu3_train_4_collect_q_learning.sh > ./log/gpu3_04.24/gpu3_4_04.16.log 2>&1 &
nohup ./gpu4_train_4_collect_q_learning.sh > ./log/gpu4_04.24/gpu4_4_04.16.log 2>&1 &
nohup ./gpu5_train_4_collect_q_learning.sh > ./log/gpu5_04.24/gpu5_4_04.16.log 2>&1 &
nohup ./gpu6_train_4_collect_q_learning.sh > ./log/gpu6_04.24/gpu6_4_04.16.log 2>&1 &
nohup ./gpu7_train_4_collect_q_learning.sh > ./log/gpu7_04.24/gpu7_4_04.16.log 2>&1 &

<5단계 - collect_offline_q_learning_dataset>

nohup maze2d-large-v1/gpu0_train_5_q_learning.sh > ./log/gpu0_04.26/gpu0_5_04.26.log 2>&1 &
nohup antmaze-medium-diverse-v2/gpu1_train_5_q_learning.sh > ./log/gpu1_04.26/gpu1_5_04.26.log 2>&1 &
nohup antmaze-large-diverse-v2/gpu2_train_5_q_learning.sh > ./log/gpu2_05.01/gpu2_5_05.01.log 2>&1 &
nohup halfcheetah-medium-expert-v2/gpu3_train_5_q_learning.sh > ./log/gpu3_04.16/gpu3_5_04.16.log 2>&1 &
nohup antmaze-large-diverse-v2/gpu2_train_5_q_learning_test.sh > ./log/gpu2_04.29/gpu2_5_04.29.log 2>&1 &
nohup ./gpu5_train_5_q_learning.sh > ./log/gpu5_04.16_origin_loss/gpu5_5_04.16.log 2>&1 &
nohup ./gpu6_train_5_q_learning.sh > ./log/gpu6_04.16_origin_loss/gpu6_5_04.16.log 2>&1 &
nohup ./gpu7_train_5_q_learning.sh > ./log/gpu7_04.16_origin_loss/gpu7_5_04.16.log 2>&1 &

아래 명령어는 백그라운드로 실행된 프로세스를 확인하는 명령어, 불필요하게 안 꺼진거 있으면 끄기

ps -ef | grep train_skills.py
ps -ef | grep collect_diffusion_data.py
ps -ef | grep train_diffusion.py
ps -ef | grep collect_offline_q_learning_dataset.py
ps -ef | grep train_q_net.py
