#!/bin/bash
source ./vars
model=$dir_base
if [ -z "$model" ]; then
    echo "model is empty"
    exit
fi
echo 'model====>' $model

model_rl_dir=model_RL/$model/0.2;
mkdir ${model_rl_dir}
python run_srnn.py --mode=RL --batch_size=128 --model_rl_dir="${model_rl_dir}" --reward_coef_r2="{0:0.2}" --reward_coef_r_crossent="{0:0.05,1000:0.08,2000:0.1,3000:0.15,4000:0.2,5000:0.3,6000:0.4,7000:0.6,8000:0.7}" --sampling_global_step=5500

model_rl_dir=model_RL/$model/0.5;
mkdir ${model_rl_dir}
python run_srnn.py --mode=RL --batch_size=128 --model_rl_dir="${model_rl_dir}" --reward_coef_r2="{0:0.5}" --reward_coef_r_crossent="{0:0.05,1000:0.08,2000:0.1,3000:0.15,4000:0.2,5000:0.3,6000:0.4,7000:0.6,8000:0.7}" --sampling_global_step=5500

model_rl_dir=model_RL/$model/0.75;
mkdir ${model_rl_dir}
python run_srnn.py --mode=RL --batch_size=128 --model_rl_dir="${model_rl_dir}" --reward_coef_r2="{0:0.75}" --reward_coef_r_crossent="{0:0.05,1000:0.08,2000:0.1,3000:0.15,4000:0.2,5000:0.3,6000:0.4,7000:0.6,8000:0.7}" --sampling_global_step=5500

model_rl_dir=model_RL/$model/1;
mkdir ${model_rl_dir}
python run_srnn.py --mode=RL --batch_size=128 --model_rl_dir="${model_rl_dir}" --reward_coef_r2="{0:1}" --reward_coef_r_crossent="{0:0.05,1000:0.08,2000:0.1,3000:0.15,4000:0.2,5000:0.3,6000:0.4,7000:0.6,8000:0.7}" --sampling_global_step=5500 

model_rl_dir=model_RL/$model/change;
mkdir ${model_rl_dir}
python run_srnn.py --mode=RL --batch_size=128 --model_rl_dir="${model_rl_dir}" --reward_coef_r2="{0:0.1,300:0.2,600:0.3,900:0.4,1200:0.5,1500:0.6,1800:0.7,2100:0.8,2400:0.9}" --reward_coef_r_crossent="{0:0.05,1000:0.08,2000:0.1,3000:0.15,4000:0.2,5000:0.3,6000:0.4,7000:0.6,8000:0.7}" --sampling_global_step=5500
