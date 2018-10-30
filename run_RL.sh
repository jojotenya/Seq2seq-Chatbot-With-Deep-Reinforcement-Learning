#!/bin/bash
model=xhj_300_4_32_jieba_s

python run_srnn.py --mode=RL --batch_size=128 --reward_coef="{0:0.2}" --sampling_global_step=2500
mkdir model_RL/$model/0.2;
mv model_RL/$model/checkpoint model_RL/xhj_300_4_32_jieba_s/0.2;
mv model_RL/$model/RL.ckpt* model_RL/xhj_300_4_32_jieba_s/0.2;

python run_srnn.py --mode=RL --batch_size=128 --reward_coef="{0:0.5}" --sampling_global_step=2500
mkdir model_RL/$model/0.5;
mv model_RL/$model/checkpoint model_RL/xhj_300_4_32_jieba_s/0.5;
mv model_RL/$model/RL.ckpt* model_RL/xhj_300_4_32_jieba_s/0.5;

python run_srnn.py --mode=RL --batch_size=128 --reward_coef="{0:0.75}" --sampling_global_step=2500
mkdir model_RL/$model/0.75;
mv model_RL/$model/checkpoint model_RL/xhj_300_4_32_jieba_s/0.75;
mv model_RL/$model/RL.ckpt* model_RL/xhj_300_4_32_jieba_s/0.75;

python run_srnn.py --mode=RL --batch_size=128 --reward_coef="{0:1}" --sampling_global_step=2100
mkdir model_RL/$model/1;
mv model_RL/$model/checkpoint model_RL/xhj_300_4_32_jieba_s/1;
mv model_RL/$model/RL.ckpt* model_RL/xhj_300_4_32_jieba_s/1;

python run_srnn.py --mode=RL --batch_size=128 --reward_coef="{0:0.1,300:0.2,600:0.3,900:0.4,1200:0.5,1500:0.6,1800:0.7,2100:0.8,2400:0.9}" --sampling_global_step=2500
mkdir model_RL/$model/change;
mv model_RL/$model/checkpoint model_RL/xhj_300_4_32_jieba_s/change;
mv model_RL/$model/RL.ckpt* model_RL/xhj_300_4_32_jieba_s/change;
