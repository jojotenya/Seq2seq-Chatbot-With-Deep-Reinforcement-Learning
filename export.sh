#!/bin/bash
source ./vars

current_dir=`pwd`
epochs=(3500 4000 4500 5000 5500)
probs=(0.2 0.5 0.75 1 "change")

mkdir -p outputs/ptt 
for e in ${epochs[@]}
  do
    echo $e
    for d in ${probs[@]}
      do
        cd "${model_RL_dir}/$d"; sed -i "1 s/\([0-9]\+\)/$e/g" checkpoint; cd ${current_dir} 
      done
    python export_evaluations.py --export_eval_dir=outputs/ptt 
    python export_evaluations_rl.py --mode=RL --export_eval_dir=outputs/ptt 
    cd outputs; python combine_files.py --prefix=ptt/ptt_300_4_32_jieba_s --epoch=${e}; cd ${current_dir};
  done

output_keep_prob=0.85
mkdir -p outputs/ptt_dropout 
for e in ${epochs[@]}
  do
    echo $e
    for d in ${probs[@]}
      do
        cd "${model_RL_dir}/$d"; sed -i "1 s/\([0-9]\+\)/$e/g" checkpoint; cd ${current_dir} 
      done
    python export_evaluations.py --export_eval_dir=outputs/ptt_dropout --output_keep_prob=${output_keep_prob} 
    python export_evaluations_rl.py --mode=RL --export_eval_dir=outputs/ptt_dropout --output_keep_prob=${output_keep_prob} 
    cd outputs; python combine_files.py --prefix=ptt_dropout/ptt_300_4_32_jieba_s --epoch=${e}; cd ${current_dir};
  done
