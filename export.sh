#!/bin/bash
source ./vars

current_dir=`pwd`
epochs=(3500 4000 4500 5000 5500)
probs=(0.2 0.5 0.75 1 "change")

outputs_dir=outputs/${which_corpus}
mkdir -p $outputs_dir
python export_evaluations.py --export_eval_dir=${outputs_dir} 
for e in ${epochs[@]}
  do
    echo $e
    for d in ${probs[@]}
      do
        cd "${model_RL_dir}/$d"; sed -i "1 s/\([0-9]\+\)/$e/g" checkpoint; cd ${current_dir} 
      done
    python export_evaluations_rl.py --mode=RL --export_eval_dir=${outputs_dir} 
    cd outputs; python combine_files.py --prefix=${which_corpus}/${dir_base} --epoch=${e}; cd ${current_dir};
  done

output_keep_prob=0.85
outputs_dir=outputs/${which_corpus}_dropout
mkdir -p $outputs_dir
python export_evaluations.py --export_eval_dir=${outputs_dir} --output_keep_prob=${output_keep_prob} 
for e in ${epochs[@]}
  do
    echo $e
    for d in ${probs[@]}
      do
        cd "${model_RL_dir}/$d"; sed -i "1 s/\([0-9]\+\)/$e/g" checkpoint; cd ${current_dir} 
      done
    python export_evaluations_rl.py --mode=RL --export_eval_dir=${outputs_dir} --output_keep_prob=${output_keep_prob} 
    cd outputs; python combine_files.py --prefix=${which_corpus}_dropout/${dir_base} --epoch=${e}; cd ${current_dir};
  done
