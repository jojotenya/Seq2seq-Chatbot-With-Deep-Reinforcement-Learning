#!/bin/bash

mkdir -p outputs/xhj 
python export_evaluations.py --export_eval_dir=outputs/xhj 
python export_evaluations_rl.py --mode=RL --export_eval_dir=outputs/xhj 
mv outputs; python combine_files.py --prefix=xhj/xhj_300_4_32_jieba_s; mv ..;

mkdir -p outputs/xhj_dropout 
python export_evaluations.py --export_eval_dir=outputs/xhj_dropout 
python export_evaluations_rl.py --mode=RL --export_eval_dir=outputs/xhj_dropout 
mv outputs; python combine_files.py --prefix=xhj_dropout/xhj_300_4_32_jieba_s; mv ..;
