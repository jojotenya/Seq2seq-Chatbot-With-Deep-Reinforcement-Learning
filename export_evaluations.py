from export_utils import *

#export evaluations
model1 = {
          'name':'ptt_300_4_32_jieba_s',
          'corpus':'ptt_300_4_32_jieba_s',
          'source_token':150000,
          'target_token':6185,
         }

model2 = {
          'name':'ptt_300_4_32_sche_jieba_s',
          'corpus':'ptt_300_4_32_sche_jieba_s',
          'source_token':150000,
          'target_token':6185,
         }

model3 = {
          'name':'xhj_ptt_300_4_32_jieba_s',
          'corpus':'xhj_ptt_300_4_32_jieba_s',
          'source_token':200000,
          'target_token':6992,
         }

model1 = {
          'name':'ptt_300_4_32_jieba_s/0.2',
          'corpus':'ptt_300_4_32_jieba_s',
          'source_token':150000,
          'target_token':6185,
         }

model2 = {
          'name':'ptt_300_4_32_jieba_s/0.5',
          'corpus':'ptt_300_4_32_jieba_s',
          'source_token':150000,
          'target_token':6185,
         }

model3 = {
          'name':'ptt_300_4_32_jieba_s/0.75',
          'corpus':'ptt_300_4_32_jieba_s',
          'source_token':150000,
          'target_token':6185,
         }

model4 = {
          'name':'ptt_300_4_32_jieba_s/1',
          'corpus':'ptt_300_4_32_jieba_s',
          'source_token':150000,
          'target_token':6185,
         }

model5 = {
          'name':'ptt_300_4_32_jieba_s/change',
          'corpus':'ptt_300_4_32_jieba_s',
          'source_token':150000,
          'target_token':6185,
         }

models = [model1,model2,model3,model4,model5]

params = [
  {'beam_search':False},
  #{'beam_search':True,'beam_size':[5,10],'length_penalty':'rerank'},
  #{'beam_search':True,'beam_size':[5,10],'length_penalty':'penalty','length_penalty_factor':[-0.6,0.0,0.6]},
]

use_current_model = False
export_each = True
export_total = True
pretrain_vec = 'fasttext'

get_all_output_dfs(models,params,use_current_model,export_total,export_each,pretrain_vec)

#export figures
index_start = 100
index_end = int(450000/check_steps)
use_current_model = False
model_names = list(map(lambda x:x['name'],models)) 
for model_name in model_names:
    get_figure(model_name,index_start=index_start,index_end=index_end,use_current_model=use_current_model)
