import matplotlib as mpl
mpl.use('Agg')
import tensorflow as tf
import os
import re
import copy
import pandas as pd
from datetime import datetime
from itertools import product
from functools import reduce
from collections import namedtuple
import hickle as hkl
from run import create_seq2seq, inference
from flags import FLAGS, buckets
import data_utils
if FLAGS.src_word_seg == 'word':
    import jieba
    jieba.initialize()
FLAGS.debug = False
batch_size = 1

dir_name = os.path.dirname(os.path.abspath(__file__))
outputs_dir = os.path.join(dir_name,'outputs')
if not os.path.exists(outputs_dir):
    print('create outputs dir: ',outputs_dir)
    os.mkdir(outputs_dir)

#### export figures ####
check_steps = FLAGS.check_step
index_start = 100
index_end = int(450000/check_steps)

#### export evaluations ####
def get_all_output_dfs(model_dicts,params,use_current_model=False,export_total=True,export_each=True,pretrain_vec='fasttext'):
    if use_current_model:
        model_dicts = [_]
    if export_total:
        dfs = []
    for model_dict in model_dicts: 
        df = get_output_dfs_per_model(model_dict,params,use_current_model,export_each=export_each,pretrain_vec=pretrain_vec)
        dfs.append(df)
    if export_total:
        model_names = list(map(lambda x:x['name'],model_dicts))
        dfs = list(map(lambda x,y:adjust_df_struct(x,y),dfs,model_names))
        dfs = pd.concat(dfs,axis=1)
        same_corpus = from_same_corpus(model_names) 
        if same_corpus:
            sources = list(map(lambda x:(x,'source'),model_names))
            targets = list(map(lambda x:(x,'target'),model_names))
            st = copy.deepcopy(sources+targets)
            source = sources.pop() 
            target = targets.pop() 
            dfs_source = dfs[source]
            dfs_target = dfs[target]
            dfs_source.name = 'source'
            dfs_target.name = 'target'
            dfs.drop(st,axis=1,inplace=True)
            dfs = pd.concat([dfs_source,dfs_target,dfs],axis=1)
        dfs.to_csv(os.path.join(outputs_dir,'%s.csv'%datetime.now().strftime('%s')),index=False)

def compare_corpus(name1,name2,result):
    name1 = ''.join(name1.split('_')[:2])
    name2 = ''.join(name2.split('_')[:2])
    result.append(name1 == name2)
    return name2

def from_same_corpus(model_names):
    result = []
    names = []
    for name in model_names:
        names.append(''.join(name.split('_')[:2]))
    reduce(lambda x,y:compare_corpus(x,y,result),names)
    return bool(reduce(lambda x,y: x*y,result))

def adjust_df_struct(df,model_name):
    cols = list(copy.deepcopy(df.columns))
    columns = pd.MultiIndex.from_product([[model_name],cols])
    df.columns = columns
    return df
     
def get_output_dfs_per_model(model_dict,params,use_current_model=False,export_each=True,pretrain_vec='fasttext'):
    if use_current_model:
        source_mapping = '%s.%s.mapping'%(FLAGS.source_data,FLAGS.src_vocab_size) 
        target_mapping = '%s.%s.mapping'%(FLAGS.target_data,FLAGS.trg_vocab_size)
        source_data = FLAGS.source_data
        target_data = FLAGS.target_data
        model_name = FLAGS.model_dir 
        model_name = re.sub('/','',model_name)
        model_name = re.sub('model','',model_name)
    else:
        source_data = 'corpus/%s/source'%model_dict['name']
        target_data = 'corpus/%s/target'%model_dict['name']
        source_mapping = '%s.%s.mapping'%(source_data,model_dict['source_token']) 
        target_mapping = '%s.%s.mapping'%(target_data,model_dict['target_token'])
        model_name = model_dict['name']
        FLAGS.source_data = source_data
        FLAGS.target_data = target_data
        FLAGS.src_vocab_size = model_dict['source_token'] 
        FLAGS.trg_vocab_size = model_dict['target_token'] 
        FLAGS.model_dir = os.path.join('model/',model_name)
        if pretrain_vec == 'fasttext':
            fasttext_hkl = 'corpus/%s/fasttext.hkl'%model_dict['name']
            FLAGS.pretrain_vec = hkl.load(fasttext_hkl)
    print('########################################################################')
    print('model_dir: ',FLAGS.model_dir)
    print('fasttext_hkl: ',fasttext_hkl)
    print('########################################################################')

    src_vocab_dict, _ = data_utils.read_map(source_mapping)
    _ , trg_vocab_dict = data_utils.read_map(target_mapping)

    d_valid = data_utils.read_data(source_data + '_val.token',target_data + '_val.token',buckets)  

    dfs = []
    DF = namedtuple("DF",("id","source","target")) 
    for bucket_id, d_val in enumerate(d_valid):
        for i, d in enumerate(d_val):
            source,target = (None,None)
            try:
                source = ''.join(data_utils.token_to_text(d[0],source_mapping))
            except IndexError:
                pass
            try:
                target = ''.join(data_utils.token_to_text(d[1],target_mapping))
                target = re.sub('EOS','',target)
            except IndexError:
                pass
            dfs.append(DF("%s_%s"%(bucket_id,i),source,target))
    dfs = pd.DataFrame(dfs)
    dfs = [dfs]
    
    with open(source_data + '_val.token' , 'r') as f:
        token_ids_list = []
        for i, row in enumerate(f.readlines()):
            if i > 1: break
            token_ids_list.append(row.strip().split(' '))
    
    tf.reset_default_graph()
    with tf.Session() as sess:
        vocab_dict, vocab_list = data_utils.read_map(source_mapping)
        model = create_seq2seq(sess, 'TEST')
        model.batch_size = batch_size 
        params = get_all_products(params)
        for param in params:
            print('param: ', param)
            df = get_output_df(sess,param,d_valid,model,src_vocab_dict,trg_vocab_dict)
            dfs.append(df)
    dfs = map(lambda x: x.set_index('id'), dfs)
    dfs = reduce(lambda x,y : x.join(y),dfs) 
    dfs = dfs.reset_index()
    if export_each:
        dfs.to_csv(os.path.join(outputs_dir,'%s.csv'%model_dict['name']),index=False)
    return dfs

def get_all_products(params):
    products = []
    for param in params:
        vals = []
        for v in param.values():
            if not isinstance(v, list):
                v = [v] 
            vals.append(v)
        vals = list(product(*vals))
        for val in vals:
            prod = {}
            for i,k in enumerate(param.keys()):
                prod[k] = val[i]
            products.append(prod)
    return products
    
def get_output_df(sess,param,d_valid,model,src_vocab_dict,trg_vocab_dict):
    if param['beam_search']:
        if param['length_penalty']:
            if param['length_penalty'] == 'rerank':
                title = 'beam%s_%s'%(param['beam_size'],param['length_penalty'])
            elif param['length_penalty'] == 'penalty':
                title = 'beam%s_%s_%s'%(param['beam_size'],param['length_penalty'],param['length_penalty_factor'])
        else:
            title = 'beam%s'%(param['beam_size'])
    else:
        title = 'greedy'

    if '-' in title: title = re.sub('-','N',title) 
    if '.' in title: title = re.sub('\.','',title) 
    DF = namedtuple("DF",["id",title])
    outputs = [] 
    for bucket_id, d_val in enumerate(d_valid):
        for i in range(len(d_val))[::model.batch_size]:
            try:
                data = d_val[i:i+model.batch_size]
            except IndexError:
                data = d_val[i:]
            encoder_inputs, decoder_inputs, weight = model.get_batch({bucket_id:data}, bucket_id, rand=False)
            output = model.run(sess, encoder_inputs, decoder_inputs, weight, bucket_id)
            output = inference(model,output,src_vocab_dict,trg_vocab_dict,debug=False,verbose=True)
            if isinstance(output,list):
                #output = list(map(lambda x,y:'%s\t%s'%(x,y),source,output)) 
                output = '\n'.join(output)
            else:
                #output = '%s\t%s\n'%(source,output)
                output = '%s\n'%output
            outputs.append(DF('%s_%s'%(bucket_id,i),output.strip())) 
    df = pd.DataFrame(outputs)
    return df


#### export figures ####
def get_figure(model_name,index_start=index_start,index_end=index_end,use_current_model=False):
    if use_current_model:
        model_path = FLAGS.model_dir
        model_name = re.sub('/','',model_path)
        model_name = re.sub('model','',model_path)
    model_path = os.path.join('model/',model_name)
    train_path = os.path.join(model_path,'loss_train')
    val_path = os.path.join(model_path,'loss_val')

    df = {}
    #load train
    with open(train_path,'r') as f:
        rows = [float(row.strip()) for row in f.readlines()]
        df['train']=rows

    #load val
    with open(val_path,'r') as f:
        rows = [row.strip() for row in f.readlines()]
        buckets = ([],[],[],[])
        for i in range(len(rows))[::5]:
            groups = rows[i:i+5]
            for b,g in enumerate(groups):
                if b > 3: break
                buckets[b].append(float(g))
        for i,b in enumerate(buckets):
            df['val_%s'%i]=b

    df = pd.DataFrame(df)
    df = df.iloc[index_start:index_end]
    df.index = range(df.shape[0])
    df = df.assign(index=df.index*check_steps)
    df = df[['index','train','val_0','val_1','val_2','val_3']]

    df = df.set_index('index')
    p = df.plot()
    fig = p.get_figure()
    fig.savefig('%s.png'%(os.path.join(outputs_dir,model_name)))
