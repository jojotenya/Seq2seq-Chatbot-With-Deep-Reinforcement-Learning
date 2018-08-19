import tensorflow as tf
import os
import json

src_vocab_size = 200000 
trg_vocab_size = 6992 
hidden_size = 512
num_layers = 4 
batch_size = 32
dir_base = 'xhj_ptt_%s_%s_%s_sche_jieba_s'%(hidden_size,num_layers,batch_size)
model_dir = 'model/%s/'%dir_base 
model_RL_dir = 'model_RL/%s/'%dir_base
corpus_dir = 'corpus/%s'%dir_base
source_data = '%s/source'%corpus_dir
target_data = '%s/target'%corpus_dir
if not os.path.exists(model_dir):
    print('create model dir: ',model_dir)
    os.mkdir(model_dir)
if not os.path.exists(model_RL_dir):
    print('create model RL dir: ',model_RL_dir)
    os.mkdir(model_RL_dir)
if not os.path.exists(corpus_dir):
    print('create corpus dir: ',corpus_dir)
    os.mkdir(corpus_dir)

tf.app.flags.DEFINE_integer('src_vocab_size', src_vocab_size, 'vocabulary size of the input')
tf.app.flags.DEFINE_integer('trg_vocab_size', trg_vocab_size, 'vocabulary size of the input')
tf.app.flags.DEFINE_integer('hidden_size', hidden_size, 'number of units of hidden layer')
tf.app.flags.DEFINE_integer('num_layers', num_layers, 'number of layers')
tf.app.flags.DEFINE_integer('batch_size', batch_size, 'batch size')
tf.app.flags.DEFINE_string('mode', 'MLE', 'mode of the seq2seq model')
tf.app.flags.DEFINE_string('source_data', source_data, 'file of source')
tf.app.flags.DEFINE_string('target_data', target_data, 'file of target')
tf.app.flags.DEFINE_string('model_dir', model_dir, 'directory of model')
tf.app.flags.DEFINE_string('model_rl_dir',model_RL_dir, 'directory of RL model')
tf.app.flags.DEFINE_integer('check_step', '500', 'step interval of saving model')
# for rnn dropout
tf.app.flags.DEFINE_float('input_keep_prob', '1.0', 'step input dropout of saving model')
tf.app.flags.DEFINE_float('output_keep_prob', '1.0', 'step output dropout of saving model')
tf.app.flags.DEFINE_float('state_keep_prob', '1.0', 'step state dropout of saving model')
# output_keep_prob is the dropout added to the RNN's outputs, the dropout will have no effect on the calculation of the subsequent states.
# beam search
tf.app.flags.DEFINE_boolean('beam_search', False, 'beam search')
tf.app.flags.DEFINE_integer('beam_size', 10 , 'beam size')
tf.app.flags.DEFINE_boolean('debug', True, 'debug')
# schedule sampling
tf.app.flags.DEFINE_string('schedule_sampling', 'linear', 'schedule sampling type[linear|exp|inverse_sigmoid|False]')
tf.app.flags.DEFINE_float('sampling_decay_rate', 0.99 , 'schedule sampling decay rate')
tf.app.flags.DEFINE_integer('sampling_global_step', 250000, 'sampling_global_step')
tf.app.flags.DEFINE_integer('sampling_decay_steps', 300, 'sampling_decay_steps')
tf.app.flags.DEFINE_boolean('reset_sampling_prob', False, 'reset_sampling_prob')
# word segmentation type
tf.app.flags.DEFINE_string('src_word_seg', 'word', 'source word segmentation type')
tf.app.flags.DEFINE_string('trg_word_seg', 'char', 'target word segmentation type')


FLAGS = tf.app.flags.FLAGS


# for data etl
SEED = 112
buckets = [(10, 10), (15, 15), (25, 25), (50, 50)]
split_ratio = 0.9

# for inference filter dirty words
with open('replace_words.json','r') as f:
    replace_words = json.load(f)

# for reset schedule sampling probability
reset_prob = 1.0

# apply same word segment strategy to both source and target or not
word_seg_strategy = 'diff'
