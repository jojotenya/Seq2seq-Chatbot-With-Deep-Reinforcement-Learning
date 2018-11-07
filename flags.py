import tensorflow as tf
import os
import json
import hickle as hkl 
dirname = os.path.dirname(os.path.abspath(__file__))

src_vocab_size = 200000 
trg_vocab_size = 6992 
# for xhj
src_vocab_size = 72255 
trg_vocab_size = 5348 
# ptt only
src_vocab_size = 150000 
trg_vocab_size = 6185 
hidden_size = 300 
num_layers = 4 
batch_size = 32 
dir_base = 'xhj_%s_%s_%s_sche_jieba_s'%(hidden_size,num_layers,batch_size)
dir_base = 'xhj_%s_%s_%s_jieba_s'%(hidden_size,num_layers,batch_size)
dir_base = 'ptt_%s_%s_%s_sche_jieba_s'%(hidden_size,num_layers,batch_size)
dir_base = 'ptt_%s_%s_%s_jieba_s'%(hidden_size,num_layers,batch_size)
print('dir_base: ',dir_base)
model_dir = 'model/%s/'%dir_base 
#print('model_dir: ',model_dir)
model_RL_dir = 'model_RL/%s/'%dir_base
corpus_dir = 'corpus/%s/'%dir_base
fasttext_model = './cc.zh.300.bin'
source_data = '%ssource'%corpus_dir
target_data = '%starget'%corpus_dir
source_mapping = '%s.%s.mapping'%(source_data,src_vocab_size)
target_mapping = '%s.%s.mapping'%(target_data,trg_vocab_size)
fasttext_hkl = '%sfasttext.hkl'%corpus_dir 
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
tf.app.flags.DEFINE_integer('skip', 0, 'skip samples')
tf.app.flags.DEFINE_string('mode', 'MLE', 'mode of the seq2seq model')
tf.app.flags.DEFINE_string('source_data', source_data, 'file of source')
tf.app.flags.DEFINE_string('target_data', target_data, 'file of target')
tf.app.flags.DEFINE_string('model_dir', model_dir, 'directory of model')
tf.app.flags.DEFINE_string('model_rl_dir',model_RL_dir, 'directory of RL model')
tf.app.flags.DEFINE_string('sentiment_model','sentiment_analysis_srnn/saved_model/Model07', 'directory of sentiment model')
tf.app.flags.DEFINE_integer('check_step', '500', 'step interval of saving model')
tf.app.flags.DEFINE_boolean('keep_best_model', True, 'if performance of validation set not gonna be better, then forgo the model')
# for rnn dropout
tf.app.flags.DEFINE_float('input_keep_prob', '1.0', 'step input dropout of saving model')
tf.app.flags.DEFINE_float('output_keep_prob', '1.0', 'step output dropout of saving model')
tf.app.flags.DEFINE_float('state_keep_prob', '1.0', 'step state dropout of saving model')
# output_keep_prob is the dropout added to the RNN's outputs, the dropout will have no effect on the calculation of the subsequent states.
# beam search
tf.app.flags.DEFINE_boolean('beam_search', False, 'beam search')
tf.app.flags.DEFINE_integer('beam_size', 10 , 'beam size')
tf.app.flags.DEFINE_string('length_penalty', 'penalty', 'length penalty type')
tf.app.flags.DEFINE_float('length_penalty_factor', 0.6, 'length penalty factor')
tf.app.flags.DEFINE_boolean('debug', True, 'debug')
# schedule sampling
tf.app.flags.DEFINE_string('schedule_sampling', 'False', 'schedule sampling type[linear|exp|inverse_sigmoid|False]')
tf.app.flags.DEFINE_float('sampling_decay_rate', 0.99 , 'schedule sampling decay rate')
#tf.app.flags.DEFINE_integer('sampling_global_step', 450000, 'sampling_global_step')
tf.app.flags.DEFINE_integer('sampling_global_step', 2500, 'sampling_global_step')
tf.app.flags.DEFINE_integer('sampling_decay_steps', 500, 'sampling_decay_steps')
tf.app.flags.DEFINE_boolean('reset_sampling_prob', False, 'reset_sampling_prob')
# word segmentation type
tf.app.flags.DEFINE_string('src_word_seg', 'word', 'source word segmentation type')
tf.app.flags.DEFINE_string('trg_word_seg', 'char', 'target word segmentation type')
tf.app.flags.DEFINE_string('sent_word_seg', 'char', 'sentiment word segmentation type')
# if load pretrain word vector
tf.app.flags.DEFINE_string('pretrain_vec', 'fasttext', 'load pretrain word vector')
tf.app.flags.DEFINE_boolean('pretrain_trainable', False, 'pretrain vec trainable or not')
# reword coeficient
tf.app.flags.DEFINE_string('reward_coef', '{0:0.2}', 'reward coeficient dictionary')

tf.app.flags.DEFINE_string('export_eval_dir', os.path.join(dirname,"outputs"), 'directory of evaluation result')

tf.app.flags.DEFINE_string('bind', '', 'Server address')


FLAGS = tf.app.flags.FLAGS

if FLAGS.schedule_sampling == 'False' or FLAGS.schedule_sampling == 'None': 
    FLAGS.schedule_sampling = False
if FLAGS.length_penalty == 'False' or FLAGS.length_penalty == 'None':
    FLAGS.length_penalty = None
if FLAGS.pretrain_vec == 'None': 
    FLAGS.pretrain_vec = None
elif FLAGS.pretrain_vec == 'fasttext':
    FLAGS.pretrain_vec = hkl.load(fasttext_hkl)
print('trainable: ',FLAGS.pretrain_trainable)

# for data etl
SEED = 112
buckets = [(10, 10), (15, 15), (25, 25), (50, 50)]
split_ratio = 0.995

# for inference filter dirty words
with open('replace_words.json','r') as f:
    replace_words = json.load(f)

# for reset schedule sampling probability
reset_prob = 1.0

# apply same word segment strategy to both source and target or not
word_seg_strategy = 'diff'

# special tags 
_PAD = b"PAD"
_GO = b"GO"
_EOS = b"EOS"
_UNK = b"UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]
SPECIAL_TAGS_COUNT = len(_START_VOCAB)

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# word segmentation dictionary
dict_path = 'dict_fasttext.txt'
dict_path = os.path.join(dirname,dict_path)
