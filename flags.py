import tensorflow as tf
import os
import json
import hickle as hkl 
dirname = os.path.dirname(os.path.abspath(__file__))

src_vocab_size = int(os.environ["src_vocab_size"]) 
trg_vocab_size = int(os.environ["trg_vocab_size"]) 
hidden_size = int(os.environ["hidden_size"]) 
num_layers  = int(os.environ["num_layers"]) 
batch_size  = int(os.environ["batch_size"]) 
dir_base = os.environ["dir_base"] 
print('dir_base: ',dir_base)
model_dir = os.environ["model_dir"] 
#print('model_dir: ',model_dir)
model_RL_dir = os.environ["model_RL_dir"] 
corpus_dir = os.environ["corpus_dir"] 
fasttext_model = os.environ["fasttext_model"] 
source_data = os.environ["source_data"] 
target_data = os.environ["target_data"]
source_mapping = os.environ["source_mapping"] 
target_mapping = os.environ["target_mapping"] 
fasttext_hkl = os.environ["fasttext_hkl"] 
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
tf.app.flags.DEFINE_integer('skip', int(os.environ["skip"]) , 'skip samples')
tf.app.flags.DEFINE_string('mode', os.environ["mode"], 'mode of the seq2seq model')
tf.app.flags.DEFINE_string('source_data', source_data, 'file of source')
tf.app.flags.DEFINE_string('target_data', target_data, 'file of target')
tf.app.flags.DEFINE_string('model_dir', model_dir, 'directory of model')
tf.app.flags.DEFINE_string('model_rl_dir',model_RL_dir, 'directory of RL model')
tf.app.flags.DEFINE_string('sentiment_model',os.environ["sentiment_model"], 'directory of sentiment model')
tf.app.flags.DEFINE_integer('check_step', int(os.environ["check_step"]), 'step interval of saving model')
tf.app.flags.DEFINE_boolean('keep_best_model', bool(os.environ["keep_best_model"]), 'if performance of validation set not gonna be better, then forgo the model')
# for rnn dropout
tf.app.flags.DEFINE_float('input_keep_prob', float(os.environ["input_keep_prob"]), 'step input dropout of saving model')
tf.app.flags.DEFINE_float('output_keep_prob', float(os.environ["output_keep_prob"]), 'step output dropout of saving model')
tf.app.flags.DEFINE_float('state_keep_prob', float(os.environ["state_keep_prob"]), 'step state dropout of saving model')
# output_keep_prob is the dropout added to the RNN's outputs, the dropout will have no effect on the calculation of the subsequent states.
# beam search
tf.app.flags.DEFINE_boolean('beam_search', bool(os.environ["beam_search"]), 'beam search')
tf.app.flags.DEFINE_integer('beam_size', int(os.environ["beam_size"]), 'beam size')
tf.app.flags.DEFINE_string('length_penalty', os.environ["length_penalty"], 'length penalty type')
tf.app.flags.DEFINE_float('length_penalty_factor', float(os.environ["length_penalty_factor"]), 'length penalty factor')
tf.app.flags.DEFINE_boolean('debug', bool(os.environ["debug"]), 'debug')
# schedule sampling
tf.app.flags.DEFINE_string('schedule_sampling', os.environ["schedule_sampling"], 'schedule sampling type[linear|exp|inverse_sigmoid|False]')
tf.app.flags.DEFINE_float('sampling_decay_rate', float(os.environ["sampling_decay_rate"]), 'schedule sampling decay rate')
#tf.app.flags.DEFINE_integer('sampling_global_step', 450000, 'sampling_global_step')
tf.app.flags.DEFINE_integer('sampling_global_step', int(os.environ["sampling_global_step"]), 'sampling_global_step')
tf.app.flags.DEFINE_integer('sampling_decay_steps', int(os.environ["sampling_decay_steps"]), 'sampling_decay_steps')
tf.app.flags.DEFINE_boolean('reset_sampling_prob', bool(os.environ["reset_sampling_prob"]), 'reset_sampling_prob')
# word segmentation type
tf.app.flags.DEFINE_string('src_word_seg', os.environ["src_word_seg"], 'source word segmentation type')
tf.app.flags.DEFINE_string('trg_word_seg', os.environ["trg_word_seg"], 'target word segmentation type')
tf.app.flags.DEFINE_string('sent_word_seg', os.environ["sent_word_seg"], 'sentiment word segmentation type')
# if load pretrain word vector
tf.app.flags.DEFINE_string('pretrain_vec', os.environ["pretrain_vec"], 'load pretrain word vector')
tf.app.flags.DEFINE_boolean('pretrain_trainable', bool(os.environ["pretrain_trainable"]), 'pretrain vec trainable or not')
# RL reword related
tf.app.flags.DEFINE_string('reward_coef_r2', os.environ["reward_coef_r2"], 'r2 coeficient dictionary')
tf.app.flags.DEFINE_string('reward_coef_r_crossent', os.environ["reward_coef_r_crossent"], 'r_crossent coeficient dictionary')
tf.app.flags.DEFINE_boolean('add_crossent', bool(os.environ["add_crossent"]), 'whether return cross entropy as reward')
tf.app.flags.DEFINE_boolean('norm_crossent', bool(os.environ["norm_crossent"]), 'whether normalize the cross entropy')
tf.app.flags.DEFINE_float('reward_gamma', float(os.environ["reward_gamma"]), 'reward discount rate')

tf.app.flags.DEFINE_string('export_eval_dir', os.path.join(dirname,"outputs"), 'directory of evaluation result')

tf.app.flags.DEFINE_string('bind', os.environ["bind"], 'Server address')


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
SEED = os.environ["SEED"] 
buckets = eval(os.environ["buckets"]) 
split_ratio = float(os.environ["split_ratio"]) 

# for inference filter dirty words
with open('replace_words.json','r') as f:
    replace_words = json.load(f)

# for reset schedule sampling probability
reset_prob = float(os.environ["reset_prob"])

# apply same word segment strategy to both source and target or not
word_seg_strategy = os.environ["word_seg_strategy"] 

# special tags 
_PAD = os.environ["_PAD"].encode('utf-8') 
_GO = os.environ["_GO"].encode('utf-8') 
_EOS = os.environ["_EOS"].encode('utf-8') 
_UNK = os.environ["_UNK"].encode('utf-8') 
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]
SPECIAL_TAGS_COUNT = len(_START_VOCAB)

PAD_ID = os.environ["PAD_ID"] 
GO_ID = os.environ["GO_ID"] 
EOS_ID = os.environ["EOS_ID"] 
UNK_ID = os.environ["UNK_ID"] 

# word segmentation dictionary
dict_path = os.environ["dict_path"] 
dict_path = os.path.join(dirname,dict_path)
