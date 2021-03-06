import tensorflow as tf
import numpy as np
import json
import re
import os
import sys 
sys.path.append('sentiment_analysis/')
import math
from termcolor import colored

import data_utils
import seq2seq_model
from seq2seq import bernoulli_sampling
#from sentiment_analysis import run
#from sentiment_analysis import dataset
from sentiment_analysis_srnn import utils as utils_srnn 
from flags import FLAGS, SEED, buckets, replace_words, reset_prob, source_mapping, target_mapping 
from utils import qulify_sentence

# mode variable has three different mode:
# 1. MLE
# 2. RL
# 3. TEST
def create_seq2seq(session, mode):

  if mode == 'TEST':
    FLAGS.schedule_sampling = False 
  else:
    FLAGS.beam_search = False
  print('FLAGS.beam_search: ',FLAGS.beam_search)
  print('FLAGS.length_penalty: ',FLAGS.length_penalty)
  print('FLAGS.length_penalty_factor: ',FLAGS.length_penalty_factor)
  if FLAGS.beam_search:
    print('FLAGS.beam_size: ',FLAGS.beam_size)
    print('FLAGS.debug: ',bool(FLAGS.debug))
      
  model = seq2seq_model.Seq2seq(src_vocab_size = FLAGS.src_vocab_size,
                                trg_vocab_size = FLAGS.trg_vocab_size,
                                buckets = buckets,
                                size = FLAGS.hidden_size,
                                num_layers = FLAGS.num_layers,
                                batch_size = FLAGS.batch_size,
                                mode = mode,
                                input_keep_prob = FLAGS.input_keep_prob,
                                output_keep_prob = FLAGS.output_keep_prob,
                                state_keep_prob = FLAGS.state_keep_prob,
                                beam_search = FLAGS.beam_search,
                                beam_size = FLAGS.beam_size,
                                schedule_sampling = FLAGS.schedule_sampling,
                                sampling_decay_rate = FLAGS.sampling_decay_rate,
                                sampling_global_step = FLAGS.sampling_global_step,
                                sampling_decay_steps = FLAGS.sampling_decay_steps,
                                pretrain_vec = FLAGS.pretrain_vec,
                                pretrain_trainable = FLAGS.pretrain_trainable,
                                length_penalty = FLAGS.length_penalty,
                                length_penalty_factor = FLAGS.length_penalty_factor
                                )
  
  if mode != 'TEST':

      if len(FLAGS.bind) > 0:
          ckpt = tf.train.get_checkpoint_state(FLAGS.bind)
      else:
          ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
  else:
      if len(FLAGS.bind) > 0:
          ckpt = tf.train.get_checkpoint_state(FLAGS.bind)
      else:
          ckpt = tf.train.get_checkpoint_state(FLAGS.model_rl_dir)
  
  if ckpt:
    print("Reading model from %s, mode: %s" % (ckpt.model_checkpoint_path, mode))
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Create model with fresh parameters, mode: %s" % mode)
    session.run(tf.global_variables_initializer())
  
  return model

def train_MLE(): 
  '''
  data_utils.prepare_whole_data(FLAGS.source_data, FLAGS.target_data, FLAGS.src_vocab_size, FLAGS.trg_vocab_size)

  # read dataset and split to training set and validation set
  d = data_utils.read_data(FLAGS.source_data + '.token', FLAGS.target_data + '.token', buckets)
  np.random.seed(SEED)
  np.random.shuffle(d)
  print('Total document size: %s' % sum(len(l) for l in d))
  print('len(d): ', len(d))
  d_train = [[] for _ in range(len(d))]
  d_valid = [[] for _ in range(len(d))]
  for i in range(len(d)):
    d_train[i] = d[i][:int(0.9 * len(d[i]))]
    d_valid[i] = d[i][int(-0.1 * len(d[i])):]
  '''

  d_train = data_utils.read_data(FLAGS.source_data + '_train.token',FLAGS.target_data + '_train.token',buckets)
  d_valid = data_utils.read_data(FLAGS.source_data + '_val.token',FLAGS.target_data + '_val.token',buckets)
  
  print('Total document size of training data: %s' % sum(len(l) for l in d_train))
  print('Total document size of validation data: %s' % sum(len(l) for l in d_valid))

  train_bucket_sizes = [len(d_train[b]) for b in range(len(d_train))]
  train_total_size = float(sum(train_bucket_sizes))
  train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                         for i in range(len(train_bucket_sizes))]
  print('train_bucket_sizes: ',train_bucket_sizes)
  print('train_total_size: ',train_total_size)
  print('train_buckets_scale: ',train_buckets_scale)
  valid_bucket_sizes = [len(d_valid[b]) for b in range(len(d_valid))]
  valid_total_size = float(sum(valid_bucket_sizes))
  valid_buckets_scale = [sum(valid_bucket_sizes[:i + 1]) / valid_total_size
                         for i in range(len(valid_bucket_sizes))]
  print('valid_bucket_sizes: ',valid_bucket_sizes)
  print('valid_total_size: ',valid_total_size)
  print('valid_buckets_scale: ',valid_buckets_scale)

  with tf.Session() as sess:

    model = create_seq2seq(sess, 'MLE')
    if FLAGS.reset_sampling_prob: 
      with tf.variable_scope('sampling_prob',reuse=tf.AUTO_REUSE):
        sess.run(tf.assign(model.sampling_probability,reset_prob))
    if FLAGS.schedule_sampling:
      if not FLAGS.keep_best_model:
        FLAGS.keep_best_model = False
      print('model.sampling_probability: ',model.sampling_probability_clip)
    #sess.run(tf.assign(model.sampling_probability,1.0))
    step = 0
    loss = 0
    loss_list = []
    perplexity_valid_min = float('Inf')
 
    if FLAGS.schedule_sampling:
      print('sampling_decay_steps: ',FLAGS.sampling_decay_steps)
      print('sampling_probability: ',sess.run(model.sampling_probability_clip))
      print('-----')
    while(True):
      step += 1

      random_number = np.random.random_sample()
      # buckets_scale 是累加百分比
      bucket_id = min([i for i in range(len(train_buckets_scale))
                         if train_buckets_scale[i] > random_number])
      encoder_input, decoder_input, weight = model.get_batch(d_train, bucket_id)
      ''' debug
      inds = [0,1,2,3,4,5] 
      for ind in inds:
          encoder_sent = [b[ind] for b in encoder_input]
          decoder_sent = [b[ind] for b in decoder_input]
          print('len of encoder_input: ',len(encoder_input))
          print('encoder_input: ',encoder_sent,data_utils.token_to_text(encoder_sent,source_mapping))
          print('decoder_input: ',decoder_sent,data_utils.token_to_text(decoder_sent,target_mapping))
          print('-------------------------')
      '''
      #print('batch_size: ',model.batch_size)      ==> 64
      #print('batch_size: ',len(encoder_input[0])) ==> 64
      #print('batch_size: ',len(encoder_input))    ==> 15,50,...
      #print('batch_size: ',len(decoder_input))    ==> 15,50,... 
      #print('batch_size: ',len(weight))           ==> 15,50,...
      loss_train, _ = model.run(sess, encoder_input, decoder_input, weight, bucket_id)
      loss += loss_train / FLAGS.check_step

      #if step!=0 and step % FLAGS.sampling_decay_steps == 0:
      #  sess.run(model.sampling_probability_decay)
      #  print('sampling_probability: ',sess.run(model.sampling_probability))
        
      if step % FLAGS.check_step == 0:
        perplexity_train = np.exp(loss)
        with open('%s/loss_train'%FLAGS.model_dir,'a') as f:
          f.write('%s\n'%perplexity_train)
        print('Step %s, Training perplexity: %s, Learning rate: %s' % (step, perplexity_train,
                                  sess.run(model.learning_rate))) 
        perplexity_valids = []
        for i in range(len(d_train)):
          encoder_input, decoder_input, weight = model.get_batch(d_valid, i)
          loss_valid, _ = model.run(sess, encoder_input, decoder_input, weight, i, forward_only = True)
          perplexity_valid = np.exp(loss_valid)
          print('  Validation perplexity in bucket %s: %s' % (i,perplexity_valid))
          perplexity_valids.append(perplexity_valid)
        if len(loss_list) > 2 and loss > max(loss_list[-3:]):
          sess.run(model.learning_rate_decay)
        else:
          if step!=0:
            if FLAGS.schedule_sampling:
              sess.run(model.sampling_probability_decay)
              print('sampling_probability: ',sess.run(model.sampling_probability_clip))
        loss_list.append(loss)  
        loss = 0

        if FLAGS.keep_best_model:
          perplexity_valids_mean = np.mean(perplexity_valids)
          if perplexity_valids_mean < perplexity_valid_min: 
            perplexity_valid_min = perplexity_valids_mean
            print('perplexity_valid_min: ',perplexity_valid_min)
            checkpoint_path = os.path.join(FLAGS.model_dir, "MLE.ckpt")
            model.saver.save(sess, checkpoint_path, global_step = step)
            print('Saving model at step %s' % step)
        else:
          checkpoint_path = os.path.join(FLAGS.model_dir, "MLE.ckpt")
          model.saver.save(sess, checkpoint_path, global_step = step)
          print('Saving model at step %s' % step)
        with open('%s/loss_val'%FLAGS.model_dir,'a') as f:
          for perplexity_valid in perplexity_valids:
            f.write('%s\n'%perplexity_valid)
          f.write('-----------------\n')
      if FLAGS.schedule_sampling:
        if step == FLAGS.sampling_global_step: break

def train_RL():
  if FLAGS.sent_word_seg == 'word':
    import jieba
    jieba.load_userdict('dict_fasttext.txt')
  g1 = tf.Graph()
  g2 = tf.Graph()
  #g3 = tf.Graph()
  sess1 = tf.Session(graph = g1)
  sess2 = tf.Session(graph = g2)
  sess_global = tf.Session()
  #sess3 = tf.Session(graph = g3)
  # model is for training seq2seq with Reinforcement Learning
  with g1.as_default():
    model = create_seq2seq(sess1, 'RL')
    # we set sample size = ?
    model.batch_size = FLAGS.batch_size 
  # model_LM is for a reward function (language model)
  with g2.as_default():
    model_LM = create_seq2seq(sess2, 'MLE')
    model_LM.beam_search = False
    # calculate probibility of only one sentence
    model_LM.batch_size = 1

  def LM(encoder_input, decoder_input, weight, bucket_id):
    return model_LM.run(sess2, encoder_input, decoder_input, weight, bucket_id, forward_only = True)
  # new reward function: sentiment score
  #with g3.as_default():
  #  model_SA = run.create_model(sess3, 'test') 
  #  model_SA.batch_size = 1
  from keras.models import load_model
  model_SA = load_model(utils_srnn.model_dir)
 
  def SA(sentence, encoder_length):
    if FLAGS.sent_word_seg == 'word':
      sentence = ''.join(sentence)
      sentence = jieba.lcut(sentence) 
      sentence = ' '.join(sentence)
    elif FLAGS.sent_word_seg == 'char':
      #sentence = ' '.join(sentence)
      pass
    
    #token_ids = dataset.convert_to_token(sentence, model_SA.vocab_map)
    token_ids = utils_srnn.text_to_sequence(sentence)
    token_ids = utils_srnn.pad_sequences([token_ids], maxlen=utils_srnn.MAX_LEN) 
    token_ids = utils_srnn.get_split_list(token_ids,utils_srnn.SPLIT_DIMS)
    print('sentence: ',''.join(sentence))
    #print('token_ids: ',token_ids)
    #encoder_input, encoder_length, _ = model_SA.get_batch([(0, token_ids)])
    return model_SA.predict(np.array(token_ids),batch_size=1)[0][0]
    #return model_SA.step(sess3, encoder_input, encoder_length)[0][0]

  '''
  data_utils.prepare_whole_data(FLAGS.source_data, FLAGS.target_data, FLAGS.src_vocab_size, FLAGS.trg_vocab_size)
  d = data_utils.read_data(FLAGS.source_data + '.token', FLAGS.target_data + '.token', buckets)

  d = data_utils.read_data(FLAGS.source_data + '_train.token',FLAGS.target_data + '_train.token',buckets)

  train_bucket_sizes = [len(d_train[b]) for b in range(len(d_train))]
  train_total_size = float(sum(train_bucket_sizes))
  train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                         for i in range(len(train_bucket_sizes))]
  '''
  d_train = data_utils.read_data(FLAGS.source_data + '_train.token',FLAGS.target_data + '_train.token',buckets)
  train_bucket_sizes = [len(d_train[b]) for b in range(len(d_train))]
  train_total_size = float(sum(train_bucket_sizes))
  train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                         for i in range(len(train_bucket_sizes))]

  # make RL object read vocab mapping dict, list  
  model.RL_readmap(src_map_path=source_mapping,trg_map_path=target_mapping)
  step = 0
  while(True):
    step += 1

    random_number = np.random.random_sample()
    bucket_id = min([i for i in range(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number])
    print('step: ',step)
    print('bucket_id: ',bucket_id)
    
    # the same encoder_input for sampling batch_size times
    #encoder_input, decoder_input, weight = model.get_batch(d, bucket_id, rand = False)    
    #encoder_input, decoder_input, weight = model.get_batch(d_train, bucket_id, rand = False, initial_id=FLAGS.skip)    
    encoder_input, decoder_input, weight = model.get_batch(d_train, bucket_id, rand = True, initial_id=FLAGS.skip)    
    #print('encoder_input: ',len(encoder_input[0]))
    #print('decoder_input: ',len(decoder_input[0]))
    print('batch_size: ',model.batch_size)
    loss = model.run(sess1, encoder_input, decoder_input, weight, bucket_id, X = LM, Y = SA, sess_global=sess_global)
    print('Loss: %s' %loss)
    print('====================')
   
    # debug 
    #encoder_input = np.reshape(np.transpose(encoder_input, (1, 0, 2)), (-1, FLAGS.vocab_size))
    #encoder_input = np.split(encoder_input, FLAGS.max_length)

    #print(model.token2word(encoder_input)[0])
    #print(model.token2word(sen)[0])
    
    if step % FLAGS.check_step == 0:
      print('Loss at step %s: %s' % (step, loss))
      checkpoint_path = os.path.join(FLAGS.model_rl_dir, "RL.ckpt")
      model.saver.save(sess1, checkpoint_path, global_step = step)
      print('Saving model at step %s' % step)
    if step == FLAGS.sampling_global_step: break
    


def inference(model,output,src_vocab_dict,trg_vocab_dict,debug=FLAGS.debug,verbose=True):
    print('output: ',type(output),len(output),type(output[0]),output[0].shape,output[0],np.sum(output[0]))
    # beam search all
    if bool(model.beam_search):
        if bool(debug):
            outs = []
            for _ in range(model.beam_size):
                outs.append([])
    
            for out in output:
                for i,o in enumerate(out):
                    outs[i].append(o)
            outs = np.array(outs)
            #print('outs: ',outs.shape)
            outputss = []
            for out in outs:
                #print('out: ',out.shape)
                outputs = [int(np.argmax(logit)) for logit in out]
                outputss.append(outputs)
    
            sys_replys = [] 
            for i,outputs in enumerate(outputss):
                sys_reply = "".join([tf.compat.as_str(trg_vocab_dict[output]) for output in outputs])
                sys_reply = data_utils.sub_words(sys_reply)
                sys_reply = qulify_sentence(sys_reply)
                if i == 0:
                    if verbose:
                        print(colored("Syetem reply(bs best): " + sys_reply,"red"))
                else:
                    if verbose:
                        print("Syetem reply(bs all): " + sys_reply)
                sys_replys.append(sys_reply)
            return sys_replys 

        else:
            outputs = [int(np.argmax(logit, axis=1)) for logit in output]
            if data_utils.EOS_ID in outputs:
              outputs = outputs[:outputs.index(data_utils.EOS_ID)]
            sys_reply = "".join([tf.compat.as_str(trg_vocab_dict[output]) for output in outputs])
            sys_reply = data_utils.sub_words(sys_reply)
            sys_reply = qulify_sentence(sys_reply)
            if verbose:
                print("Syetem reply(bs best): " + sys_reply)
    # MLE
    else:
        if verbose:
            print('output: ', len(output), output[0].shape)
        outputs = [int(np.argmax(logit, axis=1)) for logit in output]
        # If there is an EOS symbol in outputs, cut them at that point.
        if data_utils.EOS_ID in outputs:
          outputs = outputs[:outputs.index(data_utils.EOS_ID)]
        sys_reply = "".join([tf.compat.as_str(trg_vocab_dict[output]) for output in outputs])
        sys_reply = data_utils.sub_words(sys_reply)
        sys_reply = qulify_sentence(sys_reply)
        if verbose:
            print("Syetem reply(MLE): " + sys_reply)
    return sys_reply

def test():
  if FLAGS.src_word_seg == 'word':
    import jieba
    jieba.load_userdict('dict_fasttext.txt')
  sess = tf.Session()
  src_vocab_dict, _ = data_utils.read_map(source_mapping)
  _ , trg_vocab_dict = data_utils.read_map(target_mapping)
  model = create_seq2seq(sess, 'TEST')
  model.batch_size = 1
  
  sys.stdout.write("Input sentence: ")
  sys.stdout.flush()
  sentence = sys.stdin.readline()
  if FLAGS.src_word_seg == 'word':
    sentence = (' ').join(jieba.lcut(sentence))
    print('sentence: ',sentence)
  elif FLAGS.src_word_seg == 'char':
    sentence = (' ').join([s for s in sentence])
  while(sentence):
    token_ids = data_utils.convert_to_token(tf.compat.as_bytes(sentence), src_vocab_dict, False)
    bucket_id = len(buckets) - 1
    for i, bucket in enumerate(buckets):
      if bucket[0] >= len(token_ids):
        bucket_id = i
        break
    # Get a 1-element batch to feed the sentence to the model.
    encoder_input, decoder_input, weight = model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id)
    # Get output logits for the sentence.
    output = model.run(sess, encoder_input, decoder_input, weight, bucket_id)
    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    
    inference(model,output,src_vocab_dict,trg_vocab_dict)
    # Print out French sentence corresponding to outputs.
    #print("Syetem reply: " + "".join([tf.compat.as_str(trg_vocab_dict[output]) for output in outputs]))
    print("User input  : ", end="")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    if FLAGS.src_word_seg == 'word':
      sentence = (' ').join(jieba.lcut(sentence))
      print('sentence: ',sentence)
    elif FLAGS.src_word_seg == 'char':
      sentence = (' ').join([s for s in sentence])

if __name__ == '__main__':
  if FLAGS.mode == 'MLE':
    train_MLE()
  elif FLAGS.mode == 'RL':
    train_RL()
  else:
    test()
