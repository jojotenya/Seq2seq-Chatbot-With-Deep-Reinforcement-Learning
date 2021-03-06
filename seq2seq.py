from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin

#from tensorflow.contrib.rnn.python.ops import core_rnn
import tensorflow as tf
from tensorflow import nn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.rnn import LSTMStateTuple
#from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.distributions import bernoulli
from tensorflow.python.util import nest

from flags import FLAGS


# TODO(ebrevdo): Remove once _linear is fully deprecated.
#linear = core_rnn_cell_impl._linear  # pylint: disable=protected-access
linear = core_rnn_cell._linear  # pylint: disable=protected-access

def bernoulli_sampling(sampling_probability):
  select_sampler = bernoulli.Bernoulli(probs=sampling_probability,dtype=tf.bool)
  select_sampler = select_sampler.sample()
  return select_sampler

def _extract_argmax_and_embed(embedding,
                              output_projection=None,
                              update_embedding=True):
  """Get a loop_function that extracts the previous symbol and embeds it.

  Args:
    embedding: embedding tensor for symbols.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

  Returns:
    A loop function.
  """

  def loop_function(prev, i):
    if output_projection is not None:
      prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
    prev_symbol = math_ops.argmax(prev, 1)
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = array_ops.stop_gradient(emb_prev)
    return emb_prev

  return loop_function


def rnn_decoder(decoder_inputs,
                initial_state,
                cell,
                loop_function=None,
                scope=None):
  """RNN decoder for the sequence-to-sequence model.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor with shape [batch_size x cell.state_size].
    cell: core_rnn_cell.RNNCell defining the cell function and size.
    loop_function: If not None, this function will be applied to the i-th output
      in order to generate the i+1-st input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    scope: VariableScope for the created subgraph; defaults to "rnn_decoder".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing generated outputs.
      state: The state of each cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
        (Note that in some cases, like basic RNN cell or GRU cell, outputs and
         states can be the same. They are different for LSTM cells though.)
  """
  with variable_scope.variable_scope(scope or "rnn_decoder"):
    state = initial_state
    outputs = []
    prev = None
    for i, inp in enumerate(decoder_inputs):
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i)
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      output, state = cell(inp, state)
      outputs.append(output)
      if loop_function is not None:
        prev = output
  return outputs, state

def embedding_rnn_decoder(decoder_inputs,
                          initial_state,
                          cell,
                          num_symbols,
                          embedding_size,
                          output_projection=None,
                          feed_previous=False,
                          update_embedding_for_previous=True,
                          scope=None):
  """RNN decoder with embedding and a pure-decoding option.

  Args:
    decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
    initial_state: 2D Tensor [batch_size x cell.state_size].
    cell: core_rnn_cell.RNNCell defining the cell function.
    num_symbols: Integer, how many symbols come into the embedding.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_symbols] and B has
      shape [num_symbols]; if provided and feed_previous=True, each fed
      previous output will first be multiplied by W and added B.
    feed_previous: Boolean; if True, only the first of decoder_inputs will be
      used (the "GO" symbol), and all other decoder inputs will be generated by:
        next = embedding_lookup(embedding, argmax(previous_output)),
      In effect, this implements a greedy decoder. It can also be used
      during training to emulate http://arxiv.org/abs/1506.03099.
      If False, decoder_inputs are used as given (the standard decoder case).
    update_embedding_for_previous: Boolean; if False and feed_previous=True,
      only the embedding for the first symbol of decoder_inputs (the "GO"
      symbol) will be updated by back propagation. Embeddings for the symbols
      generated from the decoder itself remain unchanged. This parameter has
      no effect if feed_previous=False.
    scope: VariableScope for the created subgraph; defaults to
      "embedding_rnn_decoder".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors. The
        output is of shape [batch_size x cell.output_size] when
        output_projection is not None (and represents the dense representation
        of predicted tokens). It is of shape [batch_size x num_decoder_symbols]
        when output_projection is None.
      state: The state of each decoder cell in each time-step. This is a list
        with length len(decoder_inputs) -- one item for each time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: When output_projection has the wrong shape.
  """
  with variable_scope.variable_scope(scope or "embedding_rnn_decoder") as scope:
    if output_projection is not None:
      dtype = scope.dtype
      proj_weights = ops.convert_to_tensor(output_projection[0], dtype=dtype)
      proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])
      proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
      proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    embedding = variable_scope.get_variable("embedding",
                                            [num_symbols, embedding_size])
    loop_function = _extract_argmax_and_embed(
        embedding, output_projection,
        update_embedding_for_previous) if feed_previous else None
    emb_inp = (embedding_ops.embedding_lookup(embedding, i)
               for i in decoder_inputs)
    return rnn_decoder(
        emb_inp, initial_state, cell, loop_function=loop_function)


def embedding_rnn_seq2seq(encoder_inputs,
                          decoder_inputs,
                          cell,
                          num_encoder_symbols,
                          num_decoder_symbols,
                          embedding_size,
                          output_projection=None,
                          feed_previous=False,
                          dtype=None,
                          scope=None):
  """Embedding RNN sequence-to-sequence model.

  This model first embeds encoder_inputs by a newly created embedding (of shape
  [num_encoder_symbols x input_size]). Then it runs an RNN to encode
  embedded encoder_inputs into a state vector. Next, it embeds decoder_inputs
  by another newly created embedding (of shape [num_decoder_symbols x
  input_size]). Then it runs RNN decoder, initialized with the last
  encoder state, on embedded decoder_inputs.

  Args:
    encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    cell: core_rnn_cell.RNNCell defining the cell function and size.
    num_encoder_symbols: Integer; number of symbols on the encoder side.
    num_decoder_symbols: Integer; number of symbols on the decoder side.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_decoder_symbols] and B has
      shape [num_decoder_symbols]; if provided and feed_previous=True, each
      fed previous output will first be multiplied by W and added B.
    feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
      of decoder_inputs will be used (the "GO" symbol), and all other decoder
      inputs will be taken from previous outputs (as in embedding_rnn_decoder).
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype of the initial state for both the encoder and encoder
      rnn cells (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_rnn_seq2seq"

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors. The
        output is of shape [batch_size x cell.output_size] when
        output_projection is not None (and represents the dense representation
        of predicted tokens). It is of shape [batch_size x num_decoder_symbols]
        when output_projection is None.
      state: The state of each decoder cell in each time-step. This is a list
        with length len(decoder_inputs) -- one item for each time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  """
  with variable_scope.variable_scope(scope or "embedding_rnn_seq2seq") as scope:
    if dtype is not None:
      scope.set_dtype(dtype)
    else:
      dtype = scope.dtype

    # Encoder.
    encoder_cell = copy.deepcopy(cell)
    encoder_cell = core_rnn_cell.EmbeddingWrapper(
        encoder_cell,
        embedding_classes=num_encoder_symbols,
        embedding_size=embedding_size)
    #_, encoder_state = core_rnn.static_rnn(
    _, encoder_state = nn.static_rnn(
        encoder_cell, encoder_inputs, dtype=dtype)

    # Decoder.
    if output_projection is None:
      cell = core_rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)

    if isinstance(feed_previous, bool):
      return embedding_rnn_decoder(
          decoder_inputs,
          encoder_state,
          cell,
          num_decoder_symbols,
          embedding_size,
          output_projection=output_projection,
          feed_previous=feed_previous)

    # If feed_previous is a Tensor, we construct 2 graphs and use cond.
    def decoder(feed_previous_bool):
      reuse = None if feed_previous_bool else True
      with variable_scope.variable_scope(
          variable_scope.get_variable_scope(), reuse=reuse) as scope:
        outputs, state = embedding_rnn_decoder(
            decoder_inputs,
            encoder_state,
            cell,
            num_decoder_symbols,
            embedding_size,
            output_projection=output_projection,
            feed_previous=feed_previous_bool,
            update_embedding_for_previous=False)
        state_list = [state]
        if nest.is_sequence(state):
          state_list = nest.flatten(state)
        return outputs + state_list

    outputs_and_state = control_flow_ops.cond(feed_previous,
                                              lambda: decoder(True),
                                              lambda: decoder(False))
    outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
    state_list = outputs_and_state[outputs_len:]
    state = state_list[0]
    if nest.is_sequence(encoder_state):
      state = nest.pack_sequence_as(
          structure=encoder_state, flat_sequence=state_list)
    return outputs_and_state[:outputs_len], state

def attention_decoder(decoder_inputs,
                      initial_state,
                      attention_states,
                      cell,
                      output_size=None,
                      num_heads=1,
                      loop_function=None,
                      dtype=None,
                      scope=None,
                      initial_state_attention=False,
                      schedule_sampling=False,
                      sampling_probability=None):
  """RNN decoder with attention for the sequence-to-sequence model.

  In this context "attention" means that, during decoding, the RNN can look up
  information in the additional tensor attention_states, and it does this by
  focusing on a few entries from the tensor. This model has proven to yield
  especially good results in a number of sequence-to-sequence tasks. This
  implementation is based on http://arxiv.org/abs/1412.7449 (see below for
  details). It is recommended for complex sequence-to-sequence tasks.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: core_rnn_cell.RNNCell defining the cell function and size.
    output_size: Size of the output vectors; if None, we use cell.output_size.
    num_heads: Number of attention heads that read from attention_states.
    loop_function: If not None, this function will be applied to i-th output
      in order to generate i+1-th input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    dtype: The dtype to use for the RNN initial state (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "attention_decoder".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously
      stored decoder state and attention states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors of
        shape [batch_size x output_size]. These represent the generated outputs.
        Output i is computed from input i (which is either the i-th element
        of decoder_inputs or loop_function(output {i-1}, i)) as follows.
        First, we run the cell on a combination of the input and previous
        attention masks:
          cell_output, new_state = cell(linear(input, prev_attn), prev_state).
        Then, we calculate new attention masks:
        #  new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
        #and then we calculate the output:
        #  output = linear(cell_output, new_attn).
      state: The state of each decoder cell the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: when num_heads is not positive, there are no inputs, shapes
      of attention_states are not set, or input size cannot be inferred
      from the input.
  """
  if not decoder_inputs:
    raise ValueError("Must provide at least 1 input to attention decoder.")
  if num_heads < 1:
    raise ValueError("With less than 1 heads, use a non-attention decoder.")
  if attention_states.get_shape()[2].value is None:
    raise ValueError("Shape[2] of attention_states must be known: %s" %
                     attention_states.get_shape())
  if output_size is None:
    output_size = cell.output_size

  with variable_scope.variable_scope(
      scope or "attention_decoder", dtype=dtype) as scope:
    dtype = scope.dtype

    batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
    attn_length = attention_states.get_shape()[1].value
    if attn_length is None:
      attn_length = array_ops.shape(attention_states)[1]
    attn_size = attention_states.get_shape()[2].value
    #print('attention_states: ', attention_states)
    #print('attention_length: ', attn_length)
    #print('attention_size: ', attn_size)

    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
    hidden = array_ops.reshape(attention_states,
                               [-1, attn_length, 1, attn_size])
    hidden_features = []
    v = []
    attention_vec_size = attn_size  # Size of query vectors for attention.
    for a in xrange(num_heads):
      k = variable_scope.get_variable("AttnW_%d" % a,
                                      [1, 1, attn_size, attention_vec_size])
      # input: [batch, in_height, in_width, in_channels]; filter: [filter_height, filter_width, in_channels, out_channels]
      hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
      v.append(
          variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size]))

    state = initial_state

    def attention(query):
      """Put attention masks on hidden using hidden_features and query."""
      ds = []  # Results of attention reads will be stored here.
      if nest.is_sequence(query):  # If the query is a tuple, flatten it.
        query_list = nest.flatten(query)
        for q in query_list:  # Check that ndims == 2 if specified.
          ndims = q.get_shape().ndims
          if ndims:
            assert ndims == 2
        query = array_ops.concat(query_list, 1)
      ###
      for a in xrange(num_heads):
        with variable_scope.variable_scope("Attention_%d" % a):
          y = linear(query, attention_vec_size, True)
          #print('query: ',query)
          #print('attention_vec_size: ',attention_vec_size)
          #print('y: ',y)
          y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
          #print('y: ',y)
          # Attention mask is a softmax of v^T * tanh(...).
          s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y),
                                  [2, 3])
  
          #print('s: ',s)
          a = nn_ops.softmax(s)
          #print('a: ',a)
          # Now calculate the attention-weighted vector d. (i think it's the context vector)
          #print('hidden: ',hidden)
          d = math_ops.reduce_sum(
              array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
          #print('d: ',d)
          ds.append(array_ops.reshape(d, [-1, attn_size]))
          #print('d: ',array_ops.reshape(d, [-1, attn_size]))
          #print('---------------')
      return ds

    outputs = []
    prev = None
    batch_attn_size = array_ops.stack([batch_size, attn_size])
    #print('batch_attn_size: ',batch_attn_size)
    attns = [
        array_ops.zeros(
            batch_attn_size, dtype=dtype) for _ in xrange(num_heads)
    ]
    for a in attns:  # Ensure the second shape of attention vectors is set.
      a.set_shape([None, attn_size])
    if initial_state_attention:
      attns = attention(initial_state)
    decoder_size = decoder_inputs[0].shape[1]
    for i, inp in enumerate(decoder_inputs):
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      if schedule_sampling:
        if (loop_function is not None and prev is not None):
            inp_prev = loop_function(prev, i)
        else:
            inp_prev = inp
        inp = tf.concat([inp,inp_prev],1)
        inp = tf.cond(bernoulli_sampling(sampling_probability),lambda: inp[:,:decoder_size] , lambda: inp[:,decoder_size:])
      # If loop_function is set, we use it instead of decoder_inputs.
      else:
        if (loop_function is not None and prev is not None):
          with variable_scope.variable_scope("loop_function", reuse=True):
            inp = loop_function(prev, i)
      # Merge input and previous attentions into one vector of the right size.
      input_size = inp.get_shape().with_rank(2)[1]
      if input_size.value is None:
        raise ValueError("Could not infer input size from input: %s" % inp.name)
      x = linear([inp] + attns, input_size, True)
      #print('input_size: ',input_size)
      #print('[inp] + attns: ',[inp] + attns)
      #print('x: ',x)
      # Run the RNN.
      cell_output, state = cell(x, state)
      # Run the attention mechanism.
      if i == 0 and initial_state_attention:
        with variable_scope.variable_scope(
            variable_scope.get_variable_scope(), reuse=True):
          attns = attention(state)
      else:
        attns = attention(state)

      with variable_scope.variable_scope("AttnOutputProjection"):
        output = linear([cell_output] + attns, output_size, True)
      if loop_function is not None:
        prev = output
      outputs.append(output)

  return outputs, state


def embedding_attention_decoder(decoder_inputs,
                                initial_state,
                                attention_states,
                                cell,
                                num_symbols,
                                embedding_size,
                                embedding,
                                num_heads=1,
                                output_size=None,
                                output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False,
                                loop = None):
  """RNN decoder with embedding and attention and a pure-decoding option.

  Args:
    decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: core_rnn_cell.RNNCell defining the cell function.
    num_symbols: Integer, how many symbols come into the embedding.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    num_heads: Number of attention heads that read from attention_states.
    output_size: Size of the output vectors; if None, use output_size.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_symbols] and B has shape
      [num_symbols]; if provided and feed_previous=True, each fed previous
      output will first be multiplied by W and added B.
    feed_previous: Boolean; if True, only the first of decoder_inputs will be
      used (the "GO" symbol), and all other decoder inputs will be generated by:
        next = embedding_lookup(embedding, argmax(previous_output)),
      In effect, this implements a greedy decoder. It can also be used
      during training to emulate http://arxiv.org/abs/1506.03099.
      If False, decoder_inputs are used as given (the standard decoder case).
    update_embedding_for_previous: Boolean; if False and feed_previous=True,
      only the embedding for the first symbol of decoder_inputs (the "GO"
      symbol) will be updated by back propagation. Embeddings for the symbols
      generated from the decoder itself remain unchanged. This parameter has
      no effect if feed_previous=False.
    dtype: The dtype to use for the RNN initial states (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_attention_decoder".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously
      stored decoder state and attention states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing the generated outputs.
      state: The state of each decoder cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: When output_projection has the wrong shape.
  """
  if output_size is None:
    output_size = cell.output_size
  if output_projection is not None:
    proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
    proj_biases.get_shape().assert_is_compatible_with([num_symbols])

  with variable_scope.variable_scope(
      scope or "embedding_attention_decoder", dtype=dtype) as scope:

    #embedding = variable_scope.get_variable("embedding",
    #                                        [num_symbols, embedding_size])
    if feed_previous:
      if loop:
        loop_function = loop
      else:
        loop_function = _extract_argmax_and_embed(
                        embedding, output_projection,
                        update_embedding_for_previous)

    else:
      loop_function = None
    emb_inp = [
        embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs
    ]
    return attention_decoder(
        emb_inp,
        initial_state,
        attention_states,
        cell,
        output_size=output_size,
        num_heads=num_heads,
        loop_function=loop_function,
        initial_state_attention=initial_state_attention)

def loss_normalize(losses):
    losses = tf.convert_to_tensor(losses,dtype=tf.float32)
    mean, std = tf.nn.moments(losses,axes=[0])
    std = tf.cast(std,dtype=tf.float32)
    mean = tf.cast(mean,dtype=tf.float32)
    #mean = tf.reduce_mean(losses)
    #std = np.std(losses)
    #return tf.floordiv(tf.subtract(losses,mean), std)
    return (losses-mean)/(std+1e-12)

def sequence_loss_by_example(logits,
                             targets,
                             weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None,
                             name=None,
                             norm=False):
  # 針對一個句子的output計算, sequence_loss則是針對一個batch計算
  """Weighted cross-entropy loss for a sequence of logits (per example).

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    softmax_loss_function: Function (labels-batch, inputs-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, default: "sequence_loss_by_example".

  Returns:
    1D batch-sized float Tensor: The log-perplexity for each sequence.

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  if len(targets) != len(logits) or len(weights) != len(logits):
    raise ValueError("Lengths of logits, weights, and targets must be the same "
                     "%d, %d, %d." % (len(logits), len(weights), len(targets)))
  with ops.name_scope(name, "sequence_loss_by_example",
                      logits + targets + weights):
    log_perp_list = []
    # default的weight=1
    for logit, target, weight in zip(logits, targets, weights):
      if softmax_loss_function is None:
        # TODO(irving,ebrevdo): This reshape is needed because
        # sequence_loss_by_example is called with scalars sometimes, which
        # violates our general scalar strictness policy.
        target = array_ops.reshape(target, [-1])
        crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
            labels=target, logits=logit)
      else:
        crossent = softmax_loss_function(target, logit)
      log_perp_list.append(crossent * weight)
    # 主要是給計算cross entropy當reward用，所以可以負數。還要在思考這樣好不好
    if norm: 
        log_perp_list = loss_normalize(log_perp_list) 
        log_perps = tf.reduce_sum(log_perp_list)
    else:
        log_perps = math_ops.add_n(log_perp_list)
    if average_across_timesteps:
      total_size = math_ops.add_n(weights)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      try: 
        log_perps /= total_size
      except (ValueError,TypeError) as e:
        log_perps = tf.cast(log_perps,dtype=tf.float32)
        total_size = tf.cast(total_size,dtype=tf.float32)
        log_perps /= total_size
  return log_perps

def sequence_loss(logits,
                  targets,
                  weights,
                  average_across_timesteps=True,
                  average_across_batch=True,
                  softmax_loss_function=None,
                  name=None,
                  norm=False):
  """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    average_across_batch: If set, divide the returned cost by the batch size.
    softmax_loss_function: Function (labels-batch, inputs-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, defaults to "sequence_loss".

  Returns:
    A scalar float Tensor: The average log-perplexity per symbol (weighted).

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  with ops.name_scope(name, "sequence_loss", logits + targets + weights):
    cost = math_ops.reduce_sum(
        sequence_loss_by_example(
            logits,
            targets,
            weights,
            average_across_timesteps=average_across_timesteps,
            softmax_loss_function=softmax_loss_function,
            norm=norm))
    if average_across_batch:
      batch_size = array_ops.shape(targets[0])[0]
      return cost / math_ops.cast(batch_size, cost.dtype)
    else:
      return cost


def model_with_buckets(encoder_inputs,
                       decoder_inputs,
                       targets,
                       weights,
                       buckets,
                       seq2seq,
                       softmax_loss_function=None,
                       per_example_loss=False,
                       name=None,
                       norm=False):
  """Create a sequence-to-sequence model with support for bucketing.

  The seq2seq argument is a function that defines a sequence-to-sequence model,
  e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(
      x, y, core_rnn_cell.GRUCell(24))

  Args:
    encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
    decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
    targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
    weights: List of 1D batch-sized float-Tensors to weight the targets.
    buckets: A list of pairs of (input size, output size) for each bucket.
    seq2seq: A sequence-to-sequence model function; it takes 2 input that
      agree with encoder_inputs and decoder_inputs, and returns a pair
      consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
    softmax_loss_function: Function (labels-batch, inputs-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    per_example_loss: Boolean. If set, the returned loss will be a batch-sized
      tensor of losses for each sequence in the batch. If unset, it will be
      a scalar with the averaged loss from all examples.
    name: Optional name for this operation, defaults to "model_with_buckets".

  Returns:
    A tuple of the form (outputs, losses), where:
      outputs: The outputs for each bucket. Its j'th element consists of a list
        of 2D Tensors. The shape of output tensors can be either
        [batch_size x output_size] or [batch_size x num_decoder_symbols]
        depending on the seq2seq model used.
      losses: List of scalar Tensors, representing losses for each bucket, or,
        if per_example_loss is set, a list of 1D batch-sized float Tensors.

  Raises:
    ValueError: If length of encoder_inputs, targets, or weights is smaller
      than the largest (last) bucket.
  """
  if len(encoder_inputs) < buckets[-1][0]:
    raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                     "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
  if len(targets) < buckets[-1][1]:
    raise ValueError("Length of targets (%d) must be at least that of last"
                     "bucket (%d)." % (len(targets), buckets[-1][1]))
  if len(weights) < buckets[-1][1]:
    raise ValueError("Length of weights (%d) must be at least that of last"
                     "bucket (%d)." % (len(weights), buckets[-1][1]))

  all_inputs = encoder_inputs + decoder_inputs + targets + weights
  losses = []
  outputs = []
  with ops.name_scope(name, "model_with_buckets", all_inputs):
    for j, bucket in enumerate(buckets):
      with variable_scope.variable_scope(
          variable_scope.get_variable_scope(), reuse=True if j > 0 else None):
        #print('encoder_inputs: ',encoder_inputs)
        #print('bucket[0]: ',bucket[0])
        #print('encoder_inputs[:bucket[0]]: ',encoder_inputs[:bucket[0]])
        #print('decoder_inputs: ',decoder_inputs)
        #print('bucket[1]: ',bucket[1])
        #print('decoder_inputs[:bucket[1]]: ',decoder_inputs[:bucket[1]])
        ##print('beam decoder: 6 outputs, normal: 2 outputs')
        bucket_outputs = seq2seq(encoder_inputs[:bucket[0]],
                                 decoder_inputs[:bucket[1]])
        #print('bucket_outputs-0: ',bucket_outputs[0],len(bucket_outputs[0]))
        outputs.append(bucket_outputs[0])
        #outputs[-1]:  (?, 300)
        #targets[:bucket[1]]:  (?,)
        #weights[:bucket[1]]:  (?,)
        if per_example_loss:
          losses.append(
              sequence_loss_by_example(
                  outputs[-1],
                  targets[:bucket[1]],
                  weights[:bucket[1]],
                  softmax_loss_function=softmax_loss_function,
                  norm=norm))
        else:
          losses.append(
              sequence_loss(
                  outputs[-1],
                  targets[:bucket[1]],
                  weights[:bucket[1]],
                  softmax_loss_function=softmax_loss_function,
                  norm=norm))

  return outputs, losses




# beam search
def _extract_beam_search(embedding, beam_size, num_symbols, embedding_size, output_projection=None):

    def loop_function(prev, i, log_beam_probs, beam_path, beam_symbols):
        if output_projection is not None:
            #print('prev: ',prev)
            prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
            #print('output_projection: ',output_projection)
            #print('prev: ',prev)
        # 取log方便相加
        probs = tf.log(tf.nn.softmax(prev))
        #print('probs: ', i, probs)
        if i == 1:
            probs = tf.reshape(probs[0, :], [-1, num_symbols])
            #print('probs: ', i, probs)
        if i > 1:
            # 將序列之機率與前一序列機率相加得到的結果之前有beam_size個序列，此操作產生num_symbols個結果。
            # 所以reshape成這樣的tensor
            #print('log_beam_probs[-1]: ',log_beam_probs[-1])
            #print('probs + log_beam_probs[-1]: ',probs+log_beam_probs[-1])
            probs = tf.reshape(probs + log_beam_probs[-1], [-1, beam_size * num_symbols])
            #print('probs: ', i, probs)
        # 選出機率最大的前beam_size個序列,從beam_size * num_symbols個元素中選出beam_size個
        best_probs, indices = tf.nn.top_k(probs, beam_size)
        #print('best_probs : ',best_probs)
        #print('indices: ',indices)
        indices = tf.stop_gradient(tf.squeeze(tf.reshape(indices, [-1, 1])))
        best_probs = tf.stop_gradient(tf.reshape(best_probs, [-1, 1]))
        #print('indices: ',indices)

        # beam_size * num_symbols，看對應的是哪個序列和斷辭
        symbols = indices % num_symbols  # Which word in vocabulary.
        beam_parent = indices // num_symbols  # Which hypothesis it came from.
        beam_symbols.append(symbols)
        beam_path.append(beam_parent)
        log_beam_probs.append(best_probs)

        # 對beam-search選出的beam size個斷詞進行embedding，得到對應的詞向量
        emb_prev = embedding_ops.embedding_lookup(embedding, symbols)
        emb_prev = tf.reshape(emb_prev, [-1, embedding_size])
        return emb_prev

    return loop_function

def mask_special_tag(symbols,log_probs,beam_size,total_words,index=2):
    '''
    default index is eos (=2)
    '''

    if_special_tag_bool = tf.equal(index,symbols)
    if_special_tag_cond = tf.reduce_sum(tf.cast(if_special_tag_bool,tf.int32),axis=1)
    if_special_tag_cond = tf.cast(if_special_tag_cond,tf.bool)
    if_special_tag = tf.where(if_special_tag_bool)
    special_tag_indices = tf.segment_min(if_special_tag[:,1],if_special_tag[:,0])
    special_tag_indices = tf.cast(special_tag_indices,tf.int32)
    last_indices = tf.ones((beam_size,),dtype=tf.int32)
    last_indices = tf.multiply(last_indices,total_words-1)
    #special_tag_indices = tf.scatter_sub(last_indices,uniq_indices,tf.gather(special_tag_indices,uniq_indices))
    special_tag_indices = tf.where(if_special_tag_cond,special_tag_indices,last_indices)
    indices_matrix = tf.range(0,total_words)
    indices_matrix = tf.reshape(indices_matrix,[1,-1])
    indices_matrix = tf.tile(indices_matrix,[beam_size,1])
    special_tag_indices = tf.reshape(special_tag_indices,[-1,1])
    bool_mask = tf.less_equal(indices_matrix,special_tag_indices)
    zeros = tf.zeros([beam_size,total_words],tf.float32)
    log_probs = tf.where(bool_mask,log_probs,zeros)
    sequence_lengths = tf.reduce_sum(tf.cast(bool_mask,tf.int32),1) 
    return log_probs, sequence_lengths

def length_penalty(sequence_lengths, penalty_type='penalty', penalty_factor=0.6):
    """Calculates the length penalty reference to
    https://arxiv.org/abs/1609.08144
     Args:
      sequence_lengths: The sequence length of all hypotheses, a tensor
        of shape [beam_size, vocab_size].
      penalty_factor: A scalar that weights the length penalty.
    Returns:
      The length penalty factor, a tensor fo shape [beam_size].
    """
    if penalty_type == 'penalty':
        return tf.div((5. + tf.to_float(sequence_lengths))**penalty_factor,(5. + 1.)**penalty_factor)
    elif penalty_type == 'rerank':
        return tf.cast(sequence_lengths,tf.float32)


def cal_length_penalty(log_probs, sequence_lengths, penalty_type='penalty', penalty_factor=0.6):
    """Calculates scores for beam search hypotheses.
    """
    # Calculate the length penality
    length_penality_ = length_penalty(
        sequence_lengths=sequence_lengths,
        penalty_type = penalty_type,
        penalty_factor=penalty_factor)

    score = log_probs / length_penality_
    return score

def beam_attention_decoder(decoder_inputs,
                          initial_state,
                          attention_states,
                          cell,
                          embedding,
                          output_size=None,
                          num_heads=1,
                          loop_function=None,
                          dtype=None,
                          scope=None,
                          initial_state_attention=False, 
                          output_projection=None, 
                          beam_size=10,
                          length_penalty=None,
                          length_penalty_factor=0.6):
    if not decoder_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention decoder.")
    if not attention_states.get_shape()[1:2].is_fully_defined():
        raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                         % attention_states.get_shape())
    if output_size is None:
        output_size = cell.output_size

    with variable_scope.variable_scope(scope or "attention_decoder", dtype=dtype) as scope:
        dtype = scope.dtype
        batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
        attn_length = attention_states.get_shape()[1].value
        if attn_length is None:
            attn_length = array_ops.shape(attention_states)[1]
        attn_size = attention_states.get_shape()[2].value

        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = array_ops.reshape(attention_states, [-1, attn_length, 1, attn_size])
        hidden_features = []
        v = []
        attention_vec_size = attn_size  # Size of query vectors for attention.
        for a in range(num_heads):
            k = variable_scope.get_variable("AttnW_%d" % a, [1, 1, attn_size, attention_vec_size])
            hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
            v.append(variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size]))

        state = []
        # expand encoder last hidden layer to beam_size dimensions,因為decoder階段的batch_size = beam_size。 
        # initial_state is a list，RNN有多少層就有多少個element，each element is a LSTMStateTuple，include h,c state
        # 所以要將其擴展成beam_size維，其實是把c和h擴展，最後再合成LSTMStateTuple就可以了
        ##print('initial_state: ',initial_state,len(initial_state))
        for layers in initial_state:
            ##print('layers: ',layers,dir(layers))
            #c = [layers.c] * beam_size
            h = [layers] * beam_size
            #c = array_ops.concat(c, 0)
            h = array_ops.concat(h, 0)
            #state.append(LSTMStateTuple(c, h))
            state.append(h)
        state = tuple(state)
        ##print('state: ',state)
        # state_size = int(initial_state.get_shape().with_rank(2)[1])
        # states = []
        # for kk in range(beam_size):
        #     states.append(initial_state)
        # state = array_ops.concat(states, 0)
        # state = initial_state

        def attention(query):
            ds = []  # Results of attention reads will be stored here.
            if nest.is_sequence(query):  # If the query is a tuple, flatten it.
                query_list = nest.flatten(query)
                for q in query_list:  # Check that ndims == 2 if specified.
                    ndims = q.get_shape().ndims
                    if ndims:
                        assert ndims == 2
                query = array_ops.concat(query_list, 1)
            for a in range(num_heads):
                with variable_scope.variable_scope("Attention_%d" % a):
                    y = linear(query, attention_vec_size, True)
                    y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])
                    a = nn_ops.softmax(s)
                    # Now calculate the attention-weighted vector d.
                    d = math_ops.reduce_sum(array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
                    ds.append(array_ops.reshape(d, [-1, attn_size]))
            return ds

        outputs = []
        prev = None
        # attention也要定義成beam_size維的tensor
        batch_attn_size = array_ops.stack([beam_size, attn_size]) #(10,256)
        #print('batch_attn_size: ',batch_attn_size)
        attns = [array_ops.zeros(batch_attn_size, dtype=dtype) for _ in xrange(num_heads)]
        #print('attns: ', attns)
        for a in attns:  # Ensure the second shape of attention vectors is set.
            a.set_shape([None, attn_size])
            ##print('a: ',a)
        if initial_state_attention:
            attns = attention(initial_state)

        log_beam_probs, beam_path, beam_symbols = [], [], []
        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of decoder_inputs.
            if i == 0:
                #when i=0，input一個batch_szie=beam_size的tensor，且each element都是<GO>
                inp = tf.nn.embedding_lookup(embedding, tf.constant(1, dtype=tf.int32, shape=[beam_size]))

            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i, log_beam_probs, beam_path, beam_symbols)
            # Merge input and previous attentions into one vector of the right size.
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)
            inputs = [inp] + attns
            x = linear(inputs, input_size, True)
            #print('input_size: ',i, input_size)
            #print('[inp] + attns: ',i, [inp] + attns)
            #print('x: ',i, x)
            # Run the RNN.
            cell_output, state = cell(x, state)
            # Run the attention mechanism.
            if i == 0 and initial_state_attention:
                with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True):
                    attns = attention(state)
            else:
                attns = attention(state)

            with variable_scope.variable_scope("AttnOutputProjection"):
                inputs = [cell_output] + attns
                output = linear(inputs, output_size, True)
                #print('[cell_output] + attns: ',i,[cell_output] + attns)
                #print('output: ',i,output)
            if loop_function is not None:
                prev = output
            #outputs.append(tf.argmax(nn_ops.xw_plus_b(output, output_projection[0], output_projection[1]), axis=1))
            outputs.append(output)
            ##print('outpus[-1]: ', outputs[-1])

    #return outputs, state, tf.reshape(array_ops.concat(beam_path, 0), [-1, beam_size]), tf.reshape(tf.concat(beam_symbols, 0),[-1, beam_size])
    #log_beam_probs = tf.reshape(tf.concat(log_beam_probs,0),[-1,beam_size])
    #log_beam_probs_sum = math_ops.reduce_sum(log_beam_probs,axis=0) 
    log_beam_probs = tf.concat(log_beam_probs,1)
    beam_symbols = list(map(lambda x:tf.reshape(x,[-1,1]),beam_symbols))
    beam_symbols = tf.concat(beam_symbols,1)
    if length_penalty:
        log_beam_probs, sequence_lengths = mask_special_tag(beam_symbols,log_beam_probs,beam_size,log_beam_probs.shape[1],index=2)
        log_beam_probs_sum = math_ops.reduce_sum(log_beam_probs,axis=1)
        log_beam_probs = cal_length_penalty(log_beam_probs_sum, sequence_lengths, penalty_type=length_penalty, penalty_factor=length_penalty_factor)
    else:
        log_beam_probs_sum = math_ops.reduce_sum(log_beam_probs,axis=1)
    print('log_beam_probs: ',log_beam_probs)
    print('log_beam_probs_sum: ',log_beam_probs_sum)
    log_beam_probs_max_id = math_ops.argmax(log_beam_probs_sum) 
    _,log_beam_probs_ids_rev = tf.nn.top_k(log_beam_probs_sum,beam_size)
    print('log_beam_probs_ids_rev: ',log_beam_probs_ids_rev)
    print('--------')
    log_beam_probs_ids_rev = tf.reshape(log_beam_probs_ids_rev,[-1,1])
    #outputs = tf.gather_nd(outputs,log_beam_probs_ids_rev)
    #print('log_beam_probs_ids_rev: ',log_beam_probs_ids_rev)
    #print('outputs: ',outputs,len(outputs))
    #for out in outputs:
    #    print('out: ',tf.gather_nd(out,log_beam_probs_ids_rev))
    outputs = [tf.gather_nd(out,log_beam_probs_ids_rev) for out in outputs]
    if not bool(FLAGS.debug):
        outputs = [tf.reshape(logit[[log_beam_probs_max_id]],[1,-1]) for logit in outputs]
    #return outputs, state, tf.reshape(array_ops.concat(beam_path, 0), [-1, beam_size]), tf.reshape(tf.concat(beam_symbols, 0),[-1, beam_size]), tf.reshape(tf.concat(log_beam_probs,0),[-1,beam_size])
    return outputs, state

def embedding_attention_decoder(decoder_inputs,
                                initial_state,
                                attention_states,
                                cell,
                                num_symbols,
                                embedding_size,
                                embedding,
                                num_heads=1,
                                output_size=None,
                                output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False, 
                                beam_search=True, 
                                beam_size=10, 
                                loop=None,
                                schedule_sampling=False,
                                sampling_probability=None,
                                length_penalty=None,
                                length_penalty_factor=0.6):

    if output_size is None:
        output_size = cell.output_size
    if output_projection is not None:
        proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
        proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    with variable_scope.variable_scope(scope or "embedding_attention_decoder", dtype=dtype) as scope:
        #with tf.variable_scope('beam_search',reuse=tf.AUTO_REUSE): 
        emb_inp = [embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]
        # beam_search only happen in test mode
        if beam_search:
            #embedding = variable_scope.get_variable("embedding", [num_symbols, embedding_size])
            loop_function = _extract_beam_search(embedding, beam_size, num_symbols, embedding_size, output_projection)
            return beam_attention_decoder(
                emb_inp, 
                initial_state, 
                attention_states, 
                cell, 
                embedding, 
                output_size=output_size,
                num_heads=num_heads, 
                loop_function=loop_function,
                initial_state_attention=initial_state_attention, 
                output_projection=output_projection,
                beam_size=beam_size,
                length_penalty=length_penalty, 
                length_penalty_factor=length_penalty_factor)
        else:
            loop_function = None
            if feed_previous or schedule_sampling:
              if loop:
                loop_function = loop
              else:
                loop_function = _extract_argmax_and_embed(
                                embedding, output_projection,
                                update_embedding_for_previous)
            else:
              if schedule_sampling: 
                try:
                  tf.assert_type(sampling_probability,tf.float32) 
                except:
                  tf.assert_type(sampling_probability,tf.float64) 
            return attention_decoder(
                emb_inp,
                initial_state,
                attention_states,
                cell,
                output_size=output_size,
                num_heads=num_heads,
                loop_function=loop_function,
                initial_state_attention=initial_state_attention,
                schedule_sampling=schedule_sampling,
                sampling_probability=sampling_probability)


def embedding_attention_seq2seq(encoder_inputs,
                                decoder_inputs,
                                cell,
                                num_encoder_symbols,
                                num_decoder_symbols,
                                embedding_size,
                                embedding,
                                num_heads=1,
                                output_projection=None,
                                feed_previous=False,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False, 
                                beam_search=True,
                                beam_size=10,
                                loop=None,
                                schedule_sampling=False,
                                sampling_probability=None,
                                length_penalty=None,
                                length_penalty_factor=0.6):
    """Embedding sequence-to-sequence model with attention.

    This model first embeds encoder_inputs by a newly created embedding (of shape
    [num_encoder_symbols x input_size]). Then it runs an RNN to encode
    embedded encoder_inputs into a state vector. It keeps the outputs of this
    RNN at every step to use for attention later. Next, it embeds decoder_inputs
    by another newly created embedding (of shape [num_decoder_symbols x
    input_size]). Then it runs attention decoder, initialized with the last
    encoder state, on embedded decoder_inputs and attending to encoder outputs.

    Warning: when output_projection is None, the size of the attention vectors
    and variables will be made proportional to num_decoder_symbols, can be large.

    Args:
      encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
      decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
      cell: core_rnn_cell.RNNCell defining the cell function and size.
      num_encoder_symbols: Integer; number of symbols on the encoder side.
      num_decoder_symbols: Integer; number of symbols on the decoder side.
      embedding_size: Integer, the length of the embedding vector for each symbol.
      num_heads: Number of attention heads that read from attention_states.
      output_projection: None or a pair (W, B) of output projection weights and
        biases; W has shape [output_size x num_decoder_symbols] and B has
        shape [num_decoder_symbols]; if provided and feed_previous=True, each
        fed previous output will first be multiplied by W and added B.
      feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
        of decoder_inputs will be used (the "GO" symbol), and all other decoder
        inputs will be taken from previous outputs (as in embedding_rnn_decoder).
        If False, decoder_inputs are used as given (the standard decoder case).
      dtype: The dtype of the initial RNN state (default: tf.float32).
      scope: VariableScope for the created subgraph; defaults to
        "embedding_attention_seq2seq".
      initial_state_attention: If False (default), initial attentions are zero.
        If True, initialize the attentions from the initial state and attention
        states.

    Returns:
      A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
          shape [batch_size x num_decoder_symbols] containing the generated
          outputs.
        state: The state of each decoder cell at the final time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].
    """
    with variable_scope.variable_scope(scope or "embedding_attention_seq2seq", dtype=dtype) as scope:
        dtype = scope.dtype
        # Encoder.
        encoder_cell = copy.deepcopy(cell)
        encoder_cell = core_rnn_cell.EmbeddingWrapper(
            encoder_cell, 
            embedding_classes=num_encoder_symbols, 
            embedding_size=embedding_size)
        encoder_outputs, encoder_state = nn.static_rnn(encoder_cell, encoder_inputs, dtype=dtype)
        ##print('encoder_state: ',encoder_state)

        # First calculate a concatenation of encoder outputs to put attention on.
        top_states = [array_ops.reshape(e, [-1, 1, cell.output_size]) for e in encoder_outputs]
        attention_states = array_ops.concat(top_states, 1)

        # Decoder.
        output_size = None
        if output_projection is None:
            cell = core_rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
            output_size = num_decoder_symbols

        return embedding_attention_decoder(
            decoder_inputs,
            encoder_state,
            attention_states,
            cell,
            num_decoder_symbols,
            embedding_size,
            num_heads=num_heads,
            output_size=output_size,
            output_projection=output_projection,
            feed_previous=feed_previous,
            initial_state_attention=initial_state_attention, 
            embedding = embedding,
            beam_search=beam_search, 
            beam_size=beam_size,
            loop=loop,
            schedule_sampling=schedule_sampling,
            sampling_probability=sampling_probability,
            length_penalty=length_penalty,
            length_penalty_factor=length_penalty_factor)
