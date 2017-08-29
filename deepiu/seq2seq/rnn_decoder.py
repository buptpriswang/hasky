#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   rnn_decoder.py
#        \author   chenghuige  
#          \date   2016-12-24 00:00:05.991481
#   \Description  
# ==============================================================================
"""
dyanmic rnn_decoder support seq2seq, attention, pointer_network, generator+pointer/copy 

The diff for this RnnDecoder with google im2txt is im2txt assume a start word after image_embedding
so by doing this im2text or textsum all be the same input is embedding of <start_id> and inital_state
the first lstm_cell(image_embedding) outputs is discarded just use output_state as inital_state with <start_id>
so decode process always assume to first start with <start_id>

here we assume showandtell not add this <start_id> by default so image_embedding output assume to be first
decoded word, image_embedding is the first input ans inital_state is zero_state

#so not check if ok add additinal text start as first input before image...

and decode process always assume to have first_input and inital_state as input

TODO decode functions also accept first_input=None so just treat as im2txt
by this way general seq2seq also simpler as interactive beam decode start from start_id 
(input can start from start_id outside graph, not as right now must fetch from ingraph...
for like image_embedding as input you can not get it from lookup from embedding)  may be better(simpler) design
but one more train decode step, but since no additional decoding calc softmax not much perf hurt
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_sampled', 10000, 'num samples of neg word from vocab')
flags.DEFINE_boolean('log_uniform_sample', True, '')

flags.DEFINE_boolean('add_text_start', False, """if True will add <s> or 0 or GO before text 
                                              as first input before image, by default will be GO, 
                                              make sure add_text_start==True if you use seq2seq""")
flags.DEFINE_boolean('zero_as_text_start', False, """if add_text_start, 
                                                    here True will add 0 False will add <s>
                                                    0 means the loss from image to 0/pad not considered""")
flags.DEFINE_boolean('go_as_text_start', True, """ """)

flags.DEFINE_boolean('input_with_start_mark', False, """if input has already with <S> start mark""")
flags.DEFINE_boolean('input_with_end_mark', False, """if input has already with </S> end mark""")

flags.DEFINE_boolean('predict_with_end_mark', True, """if predict with </S> end mark""")

flags.DEFINE_float('length_normalization_factor', 1., """If != 0, a number x such that captions are
        scored by logprob/length^x, rather than logprob. This changes the
        relative scores of captions depending on their lengths. For example, if
        x > 0 then longer captions will be favored.  see tensorflow/models/im2text
        by default wil follow im2text set to 0
        Notice for train loss is same as set to 1, which is average loss per step
        2017-08-29 02:34:33 1:13:29 <p> pos [ moonshot  ] 0.208372 moon/shot/ /gd/</p>   #here is prob average per step
        2017-08-29 02:34:33 1:13:29 <p> gen:[ moon/ /shot/ /shot/ /<UNK>/</S> ] 0.000000 </p>
        2017-08-29 02:34:33 1:13:29 <p> gen_beam_0:[ moon/ /shot/ /shot/ </S> ] 0.64 </p> #here is also average per step 
        """)

flags.DEFINE_boolean('predict_use_prob', True, 'if True then use exp(logprob) and False will direclty output logprob')
flags.DEFINE_boolean('predict_no_sample', False, 'if True will use exact loss')
flags.DEFINE_integer('predict_sample_seed', 0, '')

flags.DEFINE_boolean('use_attention', False, 'wether to use attention for decoder')
flags.DEFINE_boolean('alignment_history', False, '')
flags.DEFINE_string('attention_option', 'luong', 'luong or bahdanau, luong seems faster and slightly better when visualizing alignments')

flags.DEFINE_integer('beam_size', 10, 'for seq decode beam search size')
flags.DEFINE_integer('decode_max_words', 0, 'if 0 use TEXT_MAX_WORDS from conf.py otherwise use decode_max_words')
#----notice decdoe_copy and decode_use_alignment all means gen_only mode, then restrict decode by copy ! not using copy to train
#--like copy_only and gen_copy(mix copy and gen)
flags.DEFINE_boolean('decode_copy', False, 'if True rstrict to use only input words(copy mode)')
flags.DEFINE_boolean('decode_use_alignment', False, '')

flags.DEFINE_boolean('gen_only', True, 'nomral seq2seq or seq2seq with attention')
#TODO have not finished copy <unk> words in input_text 
flags.DEFINE_boolean('copy_only', False, '''if True then only copy mode using attention, copy also means pointer, this is like 
                                            <pointer networks> used in seq2seq generation''')
flags.DEFINE_boolean('gen_copy', False, '''mix gen and copy, just add two logits(competes softmax) this is like 
                                         <Incorporating Copying Mechanism in Sequence-to-Sequence Learning>''')
#TODO much slower.. 1.5 vs 2.7 batch/s then gen_copy sparse_softmax_cross_entorpy much faster then sofmtax then sum loss ?
#TODO how is sparse_softmax_cross_entropy implemented
flags.DEFINE_boolean('gen_copy_switch', False, '''mix gen and copy, using gen or copy switch gen_probablity, 
                                                  this is like <pointing unknown words>, 
                                                  <'Get To The Point: Summarization with Pointer-Generator Networks>''')

flags.DEFINE_boolean('switch_after_softmax', True, '')

#TODO support feed_prev training mode for dynamic rnn decode
flags.DEFINE_boolean('feed_prev', False, 'wether use feed_prev mode for rnn decode during training also')

import functools
import melt 
from deepiu.util import vocabulary
from deepiu.seq2seq.decoder import Decoder
from tensorflow.python.util import nest

class SeqDecodeMethod:
  greedy = 0
  sample = 1
  full_sample = 2
  beam = 3         # ingraph beam search
  beam_search = 4  # outgraph beam search/ interactve beam search

class RnnDecoder(Decoder):
  def __init__(self, is_training=True, is_predict=False):
    self.scope = 'rnn'
    self.is_training = is_training 
    self.is_predict = is_predict

    assert not (FLAGS.decode_copy and FLAGS.decode_use_alignment)

    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size()
    self.vocab_size = vocab_size
    
    self.end_id = vocabulary.end_id()

    self.start_id = None
    self.get_start_id()

    assert self.end_id != vocabulary.vocab.unk_id(), 'input vocab generated without end id'

    self.emb_dim = emb_dim = FLAGS.emb_dim

    #--- for perf problem here exchange w_t and w https://github.com/tensorflow/tensorflow/issues/4138
    self.num_units = num_units = FLAGS.rnn_hidden_size
    with tf.variable_scope('output_projection'):
      self.w_t = melt.variable.get_weights_truncated('w_t',  
                                             [vocab_size, num_units], 
                                             stddev=FLAGS.weight_stddev) 
      #weights
      self.w = tf.transpose(self.w_t)
      #biases
      self.v = melt.variable.get_weights_truncated('v', 
                                             [vocab_size], 
                                             stddev=FLAGS.weight_stddev) 

    #TODO https://github.com/tensorflow/tensorflow/issues/6761  tf 1.0 will fail if not scope='rnn' the same as when using self.cell...
   
    self.cell = melt.create_rnn_cell( 
      num_units=num_units,
      is_training=is_training, 
      keep_prob=FLAGS.keep_prob, 
      num_layers=FLAGS.num_layers, 
      cell_type=FLAGS.cell)

    self.num_sampled = num_sampled = FLAGS.num_sampled if not (is_predict and FLAGS.predict_no_sample) else 0
    #self.softmax_loss_function is None means not need sample
    self.softmax_loss_function = None 
    if FLAGS.gen_only:
      self.softmax_loss_function = melt.seq2seq.gen_sampled_softmax_loss_function(num_sampled, 
                                                                                  self.vocab_size, 
                                                                                  weights=self.w_t,
                                                                                  biases=self.v,
                                                                                  log_uniform_sample=FLAGS.log_uniform_sample,
                                                                                  is_predict=self.is_predict,
                                                                                  sample_seed=FLAGS.predict_sample_seed,
                                                                                  vocabulary=vocabulary)

    if FLAGS.use_attention:
      print('----attention_option:', FLAGS.attention_option)
    if FLAGS.gen_copy_switch or FLAGS.gen_copy or FLAGS.copy_only:
      assert FLAGS.use_attention is True, 'must use attention if not gen_only mode seq2seq'
      FLAGS.gen_only = False
      if FLAGS.gen_copy_switch:
        print('-------gen copy switch mode!')
        FLAGS.gen_copy = False
        FLAGS.copy_only = False
      elif FLAGS.gen_copy:
        print('-------gen copy mode !')
        FLAGS.copy_only = False
      else:
        print('-------copy only mode !')
    else:
      print('--------gen only mode')

    #if use copy mode use score as alignment(no softmax)
    self.score_as_alignment = False if FLAGS.gen_only else True

    #gen only output_fn
    self.output_fn = lambda cell_output: melt.dense(cell_output, self.w, self.v)

    def copy_output(indices, batch_size, cell_output, cell_state):
      alignments = cell_state.alignments
      updates = alignments
      return tf.scatter_nd(indices, updates, shape=[batch_size, self.vocab_size])

    self.copy_output_fn = copy_output 

    #one problem is big memory for large vocabulary
    def gen_copy_output(indices, batch_size, cell_output, cell_state):
      gen_logits = self.output_fn(cell_output)
      copy_logits = copy_output(indices, batch_size, cell_output, cell_state)
      
      if FLAGS.gen_copy_switch:
          gen_probability = cell_state.gen_probability 
          #[batch_size, 1] * [batch_size, vocab_size]
          if FLAGS.switch_after_softmax:
            return gen_probability * tf.nn.softmax(gen_logits) + (1 - gen_probability) * tf.nn.softmax(copy_logits)
          else:
            return gen_probability * gen_logits + (1 - gen_probability) * copy_logits
      else:
        return gen_logits + copy_logits
        
    self.gen_copy_output_fn = gen_copy_output


    def gen_copy_output_train(time, indices, targets, sampled_values, batch_size, cell_output, cell_state):
      if self.softmax_loss_function is not None:
        labels = tf.slice(targets, [0, time], [-1, 1])

        sampled, true_expected_count, sampled_expected_count = sampled_values
        sampled_values = \
          sampled, tf.slice(tf.reshape(true_expected_count, [batch_size, -1]), [0, time], [-1, 1]), sampled_expected_count

        sampled_ids, sampled_logits = melt.nn.compute_sampled_ids_and_logits(
                                            weights=self.w_t, 
                                            biases=self.v, 
                                            labels=labels, 
                                            inputs=cell_output, 
                                            num_sampled=self.num_sampled, 
                                            num_classes=self.vocab_size, 
                                            sampled_values=sampled_values,
                                            remove_accidental_hits=False)
        gen_indices = melt.batch_values_to_indices(tf.to_int32(sampled_ids))
        gen_logits = tf.scatter_nd(gen_indices, sampled_logits, shape=[batch_size, self.vocab_size])
      else:
        gen_logits = self.output_fn(cell_output)
      
      copy_logits = copy_output(indices, batch_size, cell_output, cell_state)
      
      if FLAGS.gen_copy_switch:
          #gen_copy_switch == True. 
          gen_probability = cell_state.gen_probability
          if FLAGS.switch_after_softmax:
            return gen_probability * tf.nn.softmax(gen_logits) + (1 - gen_probability) * tf.nn.softmax(copy_logits)
          else:
            return gen_probability * gen_logits + (1 - gen_probability) * copy_logits
      else:
        return gen_logits + copy_logits

    self.gen_copy_output_train_fn = gen_copy_output_train

    
  def sequence_loss(self, sequence, 
                    initial_state=None, attention_states=None, 
                    input=None,
                    input_text=None,
                    exact_prob=False, exact_loss=False,
                    emb=None):
    """
    for general seq2seq input is None, sequence will pad <GO>, inital_state is last state from encoder
    for img2text/showandtell input is image_embedding, inital_state is None/zero set
    TODO since exact_porb and exact_loss same value, may remove exact_prob
    NOTICE! assume sequence to be padded by zero and must have one instance full length(no zero!)
    """
    if emb is None:
      emb = self.emb

    is_training = self.is_training
    batch_size = melt.get_batch_size(sequence)
    
    sequence, sequence_length = melt.pad(sequence,
                                     start_id=self.get_start_id(),
                                     end_id=self.get_end_id())

    #[batch_size, num_steps - 1, emb_dim], remove last col
    inputs = tf.nn.embedding_lookup(emb, sequence[:,:-1])

    if is_training and FLAGS.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, FLAGS.keep_prob)
    
    #inputs[batch_size, num_steps, emb_dim] input([batch_size, emb_dim] -> [batch_size, 1, emb_dim]) before concat
    if input is not None:
      #used like showandtell where image_emb is as input, additional to sequence
      inputs = tf.concat([tf.expand_dims(input, 1), inputs], 1)
    else:
      #common usage input is None, sequence as input, notice already pad <GO> before using melt.pad
      sequence_length -= 1
      sequence = sequence[:, 1:]
    
    if self.is_predict:
      #---only need when predict, since train input already dynamic length, NOTICE this will improve speed a lot
      num_steps = tf.to_int32(tf.reduce_max(sequence_length))
      sequence = sequence[:, :num_steps]
      inputs = inputs[:, :num_steps, :]

    tf.add_to_collection('sequence', sequence)
    tf.add_to_collection('sequence_length', sequence_length)

    #[batch_size, num_steps]
    targets = sequence

    if attention_states is None:
      cell = self.cell 
    else:
      cell = self.prepare_attention(attention_states, 
                                    initial_state=initial_state, 
                                    score_as_alignment=self.score_as_alignment)
      initial_state = None
    state = cell.zero_state(batch_size, tf.float32) if initial_state is None else initial_state

    #TODO: hack here add FLAGS.predict_no_sample just for Seq2seqPredictor exact_predict
    softmax_loss_function = self.softmax_loss_function
    if self.is_predict and (exact_prob or exact_loss):
      softmax_loss_function = None

    if FLAGS.gen_only:
      #gen only mode
      #for attention wrapper can not use dynamic_rnn if aligments_history=True TODO see pointer_network in application seems ok.. why
      outputs, state = tf.nn.dynamic_rnn(cell, 
                                         inputs, 
                                         initial_state=state, 
                                         sequence_length=sequence_length,
                                         dtype=tf.float32,
                                         scope=self.scope)     

      #--------below is ok but slower then dynamic_rnn 3.4batch -> 3.1 batch/s
      #helper = melt.seq2seq.TrainingHelper(inputs, tf.to_int32(sequence_length))
      ##helper = tf.contrib.seq2seq.TrainingHelper(inputs, tf.to_int32(sequence_length))
      #my_decoder = melt.seq2seq.BasicTrainingDecoder(
      ##my_decoder = tf.contrib.seq2seq.BasicDecoder(
      ##my_decoder = melt.seq2seq.BasicDecoder(
      #      cell=cell,
      #      helper=helper,
      #      initial_state=state)
      ##outputs, state, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder, scope=self.scope)
      #outputs, state, _ = melt.seq2seq.dynamic_decode(my_decoder, scope=self.scope)
      ##outputs = outputs.rnn_output
    else:
    	#---copy only or gen copy
      helper = melt.seq2seq.TrainingHelper(inputs, tf.to_int32(sequence_length))

      indices = melt.batch_values_to_indices(tf.to_int32(input_text))
      if FLAGS.copy_only:
        output_fn = lambda cell_output, cell_state: self.copy_output_fn(indices, batch_size, cell_output, cell_state)
      else:
        #gen_copy right now, not use switch ? gen_copy and switch?
        sampled_values = None 
        #TODO CHECK this is it ok? why train and predict not equal and score/exact score same? FIXME 
        #need first debug why score and exact score is same ? score should be the same as train! TODO 
        #sh ./inference/infrence-score.sh to reproduce
        #now just set num_sampled = 0 for safe, may be here train also not correct FIXME 
        if softmax_loss_function is not None:
          sampled_values = tf.nn.log_uniform_candidate_sampler(true_classes=tf.reshape(targets, [-1, 1]),
                                                    num_true=1,
                                                    num_sampled=self.num_sampled,
                                                    unique=True,
                                                    range_max=self.vocab_size)
          #TODO since perf of sampled version here is ok not modify now, but actually in addtional to sampled_values
          #sampled_w, sampled_b can also be pre embedding lookup, may imporve not much
        output_fn = lambda time, cell_output, cell_state: self.gen_copy_output_train_fn(
                                         time, indices, targets, sampled_values, batch_size, cell_output, cell_state)


      my_decoder = melt.seq2seq.BasicTrainingDecoder(
        cell=cell,
        helper=helper,
        initial_state=state,
        vocab_size=self.vocab_size,
        output_fn=output_fn)
      outputs, state, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder, scope=self.scope)
      #outputs, state, _ = melt.seq2seq.dynamic_decode(my_decoder, scope=self.scope)

    tf.add_to_collection('outputs', outputs)
    
    if not FLAGS.gen_only:
      logits = outputs
      softmax_loss_function = None
    elif softmax_loss_function is not None:
      logits = outputs
    else:
      #--softmax_loss_function is None means num_sample = 0 or exact_loss or exact_prob
      #[batch_size, num_steps, num_units] * [num_units, vocab_size]
      # -> logits [batch_size, num_steps, vocab_size] (if use exact_predict_loss)
      #or [batch_size * num_steps, vocab_size] by default flatten=True
      #this will be fine for train [batch_size * num_steps] but not good for eval since we want 
      #get score of each instance also not good for predict
      keep_dims = exact_prob or exact_loss or (not self.is_training)
      logits = melt.batch_matmul_embedding(outputs, self.w, keep_dims=keep_dims) + self.v
      if not keep_dims:
        targets = tf.reshape(targets, [-1])

    tf.add_to_collection('logits', logits)

    mask = tf.cast(tf.sign(targets), dtype=tf.float32)

    if FLAGS.gen_copy_switch and FLAGS.switch_after_softmax:
      #TODO why need more gpu mem ? ...  do not save logits ? just calc loss in output_fn ?
      #batch size 256
      #File "/home/gezi/mine/hasky/util/melt/seq2seq/loss.py", line 154, in body
      #step_logits = logits[:, i, :]
      #ResourceExhaustedError (see above for traceback): OOM when allocating tensor with shape[256,21,33470]
      num_steps = tf.shape(targets)[1]

      loss = melt.seq2seq.exact_predict_loss(logits, targets, mask, num_steps, 
                                             need_softmax=False, 
                                             average_across_timesteps=not self.is_predict,
                                             batch_size=batch_size)
    elif self.is_predict and exact_prob:
      #generate real prob for sequence
      #for 10w vocab textsum seq2seq 20 -> 4 about 
      loss = melt.seq2seq.exact_predict_loss(logits, targets, mask, 
                                             num_steps, batch_size=batch_size,
                                             average_across_timesteps=False)
    elif self.is_predict and exact_loss: 
      #force no sample softmax loss, the diff with exact_prob is here we just use cross entropy error as result not real prob of seq
      #NOTICE using time a bit less  55 to 57(prob), same result with exact prob and exact score
      #but 256 vocab sample will use only about 10ms    
      loss = melt.seq2seq.sequence_loss_by_example(logits, targets, weights=mask, 
                                                   average_across_timesteps=False)
    else:
      #loss [batch_size,] 
      loss = melt.seq2seq.sequence_loss_by_example(
          logits,
          targets,
          weights=mask,
          average_across_timesteps=not self.is_predict, #train must average, other wise long sentence big loss..
          softmax_loss_function=softmax_loss_function)
    
    #mainly for compat with [bach_size, num_losses] here may be [batch_size * num_steps,] if is_training and not exact loss/prob
    loss = tf.reshape(loss, [-1, 1])

    self.ori_loss = loss
    if self.is_predict:
      #note use avg_loss not to change loss pointer, avg_loss is same as average time step=True is length_normalize_fator=1.0
      avg_loss = self.normalize_length(loss, sequence_length)
      return avg_loss
    
    #if not is_predict loss is averaged per time step else not but avg loss will average it 
    return loss

  def generate_sequence_greedy(self, input, max_words, 
                        initial_state=None, attention_states=None,
                        convert_unk=True, 
                        input_text=None,
                        emb=None):
    """
    this one is using greedy search method
    for beam search using generate_sequence_by_beam_search with addditional params like beam_size
    """
    if emb is None:
      emb = self.emb

    batch_size = melt.get_batch_size(input)
    if attention_states is None:
      cell = self.cell 
    else:
      cell = self.prepare_attention(attention_states, 
      															initial_state=initial_state,
      															score_as_alignment=self.score_as_alignment)
      initial_state = None
    state = cell.zero_state(batch_size, tf.float32) if initial_state is None else initial_state

    helper = melt.seq2seq.GreedyEmbeddingHelper(embedding=emb, first_input=input, end_token=self.end_id)
 
    if FLAGS.gen_only:
      output_fn = self.output_fn 
    else:
      indices = melt.batch_values_to_indices(tf.to_int32(input_text))
      if FLAGS.copy_only:
        output_fn_ = self.copy_output_fn
      else:
        output_fn_ = self.gen_copy_output_fn
      output_fn = lambda cell_output, cell_state: output_fn_(indices, batch_size, cell_output, cell_state)

    my_decoder = melt.seq2seq.BasicDecoder(
          cell=cell,
          helper=helper,
          initial_state=state,
          vocab_size=self.vocab_size,
          output_fn=output_fn)

    outputs, _, _ = melt.seq2seq.dynamic_decode(my_decoder, 
                                                maximum_iterations=max_words, 
                                                scope=self.scope)
    generated_sequence = outputs.sample_id
    #------like beam search return sequence, score
    return generated_sequence, tf.zeros([batch_size,])


  def generate_sequence_beam(self, input, max_words, 
                             initial_state=None, attention_states=None,
                             beam_size=5, convert_unk=True,
                             length_normalization_factor=0., 
                             input_text=None,
                             input_text_length=None,
                             emb=None):
    """
    beam dcode means ingraph beam search
    return top (path, score)
    """
    if emb is None:
      emb = self.emb

    def loop_function(i, prev, state, decoder):
      prev, state = decoder.take_step(i, prev, state)
      next_input = tf.nn.embedding_lookup(emb, prev)
      return next_input, state

    batch_size = melt.get_batch_size(input)
    
    if initial_state is not None:
      initial_state = nest.map_structure(lambda x: tf.contrib.seq2seq.tile_batch(x, beam_size), initial_state)
    if attention_states is None:
      cell = self.cell 
    else:
      attention_states = tf.contrib.seq2seq.tile_batch(attention_states, beam_size)
      #print('tiled_attention_states', attention_states, 'tiled_initial_state', initial_state)
      cell = self.prepare_attention(attention_states, 
      	                            initial_state=initial_state,
      	                            score_as_alignment=self.score_as_alignment)
      initial_state = None

    state = cell.zero_state(batch_size * beam_size, tf.float32) \
              if initial_state is None else initial_state
        
    if FLAGS.gen_only:
      output_fn = self.output_fn 
    else:
      input_text = tf.contrib.seq2seq.tile_batch(input_text, beam_size)
      batch_size = batch_size * beam_size
      indices = melt.batch_values_to_indices(tf.to_int32(input_text))
      if FLAGS.copy_only:
        output_fn_ = self.copy_output_fn
      else:
        output_fn_ = self.gen_copy_output_fn
      output_fn = lambda cell_output, cell_state: output_fn_(indices, batch_size, cell_output, cell_state)


    ##TODO to be safe make topn the same as beam size
    return melt.seq2seq.beam_decode(input, max_words, state, 
                                    cell, loop_function, scope=self.scope,
                                    beam_size=beam_size, done_token=vocabulary.vocab.end_id(), 
                                    output_fn=output_fn,
                                    length_normalization_factor=length_normalization_factor,
                                    topn=beam_size,
                                    need_softmax=not(FLAGS.gen_copy_switch and FLAGS.switch_after_softmax))

    ##---dynamic beam search decoder can run but seems not correct or good reuslt, maybe bug TODO
    ##check with one small and simple testcase 
    #bsd = melt.seq2seq.BeamSearchDecoder(
    #          cell=cell,
    #          embedding=emb,
    #          first_input=input,
    #          end_token=self.end_id,
    #          initial_state=state,
    #          beam_width=beam_size,
    #          vocab_size=self.vocab_size,
    #          output_fn=self.output_fn,
    #          length_penalty_weight=0.0)

    ##max_words = 2
    #outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(bsd, maximum_iterations=max_words, scope=self.scope)

    ##return outputs.predicted_ids, outputs.beam_search_decoder_output.scores
    #paths = tf.transpose(outputs.predicted_ids, [0, 2, 1])
    ##paths = tf.transpose(outputs.beam_search_decoder_output.predicted_ids, [0, 2, 1])
    ##paths = tf.transpose(outputs.beam_search_decoder_output.parent_ids, [0, 2, 1])

    ##scores = tf.zeros([batch_size, beam_size])
    #scores = tf.transpose(outputs.beam_search_decoder_output.scores, [0, 2, 1])
    #scores = tf.exp(scores)
    #scores = scores[:, :, -1]

    #tf.add_to_collection('preids', outputs.beam_search_decoder_output.predicted_ids)
    #tf.add_to_collection('paids', outputs.beam_search_decoder_output.parent_ids)

    #return paths, scores


  def generate_sequence_beam_search(self, input, max_words=None, 
                                  initial_state=None, attention_states=None,
                                  beam_size=10, convert_unk=True,
                                  length_normalization_factor=0., 
                                  input_text=None,
                                  input_text_length=None,
                                  emb=None):
    """
    outgraph beam search, input should be one instance only batch_size=1
    max_words actually not used here... for it is determined outgraph..
    return top (path, score)
    TODO this is hacky, first step attention_state, input , state all size 1,
    then should be attention_state 1, input, state size is beam_size,
    also might be less then beam_size.. if not possible to find beam_size un done
    """
    if emb is None:
      emb = self.emb
    
    tf.add_to_collection('beam_search_beam_size', tf.constant(beam_size))
    if input_text is not None:
      if FLAGS.decode_copy:
        input_text = tf.squeeze(input_text)
        input_text_length = tf.to_int32(tf.squeeze(input_text_length))
        input_text = input_text[0:input_text_length]
        input_text, _ = tf.unique(input_text)
        input_text_length = tf.shape(input_text)[-1]
        #sort from small to large
        #input_text, _ = -tf.nn.top_k(-input_text, input_text_length)
        #TODO may be need to be input_text_length, so as to do more decode limit out graph like using trie!
        beam_size = tf.minimum(beam_size, input_text_length)
      elif FLAGS.decode_use_alignment:
        input_text = tf.squeeze(input_text)
        input_text_length = tf.to_int32(tf.squeeze(input_text_length))
        input_text = input_text[0:input_text_length]
        input_text_length = tf.shape(input_text)[-1]
        beam_size = tf.minimum(beam_size, input_text_length)
      else:
        if FLAGS.gen_only:
          input_text = None

    batch_size = melt.get_batch_size(input)
    if attention_states is None:
      cell = self.cell 
    else:
      cell = self.prepare_attention(attention_states, 
      	                            initial_state=initial_state,
      	                            score_as_alignment=self.score_as_alignment)
      initial_state = None
    state = cell.zero_state(batch_size, tf.float32) \
        if initial_state is None else initial_state
    
    ##--TODO hard.. since need to reuse to share ValueError: 
    ##Variable seq2seq/main/decode/memory_layer/kernel already exists, disallowed. Did you mean to set reuse=True in VarScope?
    ##another way to solve is always using tiled_batch attention_states and state, the first step will choose from only first beam
    ##will not all solve the problem since feed data might be less than beam size, so attention states always be 1 is safe
    #cell2 = self.prepare_attention(tf.contrib.seq2seq.tile_batch(attention_states, beam_size), reuse=True)

    first_state = state

    beam_search_step = functools.partial(self.beam_search_step, 
                                         beam_size=beam_size)

    #since before hack using generate_sequence_greedy, here can not set scope.reuse_variables
    #NOTICE inorder to use lstm which is in .../rnn/ nameapce here you must also add this scope to use the shared 
    with tf.variable_scope(self.scope) as scope:
      inital_attention, initial_state, initial_logprobs, initial_ids = \
            beam_search_step(input, state, cell, input_text=input_text)

      if attention_states is not None:
        tf.add_to_collection('beam_search_initial_alignments', tf.get_collection('attention_alignments')[-1])

      scope.reuse_variables()
      # In inference mode, use concatenated states for convenient feeding and
      # fetching.
      state_is_tuple = len(initial_state) == 2

      if state_is_tuple:
        initial_state = tf.concat(initial_state, 1, name="initial_state")
        state_size = sum(self.cell.state_size)
      else:
        state_size = self.cell.state_size

      #output is used only when use attention
      if attention_states is not None:
        initial_state = tf.concat([initial_state, inital_attention], 1, name="initial_attention_state")
        state_size += self.cell.output_size

      tf.add_to_collection('beam_search_initial_state', initial_state)
      tf.add_to_collection('beam_search_initial_logprobs', initial_logprobs)
      tf.add_to_collection('beam_search_initial_ids', initial_ids)


      input_feed = tf.placeholder(dtype=tf.int64, 
                                  shape=[None],  # batch_size
                                  name="input_feed")
      tf.add_to_collection('beam_search_input_feed', input_feed)
      input = tf.nn.embedding_lookup(emb, input_feed)


      # Placeholder for feeding a batch of concatenated states.
      state_feed = tf.placeholder(dtype=tf.float32,
                                  shape=[None, state_size],
                                  name="state_feed")
      tf.add_to_collection('beam_search_state_feed', state_feed)

      if attention_states is not None:
        state, attention = tf.split(state_feed, 
                                    [state_size - self.cell.output_size, self.cell.output_size], 
                                    axis=1)
      else:
        state = state_feed

      if state_is_tuple:
        state = tf.split(state, num_or_size_splits=2, axis=1)
      
      if attention_states is not None:
        state_ = first_state.clone(cell_state=state, attention=attention)
      else:
        state_ = state

      #--TODO here is not safe if change attention_wrapper, notice batch size of attention states is 1 
      #--but cell input and state is beam_size
      #attention, state, top_logprobs, top_ids = beam_search_step(input, state_, cell2)
      
      if input_text is not None and not FLAGS.decode_copy:
        input_text = tf.contrib.seq2seq.tile_batch(input_text, 
                                                    melt.get_batch_size(input)) 
  
      attention, state, top_logprobs, top_ids = beam_search_step(input, state_, cell, 
                                                                input_text=input_text)

      if state_is_tuple:
        # Concatentate the resulting state.
        state = tf.concat(state, 1, name="state")
      if attention_states is not None:
        state = tf.concat([state, attention], 1, name="attention_state")

      tf.add_to_collection('beam_search_state', state)
      tf.add_to_collection('beam_search_logprobs', top_logprobs)
      tf.add_to_collection('beam_search_ids', top_ids)

      #just same return like return path list, score list
      return tf.no_op(), tf.no_op()

  def beam_search_step(self, input, state, cell, beam_size, 
                       attention_construct_fn=None,
                       input_text=None):
    output, state = cell(input, state)

    if hasattr(state, 'alignments'):
      tf.add_to_collection('attention_alignments', state.alignments)
      tf.add_to_collection('beam_search_alignments', 
                            tf.get_collection('attention_alignments')[-1])

    #TODO: this step cause.. attenion decode each step after initalization still need input_text feed 
    #will this case attention_keys and attention_values to be recompute(means redo encoding process) each step?
    #can we avoid this? seems no better method, 
    #if enocding is slow may be feed attention_keys, attention_values each step
    if not FLAGS.decode_use_alignment:
      if FLAGS.gen_only:
        output_fn = self.output_fn 
        logits = output_fn(output)
      else:
        indices = melt.batch_values_to_indices(tf.to_int32(input_text))
        batch_size = melt.get_batch_size(input)

        if FLAGS.copy_only:
          output_fn_ = self.copy_output_fn
        else:
          output_fn_ = self.gen_copy_output_fn
        output_fn = lambda cell_output, cell_state: output_fn_(indices, batch_size, cell_output, cell_state)

        logits = output_fn(output, state)

      logprobs = tf.nn.log_softmax(logits)

      if FLAGS.decode_copy:
        logprobs = melt.gather_cols(logprobs, tf.to_int32(input_text))
    else:
      logits = state.alignments
      logits = scores[:,:tf.shape(input_text)[-1]]
      logprobs = tf.nn.log_softmax(logits)

    top_logprobs, top_ids = tf.nn.top_k(logprobs, beam_size)
    #------too slow... for transfering large data between py and c++ cost a lot!
    #top_logprobs, top_ids = tf.nn.top_k(logprobs, self.vocab_size)

    if input_text is not None and FLAGS.decode_copy:
      top_ids = tf.nn.embedding_lookup(input_text, top_ids)
    
    if hasattr(state, 'cell_state'):
      state = state.cell_state

    return output, state, top_logprobs, top_ids

  def prepare_attention(self, attention_states, initial_state=None,
                       sequence_length=None, alignment_history=False, 
                       output_alignment=False, score_as_alignment=False,
                       reuse=False):
    attention_option = FLAGS.attention_option  # can be "bahdanau"
    if attention_option is "bahdanau":
      create_attention_mechanism = melt.seq2seq.BahdanauAttention
    else:
      create_attention_mechanism = melt.seq2seq.LuongAttention
 
    #since dynamic rnn decoder also consider seq length, no need to mask again
    #memory_sequence_length can always set None

    #TODO better method to make default behavior as share? make_template
    #with tf.variable_scope('prepare_attention', reuse=reuse) as scope:
    attention_mechanism = create_attention_mechanism(
          num_units=self.num_units,
          memory=attention_states,
          memory_sequence_length=sequence_length) 

    AttentionWrapper = melt.seq2seq.AttentionWrapper 
    if FLAGS.gen_copy_switch:
        AttentionWrapper = melt.seq2seq.PointerAttentionWrapper
    cell = AttentionWrapper(
              self.cell,
              attention_mechanism,
              attention_layer_size=self.num_units,
              alignment_history=alignment_history,
              initial_cell_state=initial_state,
              output_alignment=output_alignment,
              score_as_alignment=score_as_alignment)
      #why still need reuse.. below not work?..
      #scope.reuse_variables()
    return cell

  def get_start_input(self, batch_size):
    start_input = melt.constants(self.start_id, [batch_size], tf.int32)
    return start_input

  def get_start_embedding_input(self, batch_size, emb=None):
    if emb is None:
      emb = self.emb
    start_input = self.get_start_input(batch_size)
    start_embedding_input = tf.nn.embedding_lookup(emb, start_input) 
    return start_embedding_input
 
  #always use this in predict mode
  def normalize_length(self, loss, sequence_length):
    sequence_length = tf.cast(sequence_length, tf.float32)
    #NOTICE must check if shape ok, [?,1] / [?,] will get [?,?]
    sequence_length = tf.reshape(sequence_length, [-1, 1])
    #notice for predict we always not average per step! so not need to turn it back
    #loss = loss * sequence_length 
    normalize_factor = tf.pow(sequence_length, FLAGS.length_normalization_factor)
    loss /= normalize_factor  
    return loss

  def get_start_id(self):
    if self.start_id is not None:
      return self.start_id

    start_id = None
    if not FLAGS.input_with_start_mark and FLAGS.add_text_start:
      if FLAGS.zero_as_text_start:
        start_id = 0
      elif FLAGS.go_as_text_start:
        start_id = vocabulary.go_id()
      else:
        start_id = vocabulary.start_id()
    self.start_id = start_id
    return start_id

  def get_end_id(self):
    if (FLAGS.input_with_end_mark or 
        (self.is_predict and not FLAGS.predict_with_end_mark)):
      return None 
    else:
      return self.end_id

