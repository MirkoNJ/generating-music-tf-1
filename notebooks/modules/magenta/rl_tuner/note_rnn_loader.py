# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines a class and operations for the MelodyRNN model.

Note RNN Loader allows a basic melody prediction LSTM RNN model to be loaded
from a checkpoint file, primed, and used to predict next notes.

This class can be used as the q_network and target_q_network for the RLTuner
class.

The graph structure of this model is similar to basic_rnn, but more flexible.
It allows you to either train it with data from a queue, or just 'call' it to
produce the next action.

It also provides the ability to add the model's graph to an existing graph as a
subcomponent, and then load variables from a checkpoint file into only that
piece of the overall graph.

These functions are necessary for use with the RL Tuner class.
"""

import os

import numpy as np
import tensorflow as tf
import pdb

from modules.magenta.common import sequence_example_lib
from modules.magenta.rl_tuner import rl_tuner_ops
from modules.magenta.shared import events_rnn_graph
from modules.magenta.music import melodies_lib
from modules.magenta.music import midi_io
from modules.magenta.music import sequences_lib
from modules.nikhil.MyFunctions import Input_Kernel, LSTM_Cell, LSTM_Layer, Conditional_Probability_Layer
from tensorflow.contrib.rnn import LSTMStateTuple


class NoteRNNLoader(object):
  """Builds graph for a Note RNN and instantiates weights from a checkpoint.

  Loads weights from a previously saved checkpoint file corresponding to a pre-
  trained basic_rnn model. Has functions that allow it to be primed with a MIDI
  melody, and allow it to be called to produce its predictions for the next
  note in a sequence.

  Used as part of the RLTuner class.
  """

  def __init__(self, graph, scope, checkpoint_dir, checkpoint_file=None,
               midi_primer=None, training_file_list=None, hparams=None,
               note_rnn_type='default'):
    """Initialize by building the graph and loading a previous checkpoint.

    Args:
      graph: A tensorflow graph where the MelodyRNN's graph will be added.
      scope: The tensorflow scope where this network will be saved.
      checkpoint_dir: Path to the directory where the checkpoint file is saved.
      checkpoint_file: Path to a checkpoint file to be used if none can be
        found in the checkpoint_dir
      midi_primer: Path to a single midi file that can be used to prime the
        model.
      training_file_list: List of paths to tfrecord files containing melody
        training data.
      hparams: A tf_lib.HParams object. Must match the hparams used to create
        the checkpoint file.
      note_rnn_type: If 'default', will use the basic LSTM described in the
        research paper. If 'basic_rnn', will assume the checkpoint is from a
        Magenta basic_rnn model.
    """
    self.graph = graph
    self.session = None
    self.scope = scope
    self.batch_size = 1
    self.num_timesteps = 1
    self.midi_primer = midi_primer
    self.note_rnn_type = note_rnn_type
    self.training_file_list = training_file_list
    self.checkpoint_dir = checkpoint_dir
    self.checkpoint_file = checkpoint_file
    self.hparams = tf.contrib.training.HParams(use_dynamic_rnn=True,
                                    batch_size=128,
                                    lr=0.0002,
                                    l2_reg=2.5e-5,
                                    clip_norm=5,
                                    initial_learning_rate=0.5,
                                    decay_steps=1000,
                                    decay_rate=0.85,
                                    rnn_layer_sizes_t=[128, 128],
                                    rnn_layer_sizes_n=[64, 64],
                                    midi_high = 108,
                                    midi_low = 21,
                                    output_keep_prob= 0.5,
                                    skip_first_n_losses=32,
                                    one_hot_length=rl_tuner_ops.MAX_NOTE-rl_tuner_ops.MIN_NOTE+rl_tuner_ops.NUM_SPECIAL_EVENTS,
                                    exponentially_decay_learning_rate=True)

    self.build_graph()
    self.state_value_t = self.get_zero_state_t()
    self.state_value_n = self.get_zero_state_n()
    # self.state_value_event = self.get_zero_state_event()
    self.Note_State_Batch_value = self.get_zero_Note_State()

    if midi_primer is not None:
      self.load_primer()

    self.variable_names = rl_tuner_ops.get_variable_names(self.graph,
                                                          self.scope)

    self.transpose_amount = 0

  def get_zero_state_t(self):
    """Gets an initial state of zeros of the appropriate size.

    Required size is based on the model's internal RNN cell.

    Returns:
      A matrix of batch_size x cell size zeros.
    """
    timewise_state_val=[]
    for i in range(len(self.hparams.rnn_layer_sizes_t)):
        c = np.zeros((self.batch_size * (self.hparams.midi_high + 1 - self.hparams.midi_low), self.hparams.rnn_layer_sizes_t[i])) #start every batch with zero state in LSTM time cells
        h = np.zeros((self.batch_size * (self.hparams.midi_high + 1 - self.hparams.midi_low), self.hparams.rnn_layer_sizes_t[i]))
        timewise_state_val.append(tf.contrib.rnn.LSTMStateTuple(h,c))
    timewise_state_val =  tuple(timewise_state_val)

    return timewise_state_val
  
  def get_zero_state_n(self):
    """Gets an initial state of zeros of the appropriate size.

    Required size is based on the model's internal RNN cell.

    Returns:
      A matrix of batch_size x cell size zeros.
    """

    notewise_state_val=[]
    for i in range(len(self.hparams.rnn_layer_sizes_n)):
        c = np.zeros((self.batch_size * self.num_timesteps, self.hparams.rnn_layer_sizes_n[i])) #start every batch with zero state in LSTM time cells
        h = np.zeros((self.batch_size * self.num_timesteps, self.hparams.rnn_layer_sizes_n[i]))
        notewise_state_val.append(tf.contrib.rnn.LSTMStateTuple(h,c))
    notewise_state_val =  tuple(notewise_state_val)

    return notewise_state_val

  # def get_zero_state_event(self):
  #   """Gets an initial state of zeros of the appropriate size.

  #   Required size is based on the model's internal RNN cell.

  #   Returns:
  #     A matrix of batch_size x cell size zeros.
  #   """
  #   event_state_val = []
  #   c = np.zeros((self.batch_size * (self.hparams.midi_high + 1 - self.hparams.midi_low), 100)) #start every batch with zero state in LSTM time cells
  #   h = np.zeros((self.batch_size * (self.hparams.midi_high + 1 - self.hparams.midi_low), 100))
  #   event_state_val.append(tf.contrib.rnn.LSTMStateTuple(h,c))
  #   event_state_val =  tuple(event_state_val)

  #   return event_state_val
  
  def get_zero_Note_State(self, batch_size=None):
    """Gets an initial state of zeros of the appropriate size.

    Required size is based on the model's internal RNN cell.

    Returns:
      A matrix of batch_size x cell size zeros.
    """
    if batch_size is None:
      batch_size = self.batch_size

    return np.zeros((batch_size, self.hparams.midi_high + 1 - self.hparams.midi_low, 1, 4))

  def get_input_batch(self, observation, time):
    """ Computes the input batch"""

    idx = np.where(observation==1)[0][0]
    if(time==0):
      self.Note_State_Batch_value = self.get_zero_Note_State()
    t = float(int(((time)%48) / 3))

    if idx == 0:
      input_batch = np.zeros((self.batch_size, self.hparams.midi_high + 1 - self.hparams.midi_low, 1, 4)) #Note Off event -> No notes played or articulated
    elif idx == 1:
      input_batch = self.Note_State_Batch_value # No event, no note is played, but if a note was played it is held
      input_batch[:,:,:,1] = np.zeros((self.batch_size, self.hparams.midi_high + 1 - self.hparams.midi_low, 1))
    else:
      input_batch = np.zeros((self.batch_size, self.hparams.midi_high + 1 - self.hparams.midi_low, 1, 4))
      input_batch[0, rl_tuner_ops.MIN_NOTE - self.hparams.midi_low - rl_tuner_ops.NUM_SPECIAL_EVENTS + idx,0,:] = [1, 1, 60, t]
  
    input_batch[:,:,:,3] = np.ones((self.batch_size, self.hparams.midi_high + 1 - self.hparams.midi_low, 1)) * t

    self.Note_State_Batch_value = input_batch
    return input_batch

  def restore_initialize_prime(self, session):
    """Saves the session, restores variables from checkpoint, primes model.

    Model is primed with its default midi file.

    Args:
      session: A tensorflow session.
    """
    self.session = session
    self.restore_vars_from_checkpoint(self.checkpoint_dir)
    self.prime_model()

  def initialize_and_restore(self, session):
    """Saves the session, restores variables from checkpoint.

    Args:
      session: A tensorflow session.
    """
    self.session = session
    self.restore_vars_from_checkpoint(self.checkpoint_dir)

  def initialize_new(self, session=None):
    """Saves the session, initializes all variables to random values.

    Args:
      session: A tensorflow session.
    """
    with self.graph.as_default():
      if session is None:
        self.session = tf.Session(graph=self.graph)
      else:
        self.session = session
      self.session.run(tf.initialize_all_variables())

  def get_variable_name_dict(self):
    """Constructs a dict mapping the checkpoint variables to those in new graph.

    Returns:
      A dict mapping variable names in the checkpoint to variables in the graph.
    """
    var_dict = dict()

    for var in self.variables():
      inner_name = rl_tuner_ops.get_inner_scope(var.name)
      inner_name = rl_tuner_ops.trim_variable_postfixes(inner_name)
      if any(name in var.name for name in ['fully_connected', 'Adam', 'rnn_event']): 
        # TODO(lukaszkaiser): investigate the problem here and remove this hack.
        pass
      elif self.note_rnn_type == 'basic_rnn':
        var_dict[inner_name] = var
      else:
        var_dict[inner_name] = var

    return var_dict

  def build_graph(self):
    """Constructs the portion of the graph that belongs to this model."""

    tf.logging.info('Initializing melody RNN graph for scope %s', self.scope)

    with self.graph.as_default():
      with tf.device(lambda op: ''):
        with tf.variable_scope(self.scope):

          # Graph Input Placeholders
          self.Note_State_Batch = tf.placeholder(dtype=tf.float32, shape=[None, self.hparams.midi_high + 1 - self.hparams.midi_low, None, 4], name="Note_state_batch") # actually [batch_size, num_timesteps, num_notesteps, 2]

          #Generate expanded tensor from batch of note state matrices
          # Essential the CNN 'window' of this network

          self.Note_State_Expand = Input_Kernel(input_data=self.Note_State_Batch,
                                                Midi_low=self.hparams.midi_low,
                                                Midi_high=self.hparams.midi_high)

          #Dropout
          self.output_keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name= "output_keep_prob")

          # LSTM Time Wise Training Graph 
          # Generate initial state (at t=0) placeholder
          timewise_state=[]
          for i in range(len(self.hparams.rnn_layer_sizes_t)):
              timewise_c=tf.placeholder(dtype=tf.float32, shape=[None, self.hparams.rnn_layer_sizes_t[i]]) #None = batch_size * num_notes
              timewise_h=tf.placeholder(dtype=tf.float32, shape=[None, self.hparams.rnn_layer_sizes_t[i]])
              timewise_state.append(LSTMStateTuple(timewise_h, timewise_c))

          self.timewise_state=tuple(timewise_state)

          #LSTM Note Wise Graph
          # Generate initial state (at n=0) placeholder
          notewise_state=[]
          for i in range(len(self.hparams.rnn_layer_sizes_n)):
              notewise_c=tf.placeholder(dtype=tf.float32, shape=[None, self.hparams.rnn_layer_sizes_n[i]]) #None = batch_size * num_timesteps
              notewise_h=tf.placeholder(dtype=tf.float32, shape=[None, self.hparams.rnn_layer_sizes_n[i]])
              notewise_state.append(LSTMStateTuple(notewise_h, notewise_c))

          self.notewise_state=tuple(notewise_state)

          # # LSTM for events note off/ note event
          # event_c=tf.placeholder(dtype=tf.float32, shape=[None, 100]) #None = batch_size * num_timesteps
          # event_h=tf.placeholder(dtype=tf.float32, shape=[None, 100])
          # self.event_state = tuple([LSTMStateTuple(event_h, event_c)])
          # self.cell_note_off_no_event = LSTM_Cell([100], self.output_keep_prob)


          self.initial_states = [self.timewise_state, self.notewise_state] # , self.event_state

          self.cell_t = LSTM_Cell(self.hparams.rnn_layer_sizes_t, self.output_keep_prob)

          self.cell_n = LSTM_Cell(self.hparams.rnn_layer_sizes_n, self.output_keep_prob)

          self.cells = [self.cell_t, self.cell_n] # , self.cell_note_off_no_event

          def run_network_on_melody(input_data,
                                    initial_states,
                                    cells,
                                    midi_low):

            (timewise_out, timewise_state) = LSTM_Layer(
              input_data=input_data, 
              state_init=initial_states[0], 
              cell=cells[0],
              time_or_note="time")

            (notewise_out, notewise_state) = LSTM_Layer(
              timewise_out, 
              state_init=initial_states[1], 
              cell=cells[1],
              time_or_note="note")
            
            (logits_notes, _, _) = Conditional_Probability_Layer(
              notewise_out
            )

            # (event_state_out, event_state) = LSTM_Layer(
            #     input_data=input_data, 
            #     state_init=initial_states[2], 
            #     cell=cells[2],
            #     time_or_note="event")

            logits_notes = logits_notes[:,(rl_tuner_ops.MIN_NOTE-midi_low):(rl_tuner_ops.MAX_NOTE-midi_low),:,:]
            
            # if rl_tuner_ops.APPROACH == 'from_scratch':   
              # batch_size = tf.shape(event_state_out)[0]
              # num_notes = event_state_out.get_shape()[1].value
              # num_timesteps = tf.shape(event_state_out)[2]
              # linear_input = tf.transpose(event_state_out, perm=[0,2,1,3])
              # linear_input = tf.reshape(linear_input, shape=[batch_size, num_timesteps, num_notes*100])
              # logits_note_off = tf.contrib.layers.linear(inputs=linear_input, num_outputs=1)
              # logits_note_off = tf.expand_dims(logits_note_off, axis=1)
              # logits_no_event = tf.contrib.layers.linear(inputs=linear_input, num_outputs=1)
              # logits_no_event = tf.expand_dims(logits_no_event, axis=1) 
            # else:  

            input_notes = input_data[:,(rl_tuner_ops.MIN_NOTE-midi_low):(rl_tuner_ops.MAX_NOTE-midi_low),:,37:39]
            probs_notes = tf.sigmoid(logits_notes)

            probs_not_played_notes_not_played = tf.reduce_prod((1-probs_notes)*(1-input_notes), axis=1, keep_dims=True)[:,:,:,0:1]
            probs_played_notes_played = probs_notes[:,:,:,0:1] *input_notes[:,:,:,0:1] 
            probs_played_notes_not_articulated = 1 - probs_notes[:,:,:,1:2] * input_notes[:,:,:,0:1] 
            probs_played_notes_played_and_not_articulated = tf.reduce_prod(probs_played_notes_not_articulated * probs_played_notes_played, axis=1, keep_dims=True)[:,:,:,0:1]


            probs_no_event = probs_not_played_notes_not_played + probs_played_notes_played_and_not_articulated
            probs_note_off = tf.reduce_prod(1 - probs_notes[:,:,:,0:1], axis=1, keep_dims=True)

              # input_data = tf.Print(input_data,["Input data:", input_data[:,0:38,:,36:40]], summarize=200)
              # input_notes = tf.Print(input_notes,["Input notes:",input_notes[:,0:38,:,:]], summarize=200)
              # probs_notes = tf.Print(probs_notes,["Probs notes:",probs_notes[:,0:38,:,:]], summarize=76)
              # probs_not_played_notes_not_played = tf.Print(probs_not_played_notes_not_played,["Probs not played notes not played:",probs_not_played_notes_not_played[:,0:38,:,:]], summarize=76)
              # probs_played_notes_played = tf.Print(probs_played_notes_played,["Probs played notes played:",probs_played_notes_played[:,0:38,:,:]], summarize=76)
              # probs_same_note_played = tf.Print(probs_same_note_played,["Probs same note played:",probs_same_note_played[:,0:38,:,:]], summarize=76)
              # probs_notes_not_articulated = tf.Print(probs_notes_not_articulated,["Probs notes not articulated:",probs_notes_not_articulated[:,0:38,:,:]], summarize=30)
              # probs_same_note_not_articulated = tf.Print(probs_same_note_not_articulated,["Probs same note not articulated:", probs_same_note_not_articulated[:,0:38,:,:]], summarize=76)
              # probs_no_event = tf.Print(probs_no_event,["Probs no event:",probs_no_event[:,0:36,:,:]], summarize=30)
            epsilon =  1e-14
            logits_note_off = - tf.log(1. / (probs_note_off + epsilon) - 1.)
            logits_no_event = - tf.log(1. / (probs_no_event + epsilon) - 1.)

            num_notes = logits_notes.get_shape()[1].value
            logits = tf.concat([logits_note_off, logits_no_event, logits_notes[:,:,:,0:1]], axis=1)
            logits = tf.transpose(logits, perm=[0,3,2,1])
            #print(logits)

            logits_flat = tf.reshape(logits, [-1, num_notes+rl_tuner_ops.NUM_SPECIAL_EVENTS])

            return logits_flat, timewise_state, notewise_state #, event_state



          (self.logits, self.state_tensor_t, self.state_tensor_n) = run_network_on_melody( # , self.state_tensor_event
              self.Note_State_Expand, self.initial_states, self.cells, self.hparams.midi_low)

          self.softmax = tf.nn.softmax(self.logits)

          self.run_network_on_melody = run_network_on_melody

  def restore_vars_from_checkpoint(self, checkpoint_dir):
    """Loads model weights from a saved checkpoint.

    Args:
      checkpoint_dir: Directory which contains a saved checkpoint of the
        model.
    """
    tf.logging.info('Restoring variables from checkpoint')

    var_dict = self.get_variable_name_dict()
    with self.graph.as_default():
      saver = tf.train.Saver(var_list=var_dict)

    tf.logging.info('Checkpoint dir: %s', checkpoint_dir)
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    if checkpoint_file is None:
      tf.logging.warn("Can't find checkpoint file, using %s",
                      self.checkpoint_file)
      checkpoint_file = self.checkpoint_file
    tf.logging.info('Checkpoint file: %s', checkpoint_file.replace(os.getcwd(), ''))

    saver.restore(self.session, checkpoint_file)

  def load_primer(self):
    """Loads default MIDI primer file.

    Also assigns the steps per bar of this file to be the model's defaults.
    """

    if not os.path.exists(self.midi_primer):
      tf.logging.warn('ERROR! No such primer file exists! %s', self.midi_primer)
      return

    self.primer_sequence = midi_io.midi_file_to_sequence_proto(self.midi_primer)
    quantized_seq = sequences_lib.quantize_note_sequence(
        self.primer_sequence, steps_per_quarter=4)
    extracted_melodies, _ = melodies_lib.extract_melodies(quantized_seq,
                                                          min_bars=0,
                                                          min_unique_pitches=1)
    self.primer = extracted_melodies[0]
    self.steps_per_bar = self.primer.steps_per_bar

  # def prime_model(self):
  #   """Primes the model with its default midi primer."""
  #   with self.graph.as_default():
  #     tf.logging.debug('Priming the model with MIDI file %s', self.midi_primer)

  #     # Convert primer Melody to model inputs.
  #     encoder = music.OneHotEventSequenceEncoderDecoder(
  #         music.MelodyOneHotEncoding(
  #             min_note=rl_tuner_ops.MIN_NOTE,
  #             max_note=rl_tuner_ops.MAX_NOTE))

  #     seq = encoder.encode(self.primer)
  #     features = seq.feature_lists.feature_list['inputs'].feature
  #     primer_input = [list(i.float_list.value) for i in features]

  #     # Run model over primer sequence.
  #     print(primer)
  #     primer_input_batch = np.tile([primer_input], (self.batch_size, 1, 1))
  #     self.state_value, softmax = self.session.run(
  #         [self.state_tensor, self.softmax],
  #         feed_dict={self.initial_state: self.state_value,
  #                    self.melody_sequence: primer_input_batch,
  #                    self.lengths: np.full(self.batch_size,
  #                                          len(self.primer),
  #                                          dtype=int)})
  #     priming_output = softmax[-1, :]
  #     self.priming_note = self.get_note_from_softmax(priming_output)

  def get_note_from_softmax(self, softmax):
    """Extracts a one-hot encoding of the most probable note.

    Args:
      softmax: Softmax probabilities over possible next notes.
    Returns:
      One-hot encoding of most probable note.
    """

    note_idx = np.argmax(softmax)
    note_enc = rl_tuner_ops.make_onehot([note_idx], rl_tuner_ops.NUM_CLASSES)
    return np.reshape(note_enc, (rl_tuner_ops.NUM_CLASSES))

  def __call__(self):
    """Allows the network to be called, as in the following code snippet!

        q_network = MelodyRNN(...)
        q_network()

    The q_network() operation can then be placed into a larger graph as a tf op.

    Note that to get actual values from call, must do session.run and feed in
    melody_sequence, lengths, and initial_state in the feed dict.

    Returns:
      Either softmax probabilities over notes, or raw logit scores.
    """
    with self.graph.as_default():
      with tf.variable_scope(self.scope, reuse=True):
        logits, self.state_tensor_t, self.state_tensor_n = self.run_network_on_melody( # , self.state_tensor_event
            self.Note_State_Expand, [self.timewise_state, self.notewise_state], self.cells, self.hparams.midi_low) # , self.event_state
        return logits

  def run_training_batch(self):
    """Runs one batch of training data through the model.

    Uses a queue runner to pull one batch of data from the training files
    and run it through the model.

    Returns:
      A batch of softmax probabilities and model state vectors.
    """
    if self.training_file_list is None:
      tf.logging.warn('No training file path was provided, cannot run training'
                      'batch')
      return

    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=self.session, coord=coord)

    softmax, state, lengths = self.session.run([self.train_softmax,
                                                self.train_state,
                                                self.train_lengths])

    coord.request_stop()

    return softmax, state, lengths

  def get_next_note_from_note(self, note):
    """Given a note, uses the model to predict the most probable next note.

    Args:
      note: A one-hot encoding of the note.
    Returns:
      Next note in the same format.
    """
    with self.graph.as_default():
      with tf.variable_scope(self.scope, reuse=True):
        singleton_lengths = np.full(self.batch_size, 1, dtype=int)

        input_batch = np.reshape(note,
                                 (self.batch_size, 1, rl_tuner_ops.NUM_CLASSES))

        softmax, self.state_value = self.session.run(
            [self.softmax, self.state_tensor],
            {self.melody_sequence: input_batch,
             self.initial_state: self.state_value,
             self.lengths: singleton_lengths})

        return self.get_note_from_softmax(softmax)

  def variables(self):
    """Gets names of all the variables in the graph belonging to this model.

    Returns:
      List of variable names.
    """
    with self.graph.as_default():
      return [v for v in tf.global_variables() if v.name.startswith(self.scope)]
