import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import DropoutWrapper
from modules.nikhil.batch import getPieceBatch2

import pdb



num_notes_octave = 12
Midi_low = 21
Midi_high = 108

def Input_Kernel(input_data, Midi_low=Midi_low, Midi_high=Midi_high):
    """
    ### Arguments:
        input_data: size = [batch_size x num_notes x num_timesteps x 2] 
            (the input data represents that at the previous timestep of what we are trying to predict)
        Midi_low: integer
        Midi_high: integer
    ### Returns:
        Note_State_Expand: size = [batch_size x num_notes x num_timesteps x 80]
    """ 

    # capture input_data dimensions (batch_size and num_timesteps are variable length)
    batch_size = tf.shape(input_data)[0]           #  16
    num_notes = input_data.get_shape()[1].value    #  88
    num_timesteps = tf.shape(input_data)[2]        # 128


    # beat 
    x_Time = input_data[:,:,:,3] - 1
    x_beat = tf.stack([x_Time%2,  x_Time//2%2, x_Time//4%2, x_Time//8%2, x_Time//16%2],axis=-1)

    # time signature
    time_signature = tf.reduce_max(input_data[:,:,:,3:4],  axis=2, keep_dims=True) # 12 * tf.ones((batch_size,num_notes,1,1))
    x_time_signature = tf.tile(time_signature, [1,1,num_timesteps,1])
    x_time_signature = (x_time_signature - 4) / (24 - 4) # min-max-scale to [0,1]

    input_data = input_data[:,:,:,0:3]

    # MIDI note number (only a function of the note index)
    Midi_indices = tf.squeeze(tf.range(start=Midi_low, limit = Midi_high+1, delta=1))
    Midi_indices = (Midi_indices - Midi_low) / (Midi_high+1- Midi_low) # min-max-scale to [0,1]
    x_Midi = tf.ones((batch_size, num_timesteps, 1, num_notes))*tf.cast(Midi_indices, dtype=tf.float32)
    x_Midi = tf.transpose(x_Midi, perm=[0,3,1,2]) # shape (16, 88, 128, 1) -> [batch_size, num_notes, num_timesteps, 1]

    # part_pitchclass (only a function of the note index)
    Midi_pitchclasses = tf.squeeze(tf.cast(x_Midi * (Midi_high+1- Midi_low) + Midi_low, dtype=tf.int32) % num_notes_octave, axis=3)
    x_pitch_class = tf.one_hot(tf.cast(Midi_pitchclasses, dtype=tf.uint8), depth=num_notes_octave) # shape (16, 88, 128, 12)

    # part_prev_vicinity
    # pdb.set_trace()
    input_flatten = tf.cast(tf.transpose(input_data, perm=[0,2,1,3]), dtype=tf.float32) # remove velocity if defined
    input_flatten = tf.reshape(input_flatten, [batch_size * num_timesteps, num_notes, 3]) # shape (128*16=2048, 88, 3)
    input_flatten_p = tf.slice(input_flatten, [0,0,0],size=[-1, -1, 1])                 # shape (2048, 88, 1) -> [batch size, width, in channels]
    input_flatten_a = tf.slice(input_flatten, [0,0,1],size=[-1, -1, 1])                 # shape (2048, 88, 1) -> [batch size, width, in channels]
    input_flatten_vel = tf.slice(input_flatten, [0,0,2],size=[-1, -1, 1])                 # shape (2048, 88, 1) -> [batch size, width, in channels]


    # reverse identity kernel
    filt_vicinity = tf.cast(tf.expand_dims(tf.eye(num_notes_octave * 2 + 1), axis=1), dtype=tf.float32) 
    # shape (25,1,25) = [kernel_1d_size, in_channels, "number_of_kernels_1d" = out_channels]
    # the relative values one octave up and one octave down

    # 1D convolutional filter for each play and articulate arrays 
    vicinity_p = tf.nn.conv1d(input_flatten_p, filt_vicinity, stride=1, padding='SAME') # shape (2048, 88, 25)
    vicinity_a = tf.nn.conv1d(input_flatten_a, filt_vicinity, stride=1, padding='SAME') # shape (2048, 88, 25)
    vicinity_vel = tf.nn.conv1d(input_flatten_vel, filt_vicinity, stride=1, padding='SAME') # shape (2048, 88, 25)
    vicinity_vel = (vicinity_vel - 0) / (127- 0) # min-max-scale to [0,1]
    
    # concatenate back together and restack such that play-articulate-velocity numbers alternate
    vicinity = tf.stack([vicinity_p, vicinity_a, vicinity_vel], axis=3) # 1 array shape (2048, 88, 25, 3)
    vicinity = tf.unstack(vicinity, axis=2)               # 25 arrays of shape (2048, 88, 3)
    vicinity = tf.concat(vicinity, axis=2)                # 1 array shape (2048, 88, 75) 

    # reshape by major dimensions, THEN swap axes
    x_vicinity = tf.reshape(vicinity, shape = [batch_size, num_timesteps, num_notes, (num_notes_octave * 2 + 1) * 3]) # shape (16, 128, 88, 75)
    x_vicinity = tf.transpose(x_vicinity, perm=[0,2,1,3]) # shape (16, 88, 128, 75)

    # kernel
    filt_context = tf.expand_dims(tf.tile(tf.eye(num_notes_octave), multiples=[(num_notes // num_notes_octave) * 2, 1]), axis=1) 
    # shape (168, 1, 12) = [kernel_1d_size, in_channels, "number_of_kernels_1d" = out_channels]
    # n = num_notes // num_notes_octave * 2 stacked identy matrices where n is the rounded number of octaves, times two, 
    # for both directions (eg. n = 14 -> consider the seven higher and the seven lower octaves)

    # part_prev_context
    context = tf.nn.conv1d(input_flatten_p, filt_context, stride=1, padding='SAME')
    x_context = tf.reshape(context, shape=[batch_size, num_timesteps, num_notes, num_notes_octave])
    x_context = tf.transpose(x_context, perm=[0,2,1,3])

    # add the mean velocity of one octave up and one octave down
    velocity_count = tf.cast(tf.count_nonzero(tf.round(vicinity_p), axis=2, keep_dims=True), dtype=tf.float32)
    velocity_sum = tf.reduce_sum(vicinity_vel, axis=2, keep_dims=True)
    x_velocity = velocity_sum / tf.maximum(velocity_count,1) # shape (2048, 88, 1), avoid division by zero
    x_velocity = tf.reshape(x_velocity, shape = [batch_size, num_timesteps, num_notes, 1]) 
    x_velocity = tf.transpose(x_velocity, perm=[0,2,1,3]) # shape (16, 88, 128, 1)

    # zero
    x_zero = tf.zeros([batch_size, num_notes, num_timesteps,1])

    # final array (input vectors per batch, note and timestep)
    Note_State_Expand = tf.concat([x_Midi, x_pitch_class, x_vicinity, x_context, x_beat, x_time_signature, x_velocity, x_zero], axis=-1) 
    
    return Note_State_Expand

def LSTM_Cell(rnn_layer_sizes, output_keep_prob=1.0):
    
    # generate cell list of length specified by initial state
    cell_list=[]
    for h in range(len(rnn_layer_sizes)):
        lstm_cell = BasicLSTMCell(
            num_units = rnn_layer_sizes[h], 
            forget_bias = 1.0,
            state_is_tuple = True,
            activation = math_ops.tanh, 
            reuse = None
            )
        lstm_cell = DropoutWrapper(lstm_cell, output_keep_prob=output_keep_prob)
        cell_list.append(lstm_cell)
    
    #Instantiate multi layer Time-Wise Cell
    multi_lstm_cell = tf.contrib.rnn.MultiRNNCell(cell_list, state_is_tuple=True)

    return multi_lstm_cell


def LSTM_Layer(input_data, state_init, cell, time_or_note="time"):
    """
    Arguments:
        input_data: Tensor with size = [batch_size x num_notes x num_timesteps x input_size]
        state_init: List of LSTMTuples([batch_size*num_notes x num_units[layer]], [batch_size*num_notes x num_units[layer]])
        
    Returns:
        output: tensor with size = [batch_size*num_notes x num_timesteps x num_units_final
        state: List of LSTMTuples([batch_size*num_notes x num_units[layer]], [batch_size*num_notes x num_units[layer]])
        
    # LSTM time-wise 
    # This section is the 'Model LSTM-TimeAxis' block and will run a number of LSTM cells over the time axis.
    # Every note and sample in the batch will be run in parallel with the same LSTM weights


    
    # Reshape the input
    # batch_size and num_notes dimensions of input are flattened to treat as single 'batch' dimension for LSTM cell
    # will be reshaped at the end of this block for the next stage
    # state_init is already flat for convenience
  """  
    
    # batch_size and num_timesteps are variable length
    batch_size = tf.shape(input_data)[0]
    num_notes = input_data.get_shape()[1].value
    num_timesteps = tf.shape(input_data)[2]
    input_size = input_data.get_shape()[3].value


    # Flatten input
    if time_or_note == "time" or time_or_note == "event":
        input_flatten = tf.reshape(input_data, shape=[batch_size*num_notes, num_timesteps, input_size])
    elif time_or_note == "note":
        input_flatten = tf.reshape(input_data, shape=[batch_size*num_timesteps, num_notes, input_size])
    else:
        print("Error")
        pass

    #Run through LSTM time steps and generate time-wise sequence of outputs
    # input_flatten = tf.Print(input_flatten, ["Input flatten shapes:", tf.shape(input_data), tf.shape(input_flatten), tf.shape(state_init[0]), cell.output_size], summarize = 100)

    output_flat, state_out = tf.nn.dynamic_rnn(cell=cell, inputs=input_flatten, initial_state=state_init, dtype=tf.float32, scope="rnn_" + time_or_note)

    output = tf.reshape(output_flat, shape=[batch_size, num_notes, num_timesteps, cell.output_size])
    
    return output, state_out


def Conditional_Probability_Layer(input_data, dense_units=3):
    """
    Arguments:
        input_data: size = [batch_size x num_notes x num_timesteps x size_input]
        state_init: List of LSTMTuples, each like ([batch_size*num_time_steps x num_units[layer]], [batch_size*num_timesteps x num_units[layer]]) (state_init will likely always be zero for the notewise layer)
        output_keep_prob: float between 0 and 1 specifying dropout layer retention
          
    # LSTM note-wise
    # This section is the 'Model LSTM-Note Axis' block and runs a number of LSTM cells from low note to high note
    # A batches and time steps are run in parallel in
    # The input sequence to the LSTM cell is the hidden state output from the previous block for each note
    #  concatenated with a sampled output from the previous note step
    # The input data is 'Hid_State_Final' with dimensions batch_size x num_notes x num_timesteps x num_units
    # The output will be:
    #    - y = logits(Probability=1) with dimensions batch_size x num_notes x num_timesteps x 2
    #    - note_gen with dimensions batch_size x num_notes x num_timesteps x 2

 
    """
    
    # batch_size and num_timesteps are variable length
    batch_size = tf.shape(input_data)[0]
    num_notes = input_data.get_shape()[1].value
    num_timesteps = tf.shape(input_data)[2]
    cell_output_size = input_data.get_shape()[3].value
    
    input_data = tf.transpose(input_data, perm=[0,2,1,3])
    input_data = tf.reshape(input_data, shape=[batch_size * num_timesteps, num_notes , cell_output_size])

    x = [input_data[:,n,:] for n in range(num_notes)]

    # x = tf.Print(x, ["Shape Cond Prob x:",tf.shape(x)], summarize = 100)
    x_tmp = tf.layers.dense(inputs=x[0], units=dense_units, activation=None)
    note = tf.distributions.Bernoulli(logits=x_tmp[:,0:2], dtype=x_tmp.dtype).sample()
    p = tf.slice(note, [0,0], [-1,1])
    a = tf.slice(note, [0,1], [-1,1])          
    a = p*a
    v = x_tmp[:,2:3]
    note = tf.concat([p,a,v], axis=-1)

    x_list = [x_tmp]
    note_list = [note[:,0:2]]

    #Run through notes for note-wise LSTM to obtain P(va(n) | va(<n))
    for n in range(1, num_notes):    
        #concatenate previously sampled note play-articulate-combo with timewise output
        # feed back both 'play' and 'articulate' components (articulate component is the masked version)
        x_tmp = tf.concat([x[n], note], axis=-1)
        x_tmp = tf.layers.dense(inputs=x_tmp, units=dense_units, activation=None)
        note = tf.distributions.Bernoulli(logits=x_tmp[:,0:2], dtype=x_tmp.dtype).sample()
        p = tf.slice(note, [0,0], [-1,1])
        a = tf.slice(note, [0,1], [-1,1])          
        a = p*a
        v = x_tmp[:,2:3]
        note = tf.concat([p,a,v], axis=-1)

        x_list.append(x_tmp)
        note_list.append(note[:,0:2])
    
    # Convert output list to a Tensor
    y_out = tf.stack(x_list, axis=1)
    note_gen_out = tf.stack(note_list, axis=1)
    # y_out = tf.Print(y_out, ["Output Shape:", tf.shape(y_out), tf.shape(note_gen_out)], summarize = 100)

    y_out = tf.reshape(y_out, shape=[batch_size, num_timesteps, num_notes , dense_units]) # make sure that this works correctly
    y_out = tf.transpose(y_out, perm=[0,2,1,3])    

    note_gen_out = tf.reshape(note_gen_out, shape=[batch_size, num_timesteps, num_notes , 2]) # make sure that this works correctly
    note_gen_out = tf.transpose(note_gen_out, perm=[0,2,1,3]) 

    velocity_out = y_out[:,:,:,2:3]   
    velocity_out = tf.sigmoid(velocity_out)
    velocity_out = tf.cast(velocity_out * 127, dtype=tf.float32)

    return y_out[:,:,:,0:2], velocity_out, note_gen_out #, h_state, tf.reshape(h_final_out, shape=[batch_size, 1, num_timesteps, int(h_final_out.shape[1])])

def alignXy(X, y):
    y = y[:,:,1:,:]
    X = X[:,:,:-1,:]
    return X, y

def Loss_Function_1(y_true, y_pred):
    """
    Arguments:
        Note State Batch: shape = [batch_size x num_notes x num_timesteps x 2]
        batch of logit(prob=1): shape = [batch_size x num_notes x num_timesteps x 2]
        
    ### This section is the Loss Function Block
    Note_State_Batch contains the actual binary values played and articulated for each note, at every time step, for every batch
    Entries in y_out at time step $t$ were generated by entries in Note_State_Batch at time step $t$.  The objective of the model is for 
    entries in y_out at time step $t$ to predict Note_State_Batch at time step $t+1$.  In order to properly align the tensors for the  
    loss function calculation, the last time slice of y_out is removed, and the first time slice of Note_State_Batch is removed.
    """   

    y_pred = tf.convert_to_tensor(y_pred)[:,:,:,0:2]
    y_true = tf.convert_to_tensor(y_true, dtype=y_pred.dtype)[:,:,:,0:2]
    num_notes = y_true.get_shape()[1].value

    
    # calculate log likelihoods
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true)
    
    # if note is not played, mask out loss term for articulation    
    cross_entropy_p = cross_entropy[:,:,:,0]
    cross_entropy_a = cross_entropy[:,:,:,1] * y_true[:,:,:,0] 
    cross_entropy = tf.stack([cross_entropy_p, cross_entropy_a], axis=-1)
   
    # calculate the loss function as defined in the paper
    Loss = tf.reduce_mean(cross_entropy) * 2 # negative log-likelihood of batch (factor of 2 for both play and articulate)
    
    # calculate the log-likelihood of notes at a single time step
    Log_likelihood = - Loss * num_notes
    
    return Loss, Log_likelihood

def Loss_Function_2(y_true, y_pred, epsilon = 1e-08):

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.convert_to_tensor(y_true, dtype=y_pred.dtype)

    # mask error in prediction if note is not played
    y_pred = y_pred[:,:,:,0:1] * y_true[:,:,:,0:1]
    count_non_zero = tf.maximum(tf.count_nonzero(y_pred, dtype=tf.float32), 1) # avoid division by zero

    Loss = tf.reduce_sum(tf.squared_difference(y_pred, y_true[:,:,:,2:3])) / count_non_zero + epsilon
    # Loss = tf.Print(Loss, ["Loss: ", y_pred[0,0:88,10,:], y_true[0,0:88,10,2:3], Loss, count_non_zero], summarize = 200, first_n= 10)
    return Loss

def getNumberOfBatches(pieces, batch_size, num_time_steps):
    n = 0
    for k in pieces.keys():
        start_old = 0
        piece = pieces[str(k)]
        while start_old < (len(piece)- num_time_steps):
            _, start_old = getPieceBatch2(piece, num_time_steps = num_time_steps, batch_size=batch_size, start_old=start_old)
            n += 1
    return n 


