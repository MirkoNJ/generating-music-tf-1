import random
import numpy as np
import tensorflow as tf

division_len = 48 # interval between possible start locations


def getPieceSegment(pieces, num_time_steps):
    piece_output = random.choice(list(pieces.values()))
    start = random.randrange(0, (len(piece_output) - num_time_steps), division_len)

    seg_out = piece_output[start : (start + num_time_steps)]

    return seg_out

def getPieceBatch(pieces, batch_size, num_time_steps):
    num_time_steps += 1
    out = [getPieceSegment(pieces, num_time_steps) for _ in range(batch_size)]
    out = np.array(out)
    out = np.swapaxes(out, axis1=1, axis2=2)
    return out

def getPieceSegment2(piece, num_time_steps, start):
    if len(piece) < (start + num_time_steps):
        start = len(piece) - num_time_steps
    # print("Start: {}".format(start))
    # print("End: {}".format(start + num_time_steps))
    return piece[start : (start + num_time_steps)]
    
def getPieceBatch2(piece, batch_size, num_time_steps, start_old):
    num_time_steps += 1
    starts = [int(start_old + x * num_time_steps * 0.5) for x in range(batch_size)]
    out = [getPieceSegment2(piece, num_time_steps, start) for start in starts]
    out = np.array(out)
    out = np.swapaxes(out, axis1=1, axis2=2)
    start_out = int(max(starts)+num_time_steps*0.5)
    return out, start_out