## source of original loadPieces:               https://github.com/danieldjohnson/biaxial-rnn-music-composition/blob/master/multi_training.py
## source of original midiToNoteStateMatrix:    https://github.com/danieldjohnson/biaxial-rnn-music-composition/blob/master/midi_to_statematrix.py
## source of original noteStateMatrixToMidi:    https://github.com/danieldjohnson/biaxial-rnn-music-composition/blob/master/midi_to_statematrix.py
## source of midi package                       https://github.com/louisabraham/python3-midi

import midi
import os
import numpy as np

# full range of a piano keyboard sub-contra A to five-lined C
lowerBound = 21 # inclusiv, sub-contra A is  21
upperBound = 109 # exlusiv, five-lined C is 108, the highest note to be included


def midiToNoteStateMatrix(midifile, lowerBound = 21, upperBound = 109,  verbose=False, verbose_ts=False):

    span = upperBound - lowerBound
    pattern = midi.read_midifile(midifile)
    timeleft = [track[0].tick for track in pattern]
    posns = [0 for track in pattern]
    state = [[0, 0, 0, 1] for x in range(span)]
    statematrix = []
    statematrix.append(state)
    condition = True
    time = 0
    beat = 1
    numerator = None

    if verbose_ts:
        print("Resolution", pattern.resolution)
    
    while condition:
        if verbose:
            print("time:", time)
        if (time % (pattern.resolution / 4)) == 0 and time != 0:
            beat = int((time / (pattern.resolution / 4)) % (16 * (numerator / denominator))) + 1
        if (time % (pattern.resolution / 12)) == 0 and time != 0: 
            # == (pattern.resolution / 8):#, if I am not mistaken this introduces a 32th note shift, don't know what this was intended for?
            # (time % (pattern.resolution / 12)) == 0 -> a state has the length of a 48th note
            # Crossed a note boundary. Create a new state, defaulting to holding notes
            oldstate = state
            state = [[oldstate[x][0],0, oldstate[x][2], beat] for x in range(span)]
            statematrix.append(state)

            if verbose:
                print("crossed note boundary, current state:", state)
        for i in range(len(timeleft)): #For each track
            if not condition:
                break
            while timeleft[i] == 0:
                track = pattern[i]
                pos = posns[i]

                evt = track[pos]
                if verbose_ts:
                    print(evt)
                if isinstance(evt, midi.NoteEvent):
                    if (evt.pitch < lowerBound) or (evt.pitch >= upperBound):
                        if verbose_ts:
                            print("Note {} at time {} out of bounds (ignoring)".format(evt.pitch, time))
                        pass
                    else:
                        if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                            state[evt.pitch-lowerBound] = [0, 0, 0, beat]
                            if verbose:                                
                                print("velocity:", evt.velocity)
                                print("current_state:", state)
                        else:
                            state[evt.pitch-lowerBound] = [1, 1, evt.velocity, beat]
                            if verbose:
                                print("velocity:", evt.velocity)
                                print("pitch:", evt.pitch)
                                print("current_state:", state)
                elif isinstance(evt, midi.TimeSignatureEvent):
                    if evt.numerator in (5,7,11,13):
                        if verbose_ts:
                            print("Bailing - strange time signature: " +  str(evt.numerator) + " at time:" + str(time))
                        out =  statematrix
                        condition = False
                        break
                    else:
                        # print("Name : {}".format(midifile))
                        numerator = evt.numerator
                        denominator = evt.denominator
                try:
                    if verbose:
                        print("timeleft_" + str(i) + ": " + str(timeleft[i]))
                    timeleft[i] = track[pos + 1].tick
                    posns[i] += 1
                except IndexError:
                    if verbose:
                        print("timeleft_" + str(i) + ": None")
                    timeleft[i] = None

            if timeleft[i] is not None:
                if verbose:
                    print("timeleft_" + str(i) +"_new:" + str(timeleft[i]))
                timeleft[i] -= 1

            if numerator is None:
                if verbose:
                    print("Bailing - No time signature: ")
                statematrix =  None
                condition = False
                break

        if all(t is None for t in timeleft):
            break

        time += 1

    return statematrix

def removeAdditionalNotesInEnding(statematrix):
    sixteenth_index = [x[0][3] for x in statematrix]
    latest_index = [i for i, e in enumerate(sixteenth_index) if e == max(sixteenth_index)][-1]
    return statematrix[0:(latest_index+1)]

def loadPieces(dirpath, min_time_steps, lowerBound = 21, upperBound = 109, verbose=False, verbose_name=True):
    pieces = {}

    arr = sorted(os.scandir(dirpath), key=lambda x: x.name)

    for f in arr:

        if f.name[-4:] not in ('.mid','.MID'):
            continue

        name = f.name[:-4]

        try:
            outMatrix = midiToNoteStateMatrix(os.path.join(dirpath, f.name), lowerBound , upperBound, False, verbose)
        except:
            if verbose:
                print('Skip bad file = ', name)
            continue

        if outMatrix is None:
            if verbose:
                print('No time signature = ', name)
            continue
            
        if len(outMatrix) < min_time_steps:
            if verbose:
                print('Skip too short, but valid file = ', name)
            continue

        outMatrix = removeAdditionalNotesInEnding(outMatrix)
        pieces[name] = outMatrix
        if verbose_name:
            print("Loaded {}".format(name))
    return pieces

def noteStateMatrixToMidi(statematrix, lowerBound = 21, upperBound = 109, name="example", tickscale=48):

    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)
    
    span = upperBound - lowerBound
    lastcmdtime = 0
    prevstate = [[0,0,0] for x in range(span)]
    for time, state in enumerate(statematrix[:,:,0:3] + [prevstate[:]]):  
        offNotes = []
        onNotes = []
        velocity = []
        for i in range(span):
            n = state[i]
            p = prevstate[i]
            if p[0] == 1:
                if n[0] == 0:
                    offNotes.append(i)
                elif n[1] == 1:
                    offNotes.append(i)
                    onNotes.append(i)
                    velocity.append(n[2])
            elif n[0] == 1:
                onNotes.append(i)
                velocity.append(n[2])
        for note in offNotes:
            track.append(midi.NoteOffEvent(tick=(time - lastcmdtime) * tickscale, pitch=note + lowerBound))
            lastcmdtime = time
        for note in enumerate(onNotes):
            track.append(midi.NoteOnEvent(tick=(time - lastcmdtime) * tickscale, velocity=velocity[note[0]], pitch=note[1] + lowerBound))
            lastcmdtime = time
            
        prevstate = state
    
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    midi.write_midifile("{}.mid".format(name), pattern)

def generate_audio(batch_predictions, save_path_dir, midi_name, tickscale=48, verbose=False):

    try:
        os.mkdir(save_path_dir)    
        if verbose:
            print('creating new destination folder')
    except:
        if verbose:
            print('destination folder exists')
    save_path = save_path_dir + midi_name
    
    midi_matrix = np.transpose(batch_predictions, axes=[0,2,1,3])[0,:,:,:]
    midi_matrix = midi_matrix.astype(np.int32)
    noteStateMatrixToMidi(midi_matrix, name=save_path, tickscale=tickscale)
    if verbose:
        print('Saved midi to', save_path)

EXACT_FILES = [ # Downloaded from http://www.piano-midi.de/chopin.htm
    'chop0701',
    'chop0702',
    'chop1001',
    'chop1005',
    'chop1012',
    'chop1800', # Waltz 
    'chop2300',
    'chop2501',
    'chop2502', # Etude 4/4 time signature
    'chop2503', # Etude 3/4 time signature
    'chop2504', # Etude 4/4 time signature
    'chop2511',
    'chop2512',
    'chop2701',
    'chop2702',
    'chop2801',
    'chop2802',
    'chop2803',
    'chop2804',
    'chop2805',
    'chop2806',
    'chop2807',
    'chop2808',
    'chop2809',
    'chop2810',
    'chop2811',
    'chop2812',
    'chop2813',
    'chop2814',
    'chop2815',
    'chop2816',
    'chop2817',
    'chop2818',
    'chop2819',
    'chop2820',
    'chop2821',
    'chop2822',
    'chop2823',
    'chop2824',
    'chop3100', # Scherzo No 2. 3/4
    'chop3302',
    'chop3304',
    'chop3501',
    'chop3502',
    'chop3503',
    'chop3504',
    'chop5300', # Polonaise, 3/4
    'chop6600',

    'chop0601', # Downloaded from http://www.oocities.org/vienna/4279/MIDI.html
    'chop0602',
    'chop0603',
    'chop0901',
    'chop0902', # Nocturne, 12/8 
    'chop0903',
    'chop1002',
    'chop1003',
    'chop1004',
    # 'chop1006', # No time signature
    'chop1007',
    # 'chop1008', # No time signature
    # 'chop1009', # No time signature
    'chop1501', # Nocturne, 3/4
    'chop1502', # Nocturne, 2/4
    'chop1503', # Nocturne, 3/4
    'chop2506',
    # 'chop2507', # Wrongly encoded as 2/4 instead of 3/4
    'chop2509',
    'chop3002',
    'chop3202',
    # 'chop3402', # Wrongly encoded as 2/4 instead of 3/4
    'chop3701', # Nocturne, 4/4
    'chop3702', # Nocturne, 6/8
    'chop3800', # Ballade, 6/8
    'chop4001', # Polonaise, 3/4
    'chop4002',
    'chop4200', # Waltz
    'chop4700', # Ballade, 6/8
    'chop4801', # Nocturne, 4/4
    'chop4802', # Nocturne, 4/4
    'chop5200', # Ballade, 6/8
    'chop5501', # Nocturne, 4/4
    'chop5502', # Nocturne, 12/8
    # 'chop5801', # Really weirdly encoded 4/4
    # 'chop5802', # Really weirdly encoded 4/4
    # 'chop5803', # Really weirdly encoded 4/4
    # 'chop5804', # Really weirdly encoded 4/4
    # 'chop5902', # Also wrongly encoded Mazurka 4/4 instead of 3/4
    # 'chop6201', # Really weirdly encoded 4/4
    'chop6202', # Nocturne 4/4
    # 'chop6303', # Also wrongly encoded Mazurka 4/4 instead of 3/4
    'chop6401', # Waltz
    # 'chop6402', # Wrongly encoded Waltz 4/4 instead of 3/4
    'chop6701', # Mazurka 3/4
    'chop6702', # Mazurka 3/4
    'chop6704', # Mazurka 3/4
    'chop6802', # Mazurka 3/4
    'chop6901', # Waltz
    'chop6902', # Waltz
    'chop7002', # Waltz
    'chop7201'  # Nocturne
    ]
