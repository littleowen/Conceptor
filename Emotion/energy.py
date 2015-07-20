import sys
import scipy.io.wavfile as wavfile
import numpy as np

def top_energy(fs, signal,
        frame_duration = 0.265,
        frame_shift = 0.01,
        perc = 0.1):
    orig_dtype = type(signal[0])
    typeinfo = np.iinfo(orig_dtype)
    is_unsigned = typeinfo.min >= 0
    signal = signal.astype(np.int64)
    if is_unsigned:
        signal = signal - (typeinfo.max + 1) / 2

    siglen = len(signal)
    frame_length = int(frame_duration * fs)
    frame_shift_length = int(frame_shift * fs)
    i = 0
    
    energyList = []
    while (i + frame_length) < siglen:
        subsig = signal[i: i+frame_length]
        ave_energy = np.sum(subsig ** 2) / float(len(subsig))
        energyList.append(ave_energy)
        i += frame_shift_length
    energyVec = np.hstack(energyList)
    frameNum = len(energyList)
    topNum = int(frameNum * perc) 
    indices = energyVec.argsort()[-topNum:][::-1]
        
    return indices
