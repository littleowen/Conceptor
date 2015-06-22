import numpy as np

def remove_silence(fs, 
                   signal,   
                   frame_duration = 0.02,
                   frame_shift = 0.01,
                   perc = 0.15):
    convert_rate = 32768.
    
    signal = (signal * convert_rate).astype(np.int64)
    siglen = len(signal)
    retsig = np.zeros(siglen, dtype = np.int64)
    frame_length = int(frame_duration * fs)
    frame_shift_length = int(frame_shift * fs)
    new_siglen = 0
    i = 0

    average_energy = np.sum(signal ** 2) / float(siglen)

    while i < siglen:
        subsig = signal[i:i + frame_length]
        ave_energy = np.sum(subsig ** 2) / float(len(subsig))
        if ave_energy < average_energy * perc:
            i += frame_length
        else:
            sigaddlen = min(frame_shift_length, len(subsig))
            retsig[new_siglen:new_siglen + sigaddlen] = subsig[:sigaddlen]
            new_siglen += sigaddlen
            i += frame_shift_length
    retsig = retsig[:new_siglen]
    return (retsig / convert_rate).astype(np.float32)

