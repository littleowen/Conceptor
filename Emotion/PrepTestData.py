import numpy as np
from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav
from energy import top_energy
import os
import pickle, gzip
from PrepTrainData import frame2segment, extract_feats
import sys


if __name__ == '__main__':
    
    emotionLabels = pickle.load(gzip.open('LabelNumMap.pickle.gz', 'rb'), encoding='latin1')

    # Testing data loading
    featureList = []
    idxList = []
    idx = 0
    
    fileidx = {}

    Dir_test = sys.argv[-1]

    for fn in os.listdir(Dir_test):
        if fn.endswith(".wav"):
            audio_path = Dir_test + '/' + fn
            (sr,signal) = wav.read(audio_path)
            # find the index of segments with top 10% energy
            energyVec = top_energy(sr, signal, frame_duration = 0.265, perc = 0.1)            

            # extract audio features
            feats = extract_feats(signal, sr)

            # convert frames into segments
            segments = frame2segment(feats, 25, energyVec)
            segnum = segments.shape[0]

            # testing data
            featureList.append(segments)
            indices = np.zeros(segnum) + idx
            idxList.append(indices)
            fileidx[idx] = fn
            idx += 1
            
    features_test = np.vstack(featureList).astype('float32')
    indices_test = np.hstack(idxList).astype('int')


    with gzip.open('test.pickle.gz', 'wb') as f:
        pickle.dump((features_test, indices_test), f, protocol = 2)
        
    with gzip.open('testIdx.pickle.gz', 'wb') as f:
        pickle.dump(fileidx, f, protocol = 2)

