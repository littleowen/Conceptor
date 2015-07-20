import numpy as np
from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav
from energy import top_energy
import os
import pickle, numpy, gzip
import random
import sys

def frame2segment(featMat, segSize, indices):
    resultMat = featMat
    for i in range(segSize - 1):
        resultMat = np.hstack([resultMat, np.vstack([featMat[(i + 1):, :], featMat[0:(i + 1), :]])])
    return resultMat[indices, :]

# extract audio features
def extract_feats(signal, sr):
    feats = mfcc(signal, sr)
    #fbank_feat = logfbank(signal, sr, nfilt = 17)
    #feats = np.hstack((mfcc_feat, fbank_feat))
    mu = np.mean(feats, axis = 0)
    sigma = np.std(feats, axis = 0)
    feature = (feats - mu) / sigma
    return feature

if __name__ == '__main__':

    emotionLabels = {}
    labelnum = 0

    # Training data loading
    ELMfeatureList = []
    ELMlabelList = []
    ELMidxList = []
    ELMidx = 0

    featureList = []
    labelList = []

    Dir_train = sys.argv[-2]
    subdirs_train = [x[0] for x in os.walk(Dir_train)]
    for subdir_train in subdirs_train:
        if subdir_train != Dir_train:
            label = subdir_train.split('/')[1]
            emotionLabels[label] = labelnum
            labelnum += 1
            for fn in os.listdir(subdir_train):
                if fn.endswith(".wav"):
                    audio_path = subdir_train + '/' + fn
                    (sr,signal) = wav.read(audio_path)

                    # find the index of segments with top 10% energy
                    energyVec = top_energy(sr, signal, frame_duration = 0.265, perc = 0.1)            

                    # extract audio features
                    feats = extract_feats(signal, sr)

                    # convert frames into segments
                    segments = frame2segment(feats, 25, energyVec)        
                    segnum = segments.shape[0]

                    # training data for DNN
                    labels = np.zeros(segnum) + emotionLabels[label]
                    featureList.append(segments)
                    labelList.append(labels)  

                    # training data for ELM
                    ELMfeatureList.append(segments)
                    idices = np.zeros(segnum) + ELMidx
                    ELMidxList.append(idices)
                    ELMidx += 1
                    ELMlabelList.append(emotionLabels[label])
    features_train = np.vstack(featureList).astype('float32')
    labels_train = np.hstack(labelList).astype('int')
    idxarray = np.array(range(features_train.shape[0]))
    random.shuffle(idxarray)
    features_train = features_train[idxarray,:]
    labels_train = labels_train[idxarray]

    with gzip.open('train.pickle.gz', 'wb') as f:
         pickle.dump((features_train, labels_train), f, protocol = 2)

    # Validation data loading
    featureList = []
    labelList = []

    Dir_valid = sys.argv[-1]
    subdirs_valid = [x[0] for x in os.walk(Dir_valid)]
    for subdir_valid in subdirs_valid:
        if subdir_valid != Dir_valid:
            label = subdir_valid.split('/')[1]
            for fn in os.listdir(subdir_valid):
                if fn.endswith(".wav"):
                    audio_path = subdir_valid + '/' + fn
                    (sr,signal) = wav.read(audio_path)

                    # find the index of segments with top 10% energy
                    energyVec = top_energy(sr, signal, frame_duration = 0.265, perc = 0.1)            

                    # extract audio features
                    feats = extract_feats(signal, sr)

                    # convert frames into segments
                    segments = frame2segment(feats, 25, energyVec)
                    segnum = segments.shape[0]

                    # training data for DNN
                    labels = np.zeros(segnum) + emotionLabels[label]
                    featureList.append(segments)
                    labelList.append(labels)  

                    # training data for ELM
                    ELMfeatureList.append(segments)
                    idices = np.zeros(segnum) + ELMidx
                    ELMidxList.append(idices)
                    ELMidx += 1
                    ELMlabelList.append(emotionLabels[label])


    features_valid = np.vstack(featureList).astype('float32')
    labels_valid = np.hstack(labelList).astype('int')

    with gzip.open('valid.pickle.gz', 'wb') as f:
        pickle.dump((features_valid, labels_valid), f, protocol = 2)

    features_ELM = np.vstack(ELMfeatureList).astype('float32')
    indices_ELM = np.hstack(ELMidxList).astype('int')
    labels_ELM = np.hstack(ELMlabelList).astype('int')

    with gzip.open('ELMtrain.pickle.gz', 'wb') as f:
        pickle.dump((features_ELM, indices_ELM), f, protocol = 2)

    with gzip.open('ELMlabels.pickle.gz', 'wb') as f:
        pickle.dump(labels_ELM, f, protocol = 2)

    with gzip.open('LabelNumMap.pickle.gz', 'wb') as f:
        pickle.dump(emotionLabels, f, protocol = 2)
