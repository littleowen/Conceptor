import numpy as np
import sys
import os
import pickle, gzip
from scipy.special import expit

emotionLabels = pickle.load(gzip.open('LabelNumMap.pickle.gz', 'rb'), encoding='latin1')

fileidx = pickle.load(gzip.open('testIdx.pickle.gz', 'rb'), encoding='latin1')

feature_mat = pickle.load(gzip.open('testfeature.pickle.gz', 'rb'), encoding='latin1')

_, indices = pickle.load(gzip.open('test.pickle.gz', 'rb'), encoding='latin1')

W, U = pickle.load(gzip.open('ELMWeights.pickle.gz', 'rb'), encoding='latin1')

segnum = indices.max()

utterFeatList = []

for i in range(segnum + 1):
    frames = feature_mat[indices == i, :]
    # change the way of computing energy using the new energy file
    if frames.size != 0:
        feat1 = np.amax(frames, axis=0)
        feat2 = np.amin(frames, axis=0)
        feat3 = np.mean(frames, axis=0)
        feat4 = np.mean(frames > 0.2, axis=0)
        utter_feat = np.hstack([feat1, feat2, feat3, feat4])
        utterFeatList.append(utter_feat)
    utterFeats = np.vstack(utterFeatList)
    
H = expit(W.T.dot(utterFeats.T))
Predicts = H.T.dot(U)

results = np.argmax(Predicts, axis= 1)
inv_map = {v: k for k, v in emotionLabels.items()}

text_file = open("Results.txt", "w")
for i in range(segnum + 1):
    text_file.write("%s  %s\n" % (fileidx[i], inv_map[results[i]]))
text_file.close()