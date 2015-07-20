import numpy as np
import sys
import os
import pickle, gzip
from scipy.special import expit

def ind2vec(ind, N=None):
    ind = np.asarray(ind)
    if N is None: 
        N = ind.max() + 1
    return (np.arange(N) == ind[:,None]).astype(int)

if __name__ == '__main__':


    feature_mat = pickle.load(gzip.open('ELMfeature.pickle.gz', 'rb'), encoding='latin1')

    labels = pickle.load(gzip.open('ELMlabels.pickle.gz', 'rb'), encoding='latin1')

    _, indices = pickle.load(gzip.open('ELMtrain.pickle.gz', 'rb'), encoding='latin1')

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

    L = 120
    D = utterFeats.shape[1]
    W = np.random.uniform(-1, 1, (D, L))
    H = expit(W.T.dot(utterFeats.T))
    targets = ind2vec(labels)
    U = np.linalg.inv((H.dot(H.T))).dot(H).dot(targets)


    with gzip.open('ELMWeights.pickle.gz', 'wb') as f:
        pickle.dump((W, U), f, protocol = 2)