import scipy.io.wavfile as wav
import numpy as np
import os
import pickle
from features import mfcc
from silence import remove_silence

from pycaspgmm import GMMSet, GMM

class GMMRec(object):

    def __init__(self,
                 ubmfn = None,
                 reject_threshold = 10):
        self.features = []
        self.gmmset = GMMSet()
        self.classes = []
        self.reject_threshold = reject_threshold
        if ubmfn is not None:
            self.ubm = self.load(ubmfn)

    def enroll(self, name, signal, fs = 44100):
        if len(signal.shape) > 1:
            signal = signal[:, 0]
        signal_new = remove_silence(fs, signal)
        mfcc_vecs = mfcc(signal_new, fs, numcep = 15)
        self.enroll_feat(name, mfcc_vecs)
        
    def enroll_feat(self, name, mfcc_vecs):
        mu = np.mean(mfcc_vecs, axis = 0)
        sigma = np.std(mfcc_vecs, axis = 0)
        feature = (mfcc_vecs - mu) / sigma
        feature = feature.astype(np.float32)
        self.features.append(feature)
        self.classes.append(name)

    def _get_gmm_set(self):
        return GMMSet()

    def train(self):
        self.gmmset = self._get_gmm_set()
        for name, feats in zip(self.classes, self.features):
            self.gmmset.fit_new(feats, name)
            
    def predict_feat(self, mfcc_vecs):
        mu = np.mean(mfcc_vecs, axis = 0)
        sigma = np.std(mfcc_vecs, axis = 0)
        feature = (mfcc_vecs - mu) / sigma
        feature = feature.astype(np.float32)
        return self.gmmset.predict_one(feature)

    def predict(self, signal, fs = 44100):
        if len(signal.shape) > 1:
            signal = signal[:, 0]
        signal_new = remove_silence(fs, signal)
        # if len(signal_new) < len(signal) / 4:
        #     return "Silence"
        mfcc_vecs = mfcc(signal_new, fs, numcep = 15)
        return self.predict_feat(mfcc_vecs)
    
    @staticmethod
    def totime(secs):
        m, s = divmod(secs, 60)
        h, m = divmod(m, 60)
        return h, m, s
    
    def showresult(self, signal, fs, head):
        print("%d:%02d:%02d" % (self.totime(head)), self.predict(
                    signal, fs))

   
    def recognize(self, signal, step = 1, duration = 1.5, fs = 44100):
        if len(signal.shape) > 1:
            signal = signal[:, 0]
        head = 0
        totallen = np.round(signal.shape[0] / fs).astype(int)
        print('Recognition results:')
        while head < totallen:
            tail = head + duration
            if tail > totallen:
                tail = totallen
            signali = signal[fs * head : np.min([fs * tail, fs * totallen])]           
            self.showresult(signali, fs, head)
            head += step
            
    def dump(self, fname, part = None):
        with open(fname, 'wb') as f:
            if part is None:
                pickle.dump(self, f, -1)
            else:
                pickle.dump(part, f, -1)

    @staticmethod
    def load(fname):
        with open(fname, 'rb') as f:
            R = pickle.load(f)
            return R

            
