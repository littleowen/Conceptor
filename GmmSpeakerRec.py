from scipy.io import wavfile
import numpy as np
import os
from IPython.display import display, Audio
import pickle
import librosa
from silence import remove_silence

from skgmm import GMMSet, GMM

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
        signal_new = remove_silence(fs, signal)
        hop_length = np.min([0.016 * fs, 512])
        mfcc = librosa.feature.mfcc(y = signal_new, sr = fs, n_mfcc = 15, hop_length = hop_length)
        mfcc = mfcc.T   
        mu = np.mean(mfcc, axis = 0)
        sigma = np.std(mfcc, axis = 0)
        feature = (mfcc - mu) / sigma
        self.features.append(feature)
        self.classes.append(name)

    def _get_gmm_set(self):
        return GMMSet()

    def train(self):
        self.gmmset = self._get_gmm_set()
        for name, feats in zip(self.classes, self.features):
            self.gmmset.fit_new(feats, name)

    def predict(self, signal, fs = 44100):
        signal_new = remove_silence(fs, signal)
        # if len(signal_new) < len(signal) / 4:
        #     return "Silence"
        hop_length = np.min([0.016 * fs, 512])
        mfcc = librosa.feature.mfcc(y = signal_new, sr = fs, n_mfcc = 15, hop_length = hop_length)
        mfcc = mfcc.T    
        mu = np.mean(mfcc, axis = 0)
        sigma = np.std(mfcc, axis = 0)
        feature = (mfcc - mu) / sigma
        return self.gmmset.predict_one(feature)
    
    @staticmethod
    def totime(secs):
        m, s = divmod(secs, 60)
        h, m = divmod(m, 60)
        return h, m, s
    
    def showresult(self, signal, fs, head, disp):
        print("%d:%02d:%02d" % (self.totime(head)), self.predict(
                    signal, fs))
        try:
            if disp:
                display(Audio(data = signal, rate = fs))
        except:
                pass
   
    def recognize(self, signal, step = 1, duration = 1.5, fs = 44100, disp = True):
        head = 0
        totallen = np.round(signal.shape[0] / fs).astype(int)
        print('Recognition results:')
        while head < totallen:
            tail = head + duration
            if tail > totallen:
                tail = totallen
            signali = signal[fs * head : np.min([fs * tail, fs * totallen])]           
            self.showresult(signali, fs, head, disp)
            head += step
        #signali = signal[fs * (head - step):]
        #self.showresult(signali, fs, head, disp)
        
            
            
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

            
