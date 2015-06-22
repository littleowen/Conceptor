'''
Created on June 20, 2015

@author: littleowen
@note: Pattern Recognition Component
'''

import numpy as np
import scipy as sp
import numpy.matlib
import conceptor.util as util
import conceptor.logic as logic
from scipy import interpolate
import sys

class Recognizer:
    """
    An implementaion of reservoir network
  
    """
    
    
    def __init__(self):
    
        """
        Initialize conceptor network

        @param dim: dimension of the pattern vectors to be recognized
        """
        self.dim = 0
        self.I = np.asarray([])
        self.classnum = 0
        self.CPoss = []
        self.RPoss = []
        self.ROthers = []
        self.CNegs = []
        self.Cs_best_pos = []
        self.Cs_best_neg = []     
        self.aps_pos = []
        self.aps_neg = []
        self.apt_pos = 0
        self.apt_neg = 0


    @staticmethod
    def evidence(testdata,
                 C_list):
        """
        Make predictions on test dataset using a list of conceptors

        @param testdata: the testdata to be recognized
        """
        evidence_list = []
        for C in C_list:
            evidence = sum(testdata * (C.dot(testdata)))
            evidence_list.append(evidence)
        evidence = np.row_stack(evidence_list)
        return np.argmax(evidence, axis = 0), evidence
    
    


    def compute_conceptors(self,
                           all_train_states,
                           apN = 9):
        """
        Given a list of training data from all classes, compute conceptor and correlation matrices

        @param all_train_states: a list of training datasets from all classes
        @param apN: the highest exponential to consider for aperture adaption 
        """

        statesAllClasses = np.hstack(all_train_states)
        Rall = statesAllClasses.dot(statesAllClasses.T)
        self.classnum = len(all_train_states)
        self.dim = all_train_states[0].shape[0]
        self.I = np.eye(self.dim)
        for i in range(self.classnum):
            R = all_train_states[i].dot(all_train_states[i].T)
            Rnorm = R / all_train_states[i].shape[1]
            self.RPoss.append(Rnorm)
            ROther = Rall - R
            ROthersNorm = ROther / (statesAllClasses.shape[1] - all_train_states[i].shape[1])
            self.ROthers.append(ROthersNorm)
            CPossi = []
            CNegsi = []
            for api in range(apN):
                C = Rnorm.dot(np.linalg.inv(Rnorm + (2 ** float(api)) ** (-2) * self.I))
                CPossi.append(C)
                COther = ROthersNorm.dot(np.linalg.inv(ROthersNorm + (2 ** float(api)) ** (-2) * self.I))
                CNegsi.append(self.I - COther)
            self.CPoss.append(CPossi)
            self.CNegs.append(CNegsi)

    @staticmethod        
    def compute_aperture(C_list,
                         apN = 9):
        """
        Given a list of Conceptors, compute the best aperture using the delta measure

        @param C_list: a list (differnt classes) of lists (different apertures) of conceptor matrices
        @param apN: the highest exponential to consider for aperture adaption 
        """

        best_aps = []
        apsExploreExponents = np.asarray(range(apN))
        intPts = np.arange(apsExploreExponents[0], apsExploreExponents[-1] + 0.01, 0.01)
        classnum = len(C_list)
        for i in range(classnum):
            norm = np.zeros(apN)
            for api in range(apN):
                norm[api] = np.linalg.norm(C_list[i][api], 'fro') ** 2      
            f = interpolate.interp1d(np.arange(apN), norm, kind="cubic")
            norm_inter = f(intPts)
            norm_inter_grad = (norm_inter[1:] - norm_inter[0 : -1]) / 0.01
            max_ind = np.argmax(np.abs(norm_inter_grad), axis = 0)    
            best_aps.append(2 ** intPts[max_ind])  
        return best_aps
    
    
    def aperture_adjust(self,
                        apN = 9):
        """
        Compute the best apertures for positive and negtive conceptors

        @param apN: the highest exponential to consider for aperture adaption 
        """
        CNegs = [[logic.NOT(C) for C in Clist] for Clist in self.CNegs]

        self.aps_pos = self.compute_aperture(self.CPoss, apN)
        self.apt_pos = np.mean(self.aps_pos)
        self.aps_neg = self.compute_aperture(CNegs, apN)
        self.apt_neg = np.mean(self.aps_neg)

    @staticmethod
    def combine_evidence(evidence_pos,
                         evidence_neg):

        """
        Make predictions based on both positive and negative evidence   

        @param evidence_pos: positive evidence
        @param evidence_neg: negative evidence 

        """
        minValPos = np.amin(evidence_pos, axis = 0)
        maxValPos = np.amax(evidence_pos, axis = 0)

        rangePos = maxValPos - minValPos

        minValNeg = np.amin(evidence_neg, axis = 0)
        maxValNeg = np.amax(evidence_neg, axis = 0)

        rangeNeg = maxValNeg - minValNeg

        posEvVecNorm = (evidence_pos - np.tile(minValPos, (evidence_pos.shape[0],1))) / np.tile(
            rangePos, (evidence_pos.shape[0],1))

        negEvVecNorm = (evidence_neg - np.tile(minValNeg, (evidence_neg.shape[0],1))) / np.tile(
            rangeNeg, (evidence_neg.shape[0],1))

        combEv = posEvVecNorm + negEvVecNorm
        results_comb = np.argmax(combEv, axis = 0)
        return results_comb, combEv

    
    def compute_best_conceptors(self):

        """
        Compute the best conceptors using the adapted aperture 

        @param R_list: a list of correlation matrix
        @param best_apt: the chosen best aperture 
        """
        for i in range(self.classnum):
            C_best_pos = self.RPoss[i].dot(np.linalg.inv(self.RPoss[i] + self.apt_pos ** (-2) * self.I))
            self.Cs_best_pos.append(C_best_pos)    
            C_best_neg = self.ROthers[i].dot(np.linalg.inv(self.ROthers[i] + self.apt_neg ** (-2) * self.I))
            self.Cs_best_neg.append(C_best_neg) 
        self.Cs_best_neg = [logic.NOT(C) for C in self.Cs_best_neg]
        
    def train(self,
             all_train_states,
             apN = 9):
        """
        Training for pattern recognition
        """
        
        self.compute_conceptors(states_list_train, apN)

        self.aperture_adjust(apN)
    
        self.compute_best_conceptors()
        
    def predict(self,
                all_states_test):
        results_pos, evidence_pos = self.evidence(all_states_test, self.Cs_best_pos)
        results_neg, evidence_neg = self.evidence(all_states_test, self.Cs_best_neg)    
        results_comb, combEv = self.combine_evidence(evidence_pos, evidence_neg)
        return results_comb
