#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 10:20:49 2017

@author: will
"""

import numpy as np

class CRF():
    # train a linear chain CRF with linear x-y factors
    
    def __init__(self,d,k):
        # d is the dim of X, k is the dim of y
        self.d = d
        self.k = k
        self.W = np.random.randn(d,k)/np.sqrt(d)
        self.b = np.zeros(k)
        self.W_sq = np.zeros_like(self.W) # for RMSProp
        self.b_sq = np.zeros_like(self.b) # for RMSProp
        self.TM = np.random.randn(k,k)/np.sqrt(k) # binary factors, i.e. Yt-Yt+1
        self.TM_sq = np.zeros_like(self.TM)
        self.TMexp = np.exp(self.TM)
        
    def factor_eval_XY(self,X):
        return np.dot(X,self.W)+self.b
    
    def forwardBackward(self,X):

        n = X.shape[0]
        unaryFactor = np.exp(self.factor_eval_XY(X))
        alpha = np.ones((n,self.k))
        beta = np.ones((n,self.k))
        for i in range(1,n): # forward pass
            alpha[i] = np.dot(alpha[i-1]*unaryFactor[i-1],self.TMexp)
            alpha[i] = alpha[i]/np.sum(alpha[i]) # local normalize to avoid overflow
            
        for i in range(n-2,-1,-1): # backward pass
            beta[i] = np.dot(self.TMexp,beta[i+1]*unaryFactor[i+1])
            beta[i] = beta[i]/np.sum(beta[i]) 
            
        return alpha,beta,unaryFactor
    
    def infer_Y_X(self,X,alpha,beta,unaryFactor):
        # calculate p(Y|X), where Y ranges over both time and k classes
        # needed for SGD model training. Returned value have shape T,K
        P_y = alpha*beta*unaryFactor
        Z = np.sum(P_y,1,keepdims=True) # normalizing factor
        return P_y/Z, Z
    
    def infer_YY_X(self,X,alpha,beta,unaryFactor):
        # calculate p(Yt,Yt+1|X). Returned value will have shape T-1,K,K
        n = X.shape[0]
        P_yy = np.einsum('ni,nj,ij->nij', unaryFactor[:n-1]*alpha[:n-1], \
                                          unaryFactor[1:]*beta[1:],\
                                          self.TMexp)
        Z = np.sum(P_yy,(1,2),keepdims=True)
        return P_yy/Z, Z
        
    def _unaryGrad(y,P_y):
        # used to train x-y factor
        # y should have "wide" format of shape N,K
        return y - P_y
    
    
    def _unaryUpdate(self,X,grad,r,decay):
        # r is learning rate, decay is parameter for RMSProp
        grad_b = np.sum(grad,0)
        grad_W = np.dot(X.T,grad)
        
        self.W_sq = decay * self.W_sq + (1-decay) * grad_W**2
        self.b_sq = decay * self.b_sq + (1-decay) * grad_b**2
        self.W += r * grad_W / (self.W_sq + 1e-4)
        self.b += r * grad_b / (self.b_sq + 1e-4)
    
    def _binaryGrad(freq, P_yy):
        # aggregated over T 
        return freq - np.sum(P_yy,0)
    
    def _binaryUpdate(self,grad_TM,r,decay):
        self.TM_sq = decay * self.TM_sq + (1-decay) * grad_TM**2
        self.TM += r * grad_TM / (self.TM_sq + 1e-4)
        self.TMexp = np.exp(self.TM)
        
    def fit(self,X,Y,r,decay,iterNum):
        # X,Y should have shape (N,T,d), (N,T,k), where N is the # of sequence 
        # and T is the lenth of a sequence.
        N,T,_ = X
        freq_YY = np.einsum('nti,ntj->nij',Y[:,:T-1,:],Y[:,1:,:]) # needed to calculate grad of binary factors
        for i in range(iterNum):
            index = np.random.permutation(N) 
            for j in range(N):
                x = X[index[j]]
                y = Y[index[j]]
                freq = freq_YY[index[j]]
                
                alpha,beta,unaryFactor = self.forwardBackward(x)
                P_y = self.infer_Y_X(x,alpha,beta,unaryFactor)
                P_yy = self.infer_YY_X(x,alpha,beta,unaryFactor)
                self._unaryUpdate(x,self._unaryGrad(y,P_y),r,decay)
                self._binaryUpdate(self._binaryGrad(freq,P_yy),r,decay)
                
            # monitor
            
        

            
            
''' testing '''
T = 20
d = 10
k = 3
X = np.random.randn(T,d)

        
model1 = CRF(d,k)        
alpha,beta,unaryFactor = model1.forwardBackward(X)        
P_y, Z1 = model1.infer_Y_X(X,alpha,beta,unaryFactor)
P_yy, Z2 = model1.infer_YY_X(X,alpha,beta,unaryFactor)        
        
        
        
        
        
        
        
        
        