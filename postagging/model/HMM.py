#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: HMM.py
Description: 
Author: Barry Chow
Date: 2020/2/8 9:05 AM
Version: 0.1
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

class HiddenMarkovModel:

    def __init__(self,corpus):
        self.states = corpus.states
        self.observations = corpus.observations
        self.nStates = len(corpus.tagIndex)
        self.nObservations = len(corpus.wordIndex)
        self.indexTag = corpus.indexTag
        self.wordIndex = corpus.wordIndex

        #split train and test dataset
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.observations,self.states,
                                                            test_size=0.3, random_state=42)
        self.__supervisedCalcParams()

    # calc HMM params using surprivesed methods
    def __supervisedCalcParams(self):
        self.A = np.zeros((self.nStates, self.nStates))
        self.B = np.zeros((self.nStates, self.nObservations))
        self.Pi = {}

        for observation,state in zip(self.X_train,self.Y_train):
            #calc init state probs
            if state[0] not in self.Pi:
                self.Pi[state[0]]=1
            else:
                self.Pi[state[0]]+=1
            #calc B
            for word,tag in zip(observation,state):
                self.B[tag,word]+=1
            #calc A
            for tagindex in range(0,len(state)-1):
                self.A[state[tagindex],state[tagindex+1]]+=1

        #calc Pi
        totalInitStates = sum(self.Pi.values())
        for state in range(self.nStates):
            if state not in self.Pi:
                self.Pi[state] = 0.001
            self.Pi[state] = float(self.Pi[state])/totalInitStates


        #laplace smoothing for A and B
        self.A = normalize(self.A+0.01,axis=0,norm='l1')
        self.B = normalize(self.B+0.01,axis=0,norm='l1')

        print("N X N: ",self.A.shape)
        print("N X M: ",self.B.shape)

    #test accuracy
    def testAccuracy(self):
        correct = 0;total=0
        for observation,states in zip(self.X_test,self.Y_test):
            pred_states = self.__viterbi(observation)
            for pred,trueTag in zip(pred_states,states):
                if pred==trueTag:
                    correct+=1
                total+=1
        print("The test dataset accuracy = ",float(correct)/total)

    #given observations, pred the most likely hidden states using viterbi algorithms
    def __viterbi(self, observation):
        self.delta = np.zeros((self.nStates, len(observation)))
        self.path = np.zeros((self.nStates, len(observation)))

        for t in range(len(observation)):
            word = observation[t]
            for state in range(self.nStates):
                if t==0:
                    self.delta[state,t] = self.Pi[state]*self.B[state,word]
                    continue
                probs = self.delta[:,t-1].T*self.A[:,state].T
                self.delta[state,t] = max(probs)*self.B[state,word]
                self.path[state,t-1] = np.argmax(probs)

        #traceback for the most likely state path
        max_path = []
        current = np.argmax(self.delta[:,-1])
        max_path.append(current)
        for t in reversed(range(len(observation)-1)):
            current = int(self.path[current,t])
            max_path.append(current)

        return list(reversed(max_path))

    #given observations, report the tag sequence
    def tagSequence(self,observations):

        word_index = [self.wordIndex[word] for word in observations.strip(' \r\n').split(' ') if len(word)!=0]
        pred_states = self.__viterbi(word_index)
        pred_tags = [self.indexTag[index] for index in pred_states]

        print("\n"+"\t".join(observations.strip(" \r\n").split()))
        print("\t".join(pred_tags))
























