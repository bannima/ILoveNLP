#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: corpus.py
Description: 
Author: Barry Chow
Date: 2020/2/9 9:26 AM
Version: 0.1
"""

class Corpus:
    def __init__(self,filePath):
        self.observations = []
        self.states = []
        self.wordIndex = {}
        self.tagIndex = {}
        self.indexTag = {}
        for line in open(filePath):
            words = []; state = []
            if len(line.strip(' \r\n'))==0:
                continue
            line = [word for word in line.strip(' \r\n').split(" ") if len(word)>0]
            tags = "".join([self.__tagState(word) for word in line])
            for index in range(len(line)):
                word,tag = line[index],tags[index]
                if word not in self.wordIndex:
                    self.wordIndex[word] = len(self.wordIndex)
                if tag not in self.tagIndex:
                    self.tagIndex[tag] = len(self.tagIndex)
                words.append(self.wordIndex[word]);state.append(self.tagIndex[tag])

            self.observations.append(words);self.states.append(state)

        #index=>tag
        for tag in self.tagIndex:
            self.indexTag[self.tagIndex[tag]]=tag

    #given word sequence, return the state sequence
    def __tagState(self,word):
        if len(word)==1:
            return 'S'
        else:
            return 'B'+'M'*(len(word)-2)+'E'





