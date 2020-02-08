#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: corpus.py
Description: preprocessing dataset
Author: Barry Chow
Date: 2020/2/8 10:55 AM
Version: 0.1
"""
import numpy as np

class Corpus:
    def __init__(self,filePath):
        self.wordIndex = {}
        self.tagIndex = {}
        self.states = []
        self.observations = []

        #processing initial data to num indexs
        for line in open(filePath,'r'):
            if len(line.strip(' \r\n'))==0:
                continue
            line = [wordtag for wordtag in line.strip('\r\n').split(' ') if len(wordtag)>0]
            words = [];tags = []
            for wordtag in line:
                try:
                    (word,tag) = wordtag.split('/')
                except Exception:
                    print(wordtag)
                    continue
                if word not in self.wordIndex:
                    #start from zero
                    self.wordIndex[word]= len(self.wordIndex)
                if tag not in self.tagIndex:
                    self.tagIndex[tag] = len(self.tagIndex)
                words.append(self.wordIndex[word])
                tags.append(self.tagIndex[tag])

            self.states.append(tags)
            self.observations.append(words)

        #generate index to tag
        self.indexTag = {}
        for tag in self.tagIndex:
            self.indexTag[self.tagIndex[tag]] = tag




