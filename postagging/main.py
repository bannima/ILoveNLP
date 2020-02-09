#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: main.py
Description: 
Author: Barry Chow
Date: 2020/2/8 9:05 AM
Version: 0.1
"""
import numpy as np
from corpus import Corpus
from models.HMM import HiddenMarkovModel

class POSTagging:
    def __init__(self):
        self.corpus = Corpus('./dataset/train_utf16.tag')
        self.hmm = HiddenMarkovModel(self.corpus)

    def tag(self,obs):
        word_index = [self.corpus.wordIndex[word] for word in observations.strip(' \r\n').split(' ') if len(word) != 0]
        pred_states = self.hmm.tagSequence(word_index)
        pred_tags = [self.corpus.indexTag[index] for index in pred_states]

        print("\n" + "\t".join(observations.strip(" \r\n").split()))
        print("\t".join(pred_tags))

    def test(self):
        self.hmm.testAccuracy()

if __name__ =='__main__':

    postagging = POSTagging()
    postagging.test()

    observations = "他  在  出版  的  畫冊  中  ，"
    postagging.tag(observations)
    observations = "但  部分  學校  早  有  合作  意願  。"
    postagging.tag(observations)










