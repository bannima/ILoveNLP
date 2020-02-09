#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: main.py
Description: 
Author: Barry Chow
Date: 2020/2/9 9:26 AM
Version: 0.1
"""
from corpus import Corpus
from models.HMM import HiddenMarkovModel


class ChineseWordSegmentation:
    def __init__(self,filePath):
        self.corpus = Corpus(filePath)
        self.hmm = HiddenMarkovModel(self.corpus)

    def test(self):
        self.hmm.testAccuracy()

    def tag(self,observations):
        obs = [self.corpus.wordIndex[observations[i]] for i in range(len(observations)) ]
        pred_tags = self.hmm.tagSequence(obs)
        states = [self.corpus.indexTag[state] for state in pred_tags]

        print("\n" + observations)
        print("/".join(states))


if __name__ =='__main__':

    #filepath = "./dataset/pku/pku_training.utf8"
    filepath = "./dataset/msr/msr_training.utf8"
    #filepath = "./dataset/cityu/cityu_training.utf8"

    wordseg = ChineseWordSegmentation(filepath)
    wordseg.test()
    wordseg.tag("最后，我从北京祝大家新年快乐！")