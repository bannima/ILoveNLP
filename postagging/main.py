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
from model.corpus import Corpus
from model.HMM import HiddenMarkovModel

if __name__ =='__main__':
    corpus = Corpus('./dataset/train_utf16.tag')
    hmm = HiddenMarkovModel(corpus)
    hmm.testAccuracy()

    observations = "他  在  出版  的  畫冊  中  ，"
    preds = hmm.tagSequence(observations)

    observations = "但  部分  學校  早  有  合作  意願  。"
    preds = hmm.tagSequence(observations)






