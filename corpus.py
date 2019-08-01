# -*- coding: utf-8 -*-
import os 
import sys
import logging
import json
import re

import numpy as np 

from tqdm import tqdm
from collections import defaultdict
from nltk.corpus import ptb

from grammar import ContexFreeGrammar 

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Corpus(object):
    def __init__(self, root):
        super(Corpus, self).__init__() 
        self.root = root

class PtbCorpus(Corpus):
    # splits
    _TRAIN_SEC = ['02', '03', '04', '05', '06', '07', '08', '09', '10', '11', 
                  '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
    _TEST_SEC = ['23'] 
    _DEV_SEC = ['22']

    _COLLPASED_NUMBER = '-num-'
    _RE_IS_A_NUM = '^\d+(?:[,.]\d*)?$'

    _ELLIPSIS = ['*', '*?*', '0', '*T*', '*ICH*', '*U*', '*RNR*', '*EXP*', '*PPA*', '*NOT*'] 
    _WORD_TAGS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 
                  'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 
                  'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 
                  'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
    _PUNCTUATION_TAGS = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``'] 
    _PUNCTUATION_WORDS = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``', '--', 
                          ';', '-', '?', '!', '...', '-LCB-', '-RCB-'] 
    _CURRENCY_TAGWORDS = ['#', '$', 'C$', 'A$'] 

    def __init__(self, root, 
                 read_as_cnf = False,
                 collapse_unary = False,
                 collapse_number = False,
                 remove_punction = False,
                 remove_sublabel = False) -> None:
        super(PtbCorpus, self).__init__(root) 
        self.remove_sublabel = remove_sublabel
        self.remove_punction = remove_punction
        self.collapse_number = collapse_number
        self.collapse_unary = collapse_unary
        self.read_as_cnf = read_as_cnf

        self.train_fids = []
        self.test_fids = []
        self.dev_fids = []

        self.read_file_ids()

    def read_file_ids(self):
        for droot, _, files in os.walk(self.root):
            sec = droot.split('/')[-1]
            if sec in self._TRAIN_SEC:
                fids = self.train_fids
            elif sec in self._TEST_SEC:
                fids = self.test_fids
            elif sec in self._DEV_SEC:
                fids = self.dev_fids
            else:
                continue 
            for data_file in files:
                if not data_file.endswith("mrg"):
                    continue
                fids.append(os.path.join(droot, data_file))
        logger.info("train: {} fids, test: {} fids, and dev: {} fids".format(
            len(self.train_fids), len(self.test_fids), len(self.dev_fids)))

    def statistics(self):
        def remove_punction(tree):
            for subtree in tree.subtrees():
                for idx, child in enumerate(subtree):
                    if isinstance(child, str): continue
                    if all(tag not in self._WORD_TAGS for leaf, tag in child.pos()):
                        del subtree[idx]
        def process_tree(tree):
            if self.remove_punction:
                cnt = 0
                while not all([tag in self._WORD_TAGS for _, tag in tree.pos()]):
                    remove_punction(tree)
                    cnt += 1
                    if cnt > 10: assert False
            if self.collapse_number:
                for subtree in tree.subtrees(lambda t: t.height() == 2):
                    child = subtree[0]
                    assert isinstance(child, str)
                    if subtree.label() == 'CD' and re.match(self._RE_IS_A_NUM, child):
                        subtree[0] = self._COLLPASED_NUMBER
            if self.read_as_cnf:
                tree.chomsky_normal_form(horzMarkov=0)
            if self.collapse_unary:
                tree.collapse_unary(collapsePOS=True)
            return tree
        def tree_statistics(fids, grammar): 
            # build indexer
            for fid in tqdm(fids):
                trees = ptb.parsed_sents(fid)
                for tree in tqdm(trees):
                    tree = process_tree(tree)
                    grammar.read_trees(tree) 
            grammar.build_indexer() 
            # extract rules
            for fid in tqdm(fids):
                trees = ptb.parsed_sents(fid)
                for tree in tqdm(trees):
                    tree = process_tree(tree)
                    grammar.read_rules(tree) 
            grammar.build_grammar() 

        grammar = ContexFreeGrammar()
        tree_statistics(self.train_fids[:10], grammar)
        print(grammar)
        #data_statistics(self.test_fids, grammar, False)
        #data_statistics(self.dev_fids, grammar, False)
        

if __name__ == '__main__': 
    root = '/disk/scratch1/s1847450/data/data_lveg/Data.Prd/root/' 
    root = '/disk/scratch1/s1847450/data/ptb.mrg/wsj' 
    ptb_corpus = PtbCorpus(root, 
        read_as_cnf=True, 
        collapse_number=True,
        remove_punction=True,
        collapse_unary=True) 
    ptb_corpus.statistics()
    pass
