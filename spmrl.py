# -*- coding: utf-8 -*-
import os 
import sys
import json
import re
import logging
import pathlib

import numpy as np 

from tqdm import tqdm
from collections import defaultdict
from nltk.corpus import ptb
from nltk import Tree

from corpus import Corpus
from grammar import ContexFreeGrammar 

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
#logging.basicConfig(level = logging.INFO)

"""
BASQUE: 'TOP'
GERMAN: 'VROOT'
FRENCH: ''
HEBREW: 'TOP'
HUNGARIAN: 'ROOT'
KOREAN: 'TOP'
POLISH: '?'
SWEDISH: ''
"""

DATAS = ['dev', 'test', 'train', 'train5k']
LANGS = ['BASQUE', 'GERMAN', 'FRENCH', 'HEBREW', 'HUNGARIAN', 'KOREAN', 'POLISH', 'SWEDISH'] 
ROOTS = ['TOP', 'VROOT', '', 'TOP', 'ROOT', 'TOP', None, ''] 

def remove_morph_feature(tree):
    out = tree.pformat(margin=sys.maxsize).strip()          
    while re.search('(\#\#.*?\#\#)', out) is not None:
        out = re.sub('(\#\#.*?\#\#)', '', out)
    out = out.replace(' )', ')')
    out = re.sub('\s{2,}', ' ', out)
    return out 

def remove_morph_feature_io(ifile, ofile):
    # build indexer
    trees = ptb.parsed_sents(ifile)
    with open(ofile, 'w') as fw:
        for tree in tqdm(trees):
            children = list(tree.subtrees())
            if tree.label() in ROOTS and len(children[0]) == 1:
                tree = children[0][0]
            #print(tree)
            #print()
            """
            root = Tree('ROOT', [])
            root.append(tree)
            tree = root
            """
            tree_str = remove_morph_feature(tree)
            #print(tree_str)
            fw.write(tree_str + '\n')
            #sys.exit(0)

def main_remove_morph_feature():
    for lang in LANGS[-1:]:
        for data in DATAS:
            root = '/disk/scratch1/s1847450/data/spmrl2014/SPMRL/{}_SPMRL/{}/ptb/{}/'
            lang_ = lang.lower() if lang == 'SWEDISH' else lang.capitalize()
            name = '{}.{}.gold.ptb'.format(data, lang_)
            
            iroot = root.format(lang, 'gold', data)
            ifile = iroot + name 
            if not pathlib.Path(ifile).is_file():
                print('--skip {}'.format(ifile))
                continue
            
            oroot = root.format(lang, 'proc', data)
            pathlib.Path(oroot).mkdir(parents=True, exist_ok=True) 
            ofile = oroot + name  
            remove_morph_feature_io(ifile, ofile)

class SpmrlCorpus(Corpus):

    _COLLPASED_NUMBER = '-num-'
    _RE_IS_A_NUM = '^\d+(?:[,.]\d*)?$'

    def __init__(self, root, 
                 read_as_cnf = False,
                 lowercase_word = False,
                 collapse_unary = False,
                 collapse_number = False,
                 remove_punction = False,
                 remove_sublabel = False) -> None:
        super(SpmrlCorpus, self).__init__(root) 
        self.remove_sublabel = remove_sublabel
        self.remove_punction = remove_punction
        self.collapse_number = collapse_number
        self.lowercase_word = lowercase_word
        self.collapse_unary = collapse_unary
        self.read_as_cnf = read_as_cnf

        self.train_fids = []
        self.test_fids = []
        self.dev_fids = []

        self.read_file_ids()

    def read_file_ids(self):
        def read_file_list(root):
            fids = []
            for droot, _, files in os.walk(root):
                for data_file in files:
                    if not data_file.endswith("ptb"):
                        continue
                    fids.append(os.path.join(droot, data_file))
            return fids
        # train/test/dev
        proot = self.root + 'train/'
        self.train_fids += read_file_list(proot)
        proot = self.root + 'test/'
        self.test_fids += read_file_list(proot)
        proot = self.root + 'dev/'
        self.dev_fids += read_file_list(proot)
        logger.info("train: {} fids, test: {} fids, and dev: {} fids".format(
            len(self.train_fids), len(self.test_fids), len(self.dev_fids)))

    def statistics(self):
        def remove_punction(tree, tags_kept):
            for subtree in tree.subtrees():
                for idx, child in enumerate(subtree):
                    if isinstance(child, str): continue
                    if all(tag not in tags_kept for leaf, tag in child.pos()):
                        del subtree[idx]
        def reduce_label(tree):
            for subtree in tree.subtrees():
                labels = subtree.label().split('+')
                if len(labels) > 2:
                    new_label = '{}+{}'.format(labels[0], labels[-1])
                    subtree.set_label(new_label) 
             
        def process_tree(tree):
            if self.remove_punction:
                cnt = 0
                tags_kept = self._WORD_TAGS + self._CURRENCY_TAGWORDS 
                while not all([tag in tags_kept for _, tag in tree.pos()]):
                    remove_punction(tree, tags_kept)
                    cnt += 1
                    if cnt > 10: assert False
            if self.collapse_number or self.lowercase_word:
                for subtree in tree.subtrees(lambda t: t.height() == 2):
                    child = subtree[0]
                    assert isinstance(child, str)
                    if self.lowercase_word:
                        subtree[0] = child.strip().lower()
                    if not self.collapse_number:
                        continue
                    #if subtree.label() == 'CD' and re.match(self._RE_IS_A_NUM, child):
                    if re.match(self._RE_IS_A_NUM, child):
                        subtree[0] = self._COLLPASED_NUMBER
            if self.read_as_cnf:
                tree.chomsky_normal_form(horzMarkov=0)
            if self.collapse_unary:
                tree.collapse_unary(collapsePOS=True)
            
            #reduce_label(tree) # unary chain may be long and sparse

            return tree

        def tree_statistics(fids, grammar): 
            # build indexer
            for fid in tqdm(fids):
                trees = ptb.parsed_sents(fid)
                for tree in tqdm(trees):
                    #print(tree)
                    #print()
                    tree = process_tree(tree)
                    #print(tree)
                    #sys.exit(0)
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
        #tree_statistics(self.train_fids[:10], grammar)
        #tree_statistics(self.test_fids, grammar, False)
        tree_statistics(self.dev_fids, grammar)
        print(grammar)
        

if __name__ == '__main__': 

    #main_remove_morph_feature()
    
    #sys.exit(0) 

    lang = 'GERMAN_SPMRL'
    root = '/disk/scratch1/s1847450/data/spmrl2014/SPMRL/{}/proc/ptb/'.format(lang)
    ptb_corpus = SpmrlCorpus(root, 
        read_as_cnf = True, 
        collapse_number = True,
        remove_punction = False,
        lowercase_word = True, 
        collapse_unary = False) 
    ptb_corpus.statistics()



