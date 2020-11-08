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
from nltk.corpus import BracketParseCorpusReader, LazyCorpusLoader

from grammar import ContexFreeGrammar 

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logging.basicConfig(level = logging.INFO)

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
    _CURRENCY_TAGWORDS = ['#', '$', 'C$', 'A$'] # tags: # & $; words: $, C$, and A$  

    def __init__(self, root, reader,
                 read_as_cnf = False,
                 lowercase_word = False,
                 collapse_unary = False,
                 collapse_number = False,
                 remove_punction = False,
                 remove_sublabel = False) -> None:
        super(PtbCorpus, self).__init__(root) 
        self.remove_sublabel = remove_sublabel
        self.remove_punction = remove_punction
        self.collapse_number = collapse_number
        self.lowercase_word = lowercase_word
        self.collapse_unary = collapse_unary
        self.read_as_cnf = read_as_cnf
        self.reader = reader

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
        chain_rule_lengths = defaultdict(int)
        def remove_punction(tree, tags_kept):
            for subtree in tree.subtrees():
                for idx, child in enumerate(subtree):
                    if isinstance(child, str): continue
                    if all(tag not in tags_kept for leaf, tag in child.pos()):
                        del subtree[idx]
        def reduce_label(tree):
            for subtree in tree.subtrees():
                labels = subtree.label().split('+')
                chain_rule_lengths[len(labels) - 1] += 1
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
                    if not isinstance(child, str):
                        print(tree)
                    assert isinstance(child, str)
                    if self.lowercase_word:
                        subtree[0] = child.strip().lower()
                    if not self.collapse_number:
                        continue
                    if subtree.label() == 'CD' and re.match(self._RE_IS_A_NUM, child):
                        subtree[0] = self._COLLPASED_NUMBER
            if self.read_as_cnf:
                tree.chomsky_normal_form(horzMarkov=0)
            if self.collapse_unary:
                tree.collapse_unary(collapsePOS=True)

            reduce_label(tree) # unary chain may be long and sparse

            return tree
        def tree_statistics(fids, grammar): 
            # build indexer
            for fid in tqdm(fids):
                trees = self.reader.parsed_sents(fid)
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
                trees = self.reader.parsed_sents(fid)
                for tree in tqdm(trees):
                    tree = process_tree(tree)
                    grammar.read_rules(tree) 
            grammar.build_grammar() 
        def tree_fringe_len(fids):
            lengths = defaultdict(int)
            for fid in tqdm(fids):
                trees = self.reader.parsed_sents(fid)
                for tree in tqdm(trees):
                    length = len(tree.leaves())
                    lengths[length] += 1 
            return lengths
        def print_tl(tl, mlen):
            """
            items = list(tl.items())
            items = sorted(items, key=lambda x: x[0])
            k = [x for x, y in items]
            v = [y for x, y in items]
            print(k)
            print(v)
            """
            for idx in range(1, mlen + 1):
                print(tl[idx], end=',')
            print()
        tl_train = tree_fringe_len(self.train_fids[:])
        #print_tl(tl)
        tl_test = tree_fringe_len(self.test_fids)
        #print_tl(tl)
        tl_dev = tree_fringe_len(self.dev_fids)
        #print_tl(tl)
        mlen = max(max(max(list(tl_train.keys())), max(list(tl_test.keys()))), max(list(tl_dev.keys())))

        print(list(range(1, mlen + 1)))
        print_tl(tl_train, mlen)
        print_tl(tl_test, mlen)
        print_tl(tl_dev, mlen)
        #grammar = ContexFreeGrammar()
        #tree_statistics(self.train_fids[:], grammar)
        #data_statistics(self.test_fids, grammar, False)
        #data_statistics(self.dev_fids, grammar, False)
        #print(chain_rule_lengths)
        #print(grammar)


def save_split(fids, ofile):
    with open(ofile, "w") as fw:
        for f in fids:
            sentences = ptb.parsed_sents(f)
            for tree in sentences:
                tree = tree.pformat(margin=sys.maxsize).strip()
                fw.write(tree + "\n")

def main_split_chinese_tb():
    """ train: 001–270, 400–1151; devel: 301–325; test: 271-300.
    """
    iroot = "/home/s1847450/data/Data.Prd/ctb_mrg/" 
    train_fs, devel_fs, test_fs = list(), list(), list()
    for f in os.listdir(iroot):
        f = iroot + f
        if not os.path.isfile(f): 
            continue
        if not f.endswith("mrg"): 
            continue
        x = re.findall("chtb_(\d+)\.mrg", f)
        if not x: print("unparsable {}".format(f))
        fid = int(x[0])
        if fid >= 271 and fid <= 300:
            test_fs.append(f)
        elif fid >= 301 and fid <= 325:
            devel_fs.append(f)
        elif (fid >= 1 and fid <= 270) or (fid >= 400 and fid <= 1151):
            train_fs.append(f)
    print("train: {}; devel: {}; test: {}".format(len(train_fs), len(devel_fs), len(test_fs)))
    
    oroot = "/home/s1847450/data/spmrl2014/spmrl.punct/"
    
    lang = "chinese"
    save_split(train_fs, oroot + "{}-train.txt".format(lang))
    save_split(devel_fs, oroot + "{}-valid.txt".format(lang))
    save_split(test_fs, oroot + "{}-test.txt".format(lang))


PTB_TRAIN_SEC = ['02', '03', '04', '05', '06', '07', '08', '09', '10', '11', 
              '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
PTB_TEST_SEC = ['23'] 
PTB_DEV_SEC = ['22']

def read_ptb_ids(root):
    train_fs, devel_fs, test_fs = list(), list(), list()
    for droot, _, files in os.walk(root):
        sec = droot.split('/')[-1]
        if sec in PTB_TRAIN_SEC:
            fids = train_fs
        elif sec in PTB_TEST_SEC:
            fids = test_fs
        elif sec in PTB_DEV_SEC:
            fids = devel_fs
        else:
            continue 
        for data_file in files:
            if not data_file.endswith("mrg"):
                continue
            fids.append(os.path.join(droot, data_file))
    logger.info("train: {} fids, test: {} fids, and dev: {} fids".format(
        len(train_fs), len(test_fs), len(devel_fs)))
    return train_fs, devel_fs, test_fs

def main_split_english_tb():
    iroot = "/home/s1847450/data/ptb.mrg/wsj/" 
    train_fs, devel_fs, test_fs = read_ptb_ids(iroot)

    oroot = "/home/s1847450/data/spmrl2014/spmrl.punct/"
    
    lang = "english"
    save_split(train_fs, oroot + "{}-train.txt".format(lang))
    save_split(devel_fs, oroot + "{}-valid.txt".format(lang))
    save_split(test_fs, oroot + "{}-test.txt".format(lang))


if __name__ == '__main__': 
    #main_split_chinese_tb()
    main_split_english_tb()
    sys.exit(0)

    root = '/disk/scratch1/s1847450/data/Data.Prd/ctb_dir/' 
    ctb = BracketParseCorpusReader(root, r'(?!\.).*\.mrg')
    ctb_corpus = PtbCorpus(root, ctb,
        read_as_cnf = True, 
        collapse_number = False,
        remove_punction = False,
        lowercase_word = False, 
        collapse_unary = True) 
    ctb_corpus.statistics()

    """
    root = '/disk/scratch1/s1847450/data/Data.Prd/wsj_dir/' 
    ptb_corpus = PtbCorpus(root, ptb, 
        read_as_cnf = True, 
        collapse_number = True,
        remove_punction = True,
        lowercase_word = True, 
        collapse_unary = True) 
    ptb_corpus.statistics()
    """
