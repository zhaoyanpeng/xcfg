# -*- coding: utf-8 -*-
import os 
import sys
import logging
import json
import re

import numpy as np 

from collections import defaultdict, Counter
from nltk.corpus import ptb

from operand import Operand, FloatOperand, GaussianMxitureOperand

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Grammar(object):
    def __init__(self):
        super(Grammar, self).__init__()

class RegularGrammar(Grammar):
    def __init__(self):
        super(RegularGrammar, self).__init()
        
        self.tokens = Field()
        self.pos_tags = Field()
        self.urules = Field()
        self.lrules = Field()

class ContexFreeGrammar(Grammar):
    """ Dependant of XCFG: X can be W, P, or N indicating weighted, 
        probabilistic, and neural CFGs, respectively.
    """
    def __init__(self):
        super(ContexFreeGrammar, self).__init__()
        self.tokens = Field(padding=True, keep_firstk=5)  
        self.phrase_labels = Field()  
        self.pos_tags = Field()   
        self.brules = Field() # binary rules
        self.urules = Field() # unary rules
        self.lrules = Field() # lexicon rules
        self.indexer_is_ok = False

    def __repr__(self):
        return {
            'phrase_labels': self.phrase_labels,
            'pos_tags': self.pos_tags,
            'brules': self.brules,
            'urules': self.urules,
            'lrules': self.lrules,
            'tokens': self.tokens,
        }
    
    def __str__(self):
        fields = self.__repr__()
        s = '\n'
        for k, v in fields.items():
            s += '\n--: {} # {}:\n\n{}\n'.format(k, v.len(), v) 
        return s

    def build_indexer(self):
        self.tokens.build_index()
        self.pos_tags.build_index()   
        self.phrase_labels.build_index()  
        self.indexer_is_ok = True

    def build_grammar(self): 
        self.brules.build_index()
        self.urules.build_index()
        self.lrules.build_index()

    def read_trees(self, tree):
        for subtree in tree.subtrees(lambda t: t.height() == 2):
            child = subtree[0]
            assert isinstance(child, str)
            if len(subtree) != 1: continue

            self.tokens.add(child)
            self.pos_tags.add(subtree.label())
        for subtree in tree.subtrees(lambda t: t.height() > 2):
            plabel = subtree.label()
            if len(subtree) == 2:
                llabel = subtree[0].label()
                rlabel = subtree[1].label()
                self.phrase_labels.add([plabel, llabel, rlabel])
            elif len(subtree) == 1:
                clabel = subtree[0].label()
                self.phrase_labels.add([plabel, clabel])
            else:
                continue

    def read_rules(self, tree):
        if not self.indexer_is_ok: 
            raise ValueError('Indexers have not been built.')
        for subtree in tree.subtrees(lambda t: t.height() == 2):
            child = subtree[0]
            assert isinstance(child, str)
            if len(subtree) != 1: continue
            pid = self.phrase_labels.idx(subtree.label())
            cid = self.tokens.idx(child) 

            rule = Lrule(self.phrase_labels, pid, cid) 
            self.lrules.add(rule)
        for subtree in tree.subtrees(lambda t: t.height() > 2):
            pid = self.phrase_labels.idx(subtree.label())
            if len(subtree) == 2:
                lid = self.phrase_labels.idx(subtree[0].label())
                rid = self.phrase_labels.idx(subtree[1].label()) 
                rule = Brule(self.phrase_labels, pid, lid, rid)
                self.brules.add(rule)
            elif len(subtree) == 1:
                cid = self.phrase_labels.idx(subtree[0].label())
                rule = Urule(self.phrase_labels, pid, cid)
                self.urules.add(rule)
            else:
                continue

class Field(object):
    _PAD = '<pad>' 
    _UNK = '<unk>'
    _BOS = '<bos>'
    _EOS = '<eos>'
    _DUMMY = {_PAD: 0, _UNK: 1, _BOS: 2, _EOS: 3}
    def __init__(self, padding=False, keep_firstk=0):
        super(Field, self).__init__()
        self.keep_firstk = keep_firstk 
        self.padding = padding
        self.counter = Counter() 
        self.idx2tok = {}
        self.tok2idx = {}
        if padding: # global initialization
            for k, v in self._DUMMY.items():
                self.tok2idx[k] = v
                self.idx2tok[v] = k

    def __str__(self):
        s = []
        for k, v in self.idx2tok.items():
            cnt = self.counter.get(v, 0)
            s.append('id: {:<6d} cnt: {:<6d}\t {}'.format(k, cnt, v)) 
        return '\n'.join(s)
    
    def idx(self, token):
        if token in self.tok2idx:
            return self.tok2idx[token]
        else:
            return self.tok2idx[self._UNK]
    
    def str(self, index):
        if index in self.idx2tok:
            return self.idx2tok[index]
        else:
            return self._UNK
    
    def len(self):
        return len(self.tok2idx)

    def add(self, token):
        if isinstance(token, list):
            for tok in token:
                self.counter[tok] += 1
        else:
            self.counter[token] += 1
        
    def build_index(self):
        if self.keep_firstk > 100: # first k most common tokens 
            for idx, (k, v) in enumerate(self.counter.most_common()):
                if idx >= self.keep_firstk: break
                itoken = len(self.tok2idx)
                self.tok2idx[k] = itoken 
                self.idx2tok[itoken] = k
        else: # tokens with a frequency above self.keep_firstk
            for k, v in self.counter.most_common():
                if v < self.keep_firstk: continue
                itoken = len(self.tok2idx)
                self.tok2idx[k] = itoken 
                self.idx2tok[itoken] = k
         

class Rule(object):
    
    def __init__(self, vocab, pid):
        super(Rule, self).__init__()
        self.vocab = vocab
        self.idx = None
        self.pid = pid 
        self.p = None 

        self.fun = None # a function

    def index(self):
        self.pid = self.vocab.idx[self.p]

class Urule(Rule):
    
    def __init__(self, indexer, pid, ucid):
        super(Urule, self).__init__(indexer, pid)
        self.ucid = ucid 
        self.uc = None 

    def __hash__(self):
        return hash((self.pid, self.ucid))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
            self.pid == other.pid and self.ucid == other.ucid 

    def __str__(self):
        return 'UR [{} -> {}]'.format(self.pid, self.ucid)

    def index(self):
        pass

        
class Lrule(Rule):
    
    def __init__(self, indexer, pid, ucid):
        super(Lrule, self).__init__(indexer, pid)
        self.ucid = ucid 
        self.uc = None 
    
    def __hash__(self):
        return hash((self.pid, self.ucid))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
            self.pid == other.pid and self.ucid == other.ucid 

    def __str__(self):
        return 'LR [{} -> {}]'.format(self.pid, self.ucid)

    def index(self):
        pass


class Brule(Rule):
    
    def __init__(self, indexer, pid, lcid, rcid):
        super(Brule, self).__init__(indexer, pid)
        self.lcid = lcid 
        self.rcid = lcid 
        self.lc = None 
        self.rc = None 

    def __hash__(self):
        return hash((self.pid, self.lcid, self.rcid))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
            self.pid == other.pid and self.lcid == other.lcid and self.rcid == other.rcid

    def __str__(self):
        return 'BR [{} -> {} {}]'.format(self.pid, self.lcid, self.rcid)

    def index(self):
        pass
