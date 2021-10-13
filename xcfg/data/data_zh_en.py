import os, sys, time
import multiprocessing
import subprocess
import requests
import datetime
import time
import json
import copy
import nltk
import math

import spacy
import jieba

import numpy as np
import regex as xre
import unicodedata

from tqdm import tqdm
from csv import reader, writer, DictReader
from collections import defaultdict
from bs4 import BeautifulSoup as bsoup
from nltk.tokenize import word_tokenize
from collections import Counter, defaultdict

from helper import make_vocab

home = os.environ['MYHOME'] #FIXME change me! 
root = f"{home}/data/parallel/chinese_translation"
spmrl_root = f"{home}/data/spmrl2014/data.clean"

# jieba and nltk tokenization are what you need
# do use jieba for chinese, use either nltk or spacy for english
USE_NLTK = False #True # 
USE_JIEBA = True #False

SIG = "."
if USE_NLTK:
    SIG += "nltk." 
if USE_JIEBA:
    SIG += "jieba." 

#######
# read into aligned files 
#######
def build_zh_en_map(root, ofile=None):
    cnt = 0
    pos = len(root)
    npz_dict = defaultdict(str)
    for root, dir, files in os.walk(root):
        if len(dir) > 0:
            continue
        sub_dir =  root[pos + 1:]
        print(sub_dir, root, dir, len(files))

        for ifile, fname in tqdm(enumerate(files)):
            npz_dict[fname] = f"{sub_dir}/{fname}"

        """    if ifile > 10:
                break
        print(npz_dict)
        break
        cnt += 1
        if cnt > 400:
            break
        pass """
         
    if ofile is not None:
        print(f"writting {len(npz_dict)} indexes into {ofile}")
        #json_str = json.dumps(npz_dict, indent=2)
        #
        #with open(ofile, "r") as fr:
        #    xx = json.load(fr)
        #
        with open(ofile, "w") as fw:
            json.dump(npz_dict, fw, indent=2)
    return npz_dict

def normalize_sentence(ifile, tokenizer_zh=None, tokenizer_en=None, cut=True, verbose=False):
    """ normalize text and segment it.
    """
    soup = bsoup(open(ifile, "rb"), "html.parser")
    sents = soup.find_all('seg')
    sents = [(int(sent['id']), sent.text.strip()) for sent in sents]
    sents = sorted(sents, key=lambda x: x[0])
    results = []
    for sent in sents:
        doc = sent[1]
#       doc = xre.sub('\p{N}{1,}([,.．]?\p{N}*)*', 'N', doc)
        # https://stackoverflow.com/a/2422245
        doc = unicodedata.normalize('NFKC', doc)
        if cut:
            if USE_JIEBA:
                doc = jieba.cut(doc)
                doc = " ".join(doc)
            else:
                doc = tokenizer_zh(doc)
                doc = " ".join([t.text for t in doc])
        else:
            if USE_NLTK:
                doc = word_tokenize(doc)
                doc = " ".join(doc)
            else:
                doc = tokenizer_en(doc)
                doc = " ".join([t.text for t in doc])
        results.append(doc)
        if verbose:
            print(doc)
    return results

def build_zh_en_pair():
    zh_root = f"{root}/data/source"
    en_root = f"{root}/data/translation"
    file_map = build_zh_en_map(en_root, None)
    print(f"total {len(file_map)} translation files.")
    ofile = f"{root}/data/corpus{SIG}seg"

    tokenizer_zh, tokenizer_en = zh_en_tokenizer()

    with open(ofile, "w", encoding="utf8") as fw:
        for k, v in tqdm(file_map.items()):
            zh_file = f"{zh_root}/{k}"
            en_file = f"{en_root}/{v}"
            #print(zh_file, en_file)
            assert os.path.isfile(zh_file) and os.path.isfile(en_file), \
                f"either {zh_file} or {en_file} does not exist."
            
            verbose = False
            sents_zh = normalize_sentence(
                zh_file, verbose=verbose,
                tokenizer_zh=tokenizer_zh, tokenizer_en=tokenizer_en
            )
            sents_en = normalize_sentence(
                en_file, cut=False, verbose=verbose, 
                tokenizer_zh=tokenizer_zh, tokenizer_en=tokenizer_en
            )

            assert len(sents_zh) == len(sents_en), f"mismatched sents |zh|{len(sents_zh)} != |en|{len(sents_en)}"
            for i, (zh, en) in enumerate(zip(sents_zh, sents_en)):
                line = f"{zh}\t{en}\t{v}\t{i + 1}"
                fw.write(f"{line}\n")
            #break

def zh_en_tokenizer(tokenized):
    tokenizer_zh = spacy.load('zh_core_web_md', disable=['tok2vec', 'tagger', 'parser', 'ner', 'attribute_ruler'])
    tokenizer_en = spacy.load('en_core_web_md', disable=['tok2vec', 'tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
    if tokenized: # only use tokenizers for punctuation detection
        from spacy.tokens import Doc
        class PretokenizedTokenizer:
            def __init__(self, vocab):
                self.vocab = vocab
            def __call__(self, words):
                return Doc(self.vocab, words=words)
        tokenizer_zh.tokenizer = PretokenizedTokenizer(tokenizer_zh.vocab)
        tokenizer_en.tokenizer = PretokenizedTokenizer(tokenizer_en.vocab)
    return tokenizer_zh, tokenizer_en

def is_punct(t):
    return t.is_punct or t.is_left_punct or t.is_right_punct

def rm_punct(line, tokenizer_zh, tokenizer_en, tokenized=False, MIN_LEN=2, MAX_LEN=sys.maxsize):
    sent_zh, sent_en, src, isent = line.strip().split('\t')

    if tokenized:
        sent_zh = sent_zh.split()
        sent_en = sent_en.split()

    doc_zh = tokenizer_zh(sent_zh)
    #doc_zh = tokenizer_zh.tokens_from_list(sent_zh.split())
    sent_zh = "".join([x.text_with_ws for x in doc_zh if not is_punct(x)]).strip()

    sent_zh_len = len(sent_zh.split())
    if sent_zh_len < MIN_LEN or sent_zh_len > MAX_LEN:
        return False, None

    doc_en = tokenizer_en(sent_en)
    #doc_en = tokenizer_en.tokens_from_list(sent_en.split())
    sent_en = "".join([x.text_with_ws for x in doc_en if not is_punct(x)]).strip()

    #FIXME remove space between two consecutive numbers?
    #sent_en = re.sub('(?<=\s\d+)\s+(?=\d+\s)', '', sent_en)

    sent_en_len = len(sent_en.split())
    if sent_en_len < MIN_LEN or sent_en_len > MAX_LEN:
        return False, None
    return True, f"{sent_zh}\t{sent_en}\t{src}\t{isent}"

def clean_punct():
    tokenized = True
    tokenizer_zh, tokenizer_en = zh_en_tokenizer(tokenized=tokenized)
    if not tokenized and USE_NLTK:
        from spacy.symbols import ORTH
        # Add special case rule
        special_case = [{ORTH: "``"}]
        tokenizer_en.tokenizer.add_special_case("``", special_case)

    ifile = f"{root}/data/corpus{SIG}seg"
    ofile = f"{root}/data/corpus{SIG}clean.seg"

    with open(ifile, "r", encoding="utf8") as fr, \
        open(ofile, "w", encoding="utf8") as fw:
        for iline, line in tqdm(enumerate(fr)):
            flag, new_line = rm_punct(line, tokenizer_zh, tokenizer_en, tokenized=tokenized)
            if flag:
                fw.write(f"{new_line}\n")
            else:
                #raise ValueError(f"{iline} {line}")
                print(f"{iline} {line}")
            if iline > 6:
                pass #break

def zh_en_vocab():
    vocab_en_name = "english.dict"
    vocab_zh_name = "chinese.dict"
    vocab_en_file = f"{spmrl_root}/{vocab_en_name}"
    vocab_zh_file = f"{spmrl_root}/{vocab_zh_name}"

    vocab_en = make_vocab(vocab_en_file)
    vocab_zh = make_vocab(vocab_zh_file)

    en_keys = vocab_en.word2idx.keys()
    zh_keys = vocab_zh.word2idx.keys()
    return en_keys, zh_keys

def clean_number(w):
    new_w = xre.sub('\p{N}{1,}([,.．]?\p{N}*)*', 'N', w)
    return new_w

def rate(sent, keys):
    cnt = 0
    words = sent.split()
    for w in words:
        w = clean_number(w)
        if w not in keys:
            cnt += 1
    return cnt, len(words)

def unknown_rate(ifile, save_file=None, save=False, bsize=1000):
    en_keys, zh_keys = zh_en_vocab()
    vocab = defaultdict(list)
    cnt = 0
    with open(ifile, "r") as fr:
        for line in fr:
            cnt += 1
            data = line.strip().split("\t")
            if len(data) == 1:
                print(f"{cnt}\n{line}\n{data}")
                break
            sent_zh, sent_en = data[:2]
            rate_zh, len_zh = rate(sent_zh.lower(), zh_keys)
            rate_en, len_en = rate(sent_en.lower(), en_keys)
            key = f"{rate_zh}_{len_zh}_{rate_en}_{len_en}"
            vocab[key].append(cnt)
            if cnt % bsize == 0:
                print("--select {} x {}".format(bsize, cnt // bsize))
        if cnt % bsize != 0:
            print("--select {} x {}".format(cnt % bsize, math.ceil(cnt / bsize)))
    lengths = sorted(vocab.items(), key=lambda x: len(x[1]))
    lengths = sorted(lengths, key=lambda x: x[0])
    if save and save_file is not None:
        with open(save_file, 'w') as fw:
            for i, (l, c) in enumerate(lengths):
                fw.write("{} {}\n".format(l, c))
    l = [x for x, y in lengths]
    c = [y for x, y in lengths]
    return l, c

def main_unknown():
    del_punct = True
    key = "clean." if del_punct else "" #

    ifile = f"{root}/data/corpus{SIG}{key}seg"
    ofile = f"{root}/data/corpus{SIG}{key}unk"

    print("ifile: {}".format(ifile))
    print("ofile: {}".format(ofile))

    t = time.time()
    unknown_rate(ifile, ofile, True)
    t = math.ceil(time.time() - t)

    h = t // 3600
    m = (t % 3600) // 60
    s = (t % 3600)  % 60
    print(f"--{h:d}h:{m:d}m:{s:d}s")

def token_list(ifile, save_file=None, save=False, bsize=1000):
    vocab_zh = Counter() 
    vocab_en = Counter() 
    cnt = 0
    with open(ifile, "r") as fr:
        for line in fr:
            cnt += 1
            data = line.strip().split("\t")
            sent_zh, sent_en = data[:2]
            
            for t in sent_zh.lower().split():
                t = clean_number(t)
                vocab_zh[t] += 1
            for t in sent_en.lower().split():
                t = clean_number(t)
                vocab_en[t] += 1

            if cnt % bsize == 0:
                print("--select {} x {}".format(bsize, cnt // bsize))
        if cnt % bsize != 0:
            print("--select {} x {}".format(cnt % bsize, math.ceil(cnt / bsize)))
    if save and save_file is not None:
        with open(save_file + '.zh', 'w') as fw:
            for i, (l, c) in enumerate(vocab_zh.most_common()):
                fw.write("{} {}\n".format(l, c))
        with open(save_file + '.en', 'w') as fw:
            for i, (l, c) in enumerate(vocab_en.most_common()):
                fw.write("{} {}\n".format(l, c))
    return vocab_zh, vocab_en 

def main_token():
    del_punct = True
    key = "clean." if del_punct else "" #

    ifile = f"{root}/data/corpus{SIG}{key}seg"
    ofile = f"{root}/data/corpus{SIG}{key}tok"

    print("ifile: {}".format(ifile))
    print("ofile: {}".format(ofile))

    t = time.time()

    token_list(ifile, ofile, True)

    t = math.ceil(time.time() - t)

    h = t // 3600
    m = (t % 3600) // 60
    s = (t % 3600)  % 60
    print(f"--{h:d}h:{m:d}m:{s:d}s")

def check_vocab():
    en_keys, zh_keys = zh_en_vocab()
    def read_word_list(ifile):
        vocab = Counter()
        with open(ifile, 'r') as fr:
            for line in fr:
                word, cnt = line.strip().split()
                vocab[word] = int(cnt)
        return vocab

    del_punct = True
    key = "clean." if del_punct else "" #

    ifile = f"{root}/data/corpus{SIG}{key}tok"
    en_counter = read_word_list(f"{ifile}.en")
    zh_counter = read_word_list(f"{ifile}.zh")
    
    # most common
    def overlap_most_common(counter, keyset, k, msg=""):
        sub_counter = counter.most_common()[:k]
        sub_keys = set([t for t, _ in sub_counter])
        additional = sub_keys - keyset
        print(f"{msg}{len(additional)} new keys out of the first {k}.")
        pass

    overlap_most_common(en_counter, en_keys, 5000, "EN: ")
    overlap_most_common(zh_counter, zh_keys, 5000, "ZH: ")

    # minimum frequency 
    def overlap_min_freq(counter, keyset, k, msg=""):
        sub_keys = set([t for t, v in counter.items() if v >= k])
        additional = sub_keys - keyset
        print(f"{msg}{len(additional)} new keys out of {len(sub_keys)} keys with the minimum frequency {k}.")
        pass

    overlap_min_freq(en_counter, en_keys, 5, "EN: ")
    overlap_min_freq(zh_counter, zh_keys, 5, "ZH: ")

if __name__ == '__main__':
    #build_zh_en_pair()
    #clean_punct()
    #main_unknown()
    #main_token()
    #check_vocab()
    pass
