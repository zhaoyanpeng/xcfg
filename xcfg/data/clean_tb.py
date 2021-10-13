# -*- coding: utf-8 -*-
import os 
import sys
import json
import re
import logging

import numpy as np 

from tqdm import tqdm
from nltk.corpus import ptb
from nltk import Tree

from constant import STRIPPED_TAGS

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logging.basicConfig(level = logging.INFO)

LANGS = ("ENGLISH", "CHINESE", "BASQUE", "GERMAN", "FRENCH", "HEBREW", "HUNGARIAN", "KOREAN", "POLISH", "SWEDISH") 
SPLITS = ["train", "valid", "test"]

def del_tags(tree, word_tags):    
    for sub in tree.subtrees():
        for n, child in enumerate(sub):
            if isinstance(child, str):
                continue
            v = all_tags(child, word_tags)
            v = [not x for x in v]
            if all(v):
                del sub[n]

def all_tags(tree, word_tags):
    v = []
    for _, tag in tree.pos():
        label = tag.strip()
        if not label.startswith("-"):
            label = re.split("\=|\+|\-|\_", label)[0] 
        v.append(label not in word_tags)
    return v

def remove_punct(tree, word_tags):
    c = 0
    while not all(all_tags(tree, word_tags)):
        del_tags(tree, word_tags)
        c += 1
        if c > 10:
            assert False
    out = tree.pformat(margin=sys.maxsize).strip()          
    # remove (X ), i.e., zero-length spans
    while re.search('\(([a-zA-Z0-9]{1,})((\-|\=|\+|\_)[a-zA-Z0-9]*)*\s{1,}\)', out) is not None:
        out = re.sub('\(([a-zA-Z0-9]{1,})((\-|\=|\+|\_)[a-zA-Z0-9]*)*\s{1,}\)', '', out)
    out = out.replace(' )', ')')
    out = re.sub('\s{2,}', ' ', out)
    return out

def remove_punct_io(ifile, ofile, word_tags):
    trees = ptb.parsed_sents(ifile)
    with open(ofile, 'w') as fw:
        i = 0 
        empty_lines = []
        for tree in tqdm(trees):
            i += 1
            tree_str = remove_punct(tree, word_tags)
            if tree_str.strip() == "":
                empty_lines.append(i)
                continue
            fw.write(tree_str + '\n')
    return empty_lines

def main_remove_punct(iroot, oroot):
    for lang in LANGS:
        if False and lang != "SWEDISH":
            continue
        tags = STRIPPED_TAGS[lang]
        logger.info('processing {}...will remove {}'.format(lang, tags))
        lang = lang.lower()
        for split in SPLITS:
            ifile = f"{iroot}/{lang}-{split}.txt"
            ofile = f"{oroot}/{lang}-{split}.txt"
            empty_lines = remove_punct_io(ifile, ofile, tags)
            logger.info(empty_lines)

if __name__ == '__main__': 
    choices = {"punct"}
    command = f"python {sys.argv[0]} {sys.argv[1] in choices} {os.path.isdir(sys.argv[2])} {os.path.isdir(sys.argv[3])}" 
    assert (
        len(sys.argv) > 3 and
        sys.argv[1] in choices and 
        os.path.isdir(sys.argv[2]) and
        os.path.isdir(sys.argv[3])
    ), (
        f"{command}\n\nPlease follow the following example command:\n\n" + 
        "python clean_tb.py [punct] [ipath] [opath]\n"
    )
    logger.info(command)
    data_name = sys.argv[1].strip().lower()
    if data_name == "punct":
        main_remove_punct(sys.argv[2], sys.argv[3])
