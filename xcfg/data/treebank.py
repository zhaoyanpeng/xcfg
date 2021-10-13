# -*- coding: utf-8 -*-
import os 
import sys
import logging
import json
import re
import shutil
import pathlib
import subprocess
import numpy as np 

from tqdm import tqdm
from nltk import Tree
from nltk.corpus import ptb

from constant import (
    PTB_TRAIN_SEC, PTB_TEST_SEC, PTB_DEV_SEC,
    DATASETS, SPMRL_SPLITS, SPMRL_LANGS, SPMRL_ROOTS
)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logging.basicConfig(level = logging.INFO)


def save_split(fids, ofile):
    logger.info(f"saving to {ofile}")
    with open(ofile, "w") as fw:
        for f in tqdm(fids):
            sentences = ptb.parsed_sents(f)
            for tree in sentences:
                tree = tree.pformat(margin=sys.maxsize).strip()
                fw.write(tree + "\n")

def main_split_chinese_tb(iroot, oroot):
    """ train: 001–270, 400–1151; devel: 301–325; test: 271-300.
    """
    train_fs, devel_fs, test_fs = list(), list(), list()
    for f in os.listdir(iroot):
        f = iroot + f
        if not os.path.isfile(f): 
            continue
        if not f.endswith("mrg"): 
            continue
        x = re.findall("chtb_(\d+)\.mrg", f)
        if not x: logger.info("unparsable {}".format(f))
        fid = int(x[0])
        if fid >= 271 and fid <= 300:
            test_fs.append(f)
        elif fid >= 301 and fid <= 325:
            devel_fs.append(f)
        elif (fid >= 1 and fid <= 270) or (fid >= 400 and fid <= 1151):
            train_fs.append(f)
    logger.info(
        "train: {} fids; devel: {} fids; test: {} fids".format(
          len(train_fs), len(devel_fs), len(test_fs)
    ))
    
    lang = "chinese"
    save_split(train_fs, oroot + "/{}-train.txt".format(lang))
    save_split(devel_fs, oroot + "/{}-valid.txt".format(lang))
    save_split(test_fs, oroot + "/{}-test.txt".format(lang))

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
    logger.info(
        "train: {} fids, test: {} fids, and dev: {} fids".format(
        len(train_fs), len(test_fs), len(devel_fs)
    ))
    return train_fs, devel_fs, test_fs

def main_split_english_tb(iroot, oroot):
    train_fs, devel_fs, test_fs = read_ptb_ids(iroot)

    lang = "english"
    save_split(train_fs, oroot + "/{}-train.txt".format(lang))
    save_split(devel_fs, oroot + "/{}-valid.txt".format(lang))
    save_split(test_fs, oroot + "/{}-test.txt".format(lang))

def remove_morph_feature(tree):
    out = tree.pformat(margin=sys.maxsize).strip()          
    while re.search('(\#\#.*?\#\#)', out) is not None:
        out = re.sub('(\#\#.*?\#\#)', '', out)
    out = out.replace(' )', ')')
    out = re.sub('\s{2,}', ' ', out)
    return out 

def remove_morph_feature_io(ifile, ofile):
    logger.info(f"saving to {ofile}")
    trees = ptb.parsed_sents(ifile)
    with open(ofile, 'w') as fw:
        for tree in tqdm(trees):
            children = list(tree.subtrees())
            if tree.label() in SPMRL_ROOTS and len(children[0]) == 1:
                tree = children[0][0]
            # keep the same ROOT
            root = Tree('ROOT', [])
            root.append(tree)
            tree = root
            tree_str = remove_morph_feature(tree)
            fw.write(tree_str + '\n')

def main_remove_morph_feature(iroot, oroot):
    maps = {'dev': "valid", 'test': "test", 'train': "train", 'train5k': "train"}
    root = iroot + '/{}_SPMRL/{}/ptb/{}/'
    for lang in SPMRL_LANGS:
        splits = []
        source = f"{iroot}/{lang}_SPMRL.tar.gz"
        if not os.path.isdir(f"{iroot}/{lang}_SPMRL") and os.path.isfile(source):
            logger.info(f"uncrompress `{source}`")
            subprocess.run(["tar", "xzf", f"{source}", "-C", f"{iroot}"])
        lang_ = lang.lower() if lang == 'SWEDISH' else lang.capitalize()
        for data in SPMRL_SPLITS:
            name = '{}.{}.gold.ptb'.format(data, lang_)
            ipath = root.format(lang, 'gold', data)
            ifile = ipath + name 
            if not pathlib.Path(ifile).is_file():
                logger.info('--skip {}'.format(ifile))
                continue
            splits.append(data)

            opath = root.format(lang, 'proc', data)
            pathlib.Path(opath).mkdir(parents=True, exist_ok=True) 
            ofile = opath + name  
            remove_morph_feature_io(ifile, ofile)

        if len(splits) == 4:
            splits.remove("train5k")
        for data in splits:
            name = '{}.{}.gold.ptb'.format(data, lang_)
            ipath = root.format(lang, 'proc', data)
            ifile = ipath + name
            ofile = oroot + f"/{lang.lower()}-{maps[data]}.txt" 
            shutil.copy2(ifile, ofile)

if __name__ == '__main__': 
    command = f"python {sys.argv[0]} {sys.argv[1] in DATASETS} {os.path.isdir(sys.argv[2])} {os.path.isdir(sys.argv[3])}" 
    assert (
        len(sys.argv) > 3 and
        sys.argv[1] in DATASETS and 
        os.path.isdir(sys.argv[2]) and
        os.path.isdir(sys.argv[3])
    ), (
        f"{command}\n\nPlease follow the following example command:\n\n" + 
        "python treebank.py [ctb, ptb, spmrl] [ipath] [opath]\n\n" +
        "For `ctb`, `ipath` is the parent directory of all `.mrg` files; \n" +
        "for `ptb`, `ipath` is the parent directory of all section directories; \n" + 
        "for `spmrl`, `ipath` is the parent directory of all treebank directories. \n"
    )
    logger.info(command)
    data_name = sys.argv[1].strip().lower()
    if data_name == "ctb":
        main_split_chinese_tb(sys.argv[2], sys.argv[3])
    elif data_name == "ptb":
        main_split_english_tb(sys.argv[2], sys.argv[3])
    elif data_name == "spmrl":
        main_remove_morph_feature(sys.argv[2], sys.argv[3])
