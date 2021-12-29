import sys, os
import json
import logging
import argparse
import numpy as np

from batchify import Indexer
from binarize import save_labeled_tree, binarize_linear_tree
from clean_tb import remove_punct_io
from constant import STRIPPED_TAGS
from helper import clean_number

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logging.basicConfig(level = logging.INFO)
echo = logger.info

np.random.seed(3435)

parser = argparse.ArgumentParser()

parser.add_argument('--binarize', default=False, type=bool)
parser.add_argument('--ifile', default=None, type=str)
parser.add_argument('--ofile', default=None, type=str)


def extract_spans(tree):
    spans, stack, sent = list(), list(), list()
    items = tree.split()
    cur_index = 0
    for item in items:
        if item == ')':
            pos = -1
            left = None
            right = stack[pos][1]
            while stack[pos] != '(':
                left = stack[pos][0]
                pos -= 1
            assert left is not None
            assert right is not None
            stack = stack[:pos] + [(left, right)]
            spans.append((left, right))
        elif item == '(':
            stack.append(item)
        else:
            sent.append(item)
            stack.append((cur_index, cur_index))
            cur_index += 1
    return spans, sent

def main_save_text(args):
    with open(args.ifile, "r") as fr, open(args.ofile, "w") as fw:
        for line in fr:
            (caption, _) = json.loads(line)
            caption = caption.strip()
            fw.write("{}\n".format(caption))

def main_make_btree_json(args):
    with open(args.ofile, 'w') as fw:
        for tree in open(args.ifile, "r"):
            tree = tree.strip()
            span, sent = extract_spans(tree)
            json.dump((' '.join(sent), span), fw)
            fw.write('\n') 

def main_remove_punct(args):
    lang = "ENGLISH"
    tags = STRIPPED_TAGS[lang]
    logger.info('processing {}...will remove {}'.format(lang, tags))
    empty_lines = remove_punct_io(args.ifile, args.ofile, tags)
    logger.info(empty_lines)

def main_make_vocab(
    args, min_len=0, max_len=150, vocab_size=10000, lowercase=True, replace_number=True
):
    indexer = Indexer(["<pad>","<unk>","<s>","</s>"])
    num_sent = 0
    sent_len = -1
    for line in open(args.ifile, 'r'):
        sent = line.strip()
        if lowercase:
            sent = sent.lower()
        sent = [clean_number(w) if replace_number else w for w in sent.split()]
        if len(sent) > max_len or len(sent) < min_len:
            continue
        num_sent += 1
        sent_len = max(sent_len, len(sent))
        for word in sent:
            indexer.vocab[word] += 1
    indexer.prune_vocab(vocab_size, False)
    indexer.write(args.ofile)
    echo(f"Vocab size: Original = {len(indexer.vocab)}, Pruned = {len(indexer.d)}")

if __name__ == '__main__':
    args = parser.parse_args()
    
    # clean parses 
    main_remove_punct(args)

    # non-binarized gold parses 
    args.ifile = args.ofile
    args.ofile = args.ifile.rsplit(".")[0] + "_gold_caps.json"
    save_labeled_tree(args)

    # binarized gold parses 
    args.binarize = True
    args.ofile = args.ifile.rsplit(".")[0] + "_caps.bin"
    binarize_linear_tree(args)

    # binarized gold parses 
    args.ifile = args.ofile
    args.ofile = args.ifile.rsplit(".")[0] + ".json"
    main_make_btree_json(args)

    # sentences 
    args.ifile = args.ofile
    args.ofile = args.ifile.rsplit(".")[0] + ".text"
    main_save_text(args)
    
    # vocabulary
    if "train" in args.ofile:
        args.ifile = args.ofile
        args.ofile = args.ifile.rsplit("/", 1)[0] + "/data.dict"
        main_make_vocab(args)
