import os, sys, re
import pickle, json
import numpy as np
import random

from nltk import Tree
from tqdm import tqdm
from collections import Counter, defaultdict 

SEEDS = [1213, 2324, 3435, 4546, 5657]
LANGS = ("ENGLISH", "CHINESE", "BASQUE", "GERMAN", "FRENCH", "HEBREW", "HUNGARIAN", "KOREAN", "POLISH", "SWEDISH") 

seed = SEEDS[0] 
random.seed(seed)
np.random.seed(seed)

def get_stats(span1, span2):
    tp = 0
    fp = 0
    fn = 0
    for span in span1:
        if span in span2:
            tp += 1
        else:
            fp += 1
    for span in span2:
        if span not in span1:
            fn += 1
    return tp, fp, fn

class Node:
    def __init__(self, idx):
        self.idx = idx 
        self.span = [-1, -1]
        self.child = [None, None]

def build_spans(tree, l):
    if tree is None:
        return l, l 
    l, r = build_spans(tree.child[0], l)
    tree.span[0] = l
    l = r + 1
    l, r = build_spans(tree.child[1], l)
    tree.span[1] = r
    return tree.span[0], tree.span[1] 

def random_tree(n):
    nodes = [Node(0)]
    free_edges = [(0, 0), (0, 1)]
    for i in range(n - 1):
        father_idx, child_idx = random.choice(free_edges)
        assert nodes[father_idx].child[child_idx] == None

        node_idx = len(nodes)
        new_node = Node(node_idx)
        nodes[father_idx].child[child_idx] = new_node 
        nodes.append(new_node)

        free_edges.remove((father_idx, child_idx))
        free_edges.extend([(node_idx, 0), (node_idx, 1)])

    build_spans(nodes[0], 0)
    spans = [] 
    for node in nodes:
        l, r = -1, -1
        if node.child[0] is not None:
            l = node.child[0].idx 
        if node.child[1] is not None:
            r = node.child[1].idx 
        spans.append(node.span)
    spans.reverse()
    return spans

def lr_f1(per_label_f1, by_length_f1, corpus_f1, sent_f1, argmax_spans, spans, labels):
    pred = [(a[0], a[1]) for a in argmax_spans if a[0] != a[1]]
    pred_set = set(pred[:-1])
    gold = [(l, r) for l, r in spans if l != r] 
    gold_set = set(gold[:-1])

    tp, fp, fn = get_stats(pred_set, gold_set) 
    corpus_f1[0] += tp
    corpus_f1[1] += fp
    corpus_f1[2] += fn
    
    overlap = pred_set.intersection(gold_set)
    prec = float(len(overlap)) / (len(pred_set) + 1e-8)
    reca = float(len(overlap)) / (len(gold_set) + 1e-8)
    
    if len(gold_set) == 0:
        reca = 1. 
        if len(pred_set) == 0:
            prec = 1.
    f1 = 2 * prec * reca / (prec + reca + 1e-8)
    sent_f1.append(f1)

    for j, gold_span in enumerate(gold[:-1]):
        label = labels[j]
        label = re.split("=|-", label)[0]
        per_label_f1.setdefault(label, [0., 0.]) 
        per_label_f1[label][0] += 1

        lspan = gold_span[1] - gold_span[0] + 1
        by_length_f1.setdefault(lspan, [0., 0.])
        by_length_f1[lspan][0] += 1

        if gold_span in pred_set:
            per_label_f1[label][1] += 1 
            by_length_f1[lspan][1] += 1

def lr_branching(ifile, btype=1):
    per_label_f1 = defaultdict(list) 
    by_length_f1 = defaultdict(list)
    sent_f1, corpus_f1 = [], [0., 0., 0.] 
    
    counter = Counter()
    with open(ifile, "r") as fr: 
        for line in fr:
            (caption, span, label, tag) = json.loads(line)
            caption = caption.strip().split()
            if len(caption) < 2:
                continue
            nword = len(caption)
            if btype == 0:
                token = "Left Branching: "
                argmax_span = [(0, r) for r in range(1, nword)] 
            elif btype == 1:
                token = "Right Branching: "
                argmax_span = [(l, nword - 1) for l in range(0, nword -1)] 
                argmax_span = argmax_span[1:] + argmax_span[:1]
            elif btype == 3:
                token = "Random Trees: "
                argmax_span = random_tree(nword - 1) 
                assert len(argmax_span) == nword - 1 
            lr_f1(per_label_f1, by_length_f1, corpus_f1, sent_f1, argmax_span, span, label)

    tp, fp, fn = corpus_f1  
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    corpus_f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.
    sent_f1 = np.mean(np.array(sent_f1))

    token += ifile 
    print('\n{}\n\nCorpus F1: {:.4f}, Sentence F1: {:.4f}'.format(token, corpus_f1, sent_f1))

    f1_ids=["CF1", "SF1", "NP", "VP", "PP", "SBAR", "ADJP", "ADVP"]
    f1s = {"CF1": corpus_f1, "SF1": sent_f1}

    print("PER-LABEL-F1 (label, acc)\n")
    for k, v in per_label_f1.items():
        print("{}\t{:.4f} = {}/{}".format(k, v[1] / v[0], v[1], v[0]))
        if True or k in f1_ids:
            f1s[k] = v[1] / v[0]
    # special case for SPMRL
    exist = len([x for x in f1_ids if x in f1s]) == len(f1_ids) 
    if not exist:
        xx = sorted(list(per_label_f1.items()), key=lambda x: -x[1][0])
        f1_ids = ["CF1", "SF1"] + [x[0] for x in xx[:8]]
    f1s = ['{:.2f}'.format(float(f1s[x]) * 100) for x in f1_ids] 
    print("\t".join(f1_ids))
    print(seed, " ".join(f1s))

    acc = []
    print("\nPER-LENGTH-F1 (length, acc)\n")
    xx = sorted(list(by_length_f1.items()), key=lambda x: x[0])
    for k, v in xx:
        print("{}\t{:.4f} = {}/{}".format(k, v[1] / v[0], v[1], v[0]))
        if v[0] >= 5:
            acc.append((str(k), '{:.2f}'.format(v[1] / v[0])))
    k = [x for x, _ in acc]
    v = [x for _, x in acc]
    print(" ".join(k))
    print(" ".join(v))

def main_lr_branching(iroot):
    for lang in LANGS:
        if lang.lower() != "swedish":
            continue 
        print("{} BEGINS".format(lang))
        for btype, name in enumerate(["LEFT-BRANCHING", "RIGHT-BRANCHING"]):
            ifile = iroot + f"{lang.lower()}-train.json"
            lr_branching(ifile, btype) 
            print("\n")
        print("{} ENDS\n".format(lang))

def main_label_by_length(ifile, max_span_len=20, labels=["NP", "VP", "PP", "SBAR", "ADJP", "ADVP"]):
    """ Label distribution over constituent lengths.
        Return: (n x m) matrix: n labels and m lengths.
    """
    per_label_len = defaultdict(dict) 
    lspans = list(range(2, max_span_len))
    with open(ifile, "r") as fr:
        for line in fr:
            (caption, spans, labels, tag) = json.loads(line)

            spans = spans[:-1]
            labels = labels[:-1]
            for gold_span, label in zip(spans, labels):

                lspan = gold_span[1] - gold_span[0] + 1
                per_label_len.setdefault(lspan, defaultdict(float)) 

                label = re.split("=|-", label)[0]
                per_label_len[lspan][label] += 1

    nspan = 0 
    for k, v in per_label_len.items():
        x = sum(v.values())
        nspan += x

    for k, v in per_label_len.items():
        for kk in v.keys():
            v[kk] = v[kk] / nspan

    data = []
    for lspan in lspans:
        d = [per_label_len[lspan][label] for label in labels]
        data.append(d)

    data = np.array(data)
    print(data)
    return data

if __name__ == '__main__':
    ifile = sys.argv[1]
    main_lr_branching(ifile)
    #main_label_by_length(ifile)

