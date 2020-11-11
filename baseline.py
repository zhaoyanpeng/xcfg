import os, sys, re
import pickle, json
import numpy as np
import random

from tqdm import tqdm
from collections import Counter, defaultdict 
from nltk import Tree

from self1 import extract_spans, self1 

seed = 1213 
random.seed(seed)
np.random.seed(seed)

seeds = [1213, 2324, 3435, 4546, 5657]

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
        #print(node.idx, (l, r), node.span)
        spans.append(node.span)
    spans.reverse()
    return spans

def main_random_tree():
    x = random_tree(4)
    print(x)

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

def main_lr_branching(lang="english", btype=1):
    name = "val"
    name = "train"
    name = "test"
    root = "/home/s1847450/data/vsyntax/mscoco/" 


    #name = "{}-test".format(lang)
    #root = "/home/s1847450/data/spmrl2014/spmrl.clean/" 

    btype = 0 

    per_label_f1 = defaultdict(list) 
    by_length_f1 = defaultdict(list)
    sent_f1, corpus_f1 = [], [0., 0., 0.] 
    
    fname= root + "{}_gold_caps.json".format(name)
    
    counter = Counter()
    with open(fname, "r") as fr: 
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

    token += fname
    print('\n{}\n\nCorpus F1: {:.4f}, Sentence F1: {:.4f}'.format(token, corpus_f1, sent_f1))

    f1_ids=["CF1", "SF1", "NP", "VP", "PP", "SBAR", "ADJP", "ADVP"]
    f1s = {"CF1": corpus_f1, "SF1": sent_f1}

    print("PER-LABEL-F1 (label, acc)\n")
    for k, v in per_label_f1.items():
        print("{}\t{:.4f} = {}/{}".format(k, v[1] / v[0], v[1], v[0]))
        if True or k in f1_ids:
            f1s[k] = v[1] / v[0]




    exist = len([x for x in f1_ids if x in f1s]) == len(f1_ids) 
    if not exist:
      xx = sorted(list(per_label_f1.items()), key=lambda x: -x[1][0])
      f1_ids = ["CF1", "SF1"] + [x[0] for x in xx[:8]]
    f1s = ['{:.2f}'.format(float(f1s[x]) * 100) for x in f1_ids] 




    #f1s = ['{:.2f}'.format(float(f1s[x]) * 100) for x in f1_ids] 
    print("\t".join(f1_ids))
    #print(" & ".join(f1s))
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


def main_lr_branching_spmrl():
    LANGS = ("ENGLISH", "CHINESE", "BASQUE", "GERMAN", "FRENCH", "HEBREW", "HUNGARIAN", "KOREAN", "POLISH", "SWEDISH") 
    for lang in LANGS:
        print("{} BEGINS".format(lang))
        for btype, name in enumerate(["LEFT-BRANCHING", "RIGHT-BRANCHING"]):
            main_lr_branching(lang.lower(), btype) 
            print("\n")
        print("{} ENDS\n".format(lang))


def lr_similarity(fname, left=True):
    sent_f1, corpus_f1 = [], [0., 0., 0.] 
    with open(fname, "r") as f:
        for line in f:
            line = json.loads(line)

            nword = len(line[0])
            if left:
                gold = [(0, r) for r in range(1, nword)] 
            else:
                gold = [(l, nword - 1) for l in range(0, nword -1)] 

            pred = line[-1]
            assert len(gold) == len(pred)

            pred = [(l, r) for l, r in pred if l != r] 
            gold = [(l, r) for l, r in gold if l != r] 

            gold_set = set(gold[:-1])
            pred_set = set(pred[:-1])

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

    tp, fp, fn = corpus_f1  
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    corpus_f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.
    sent_f1 = np.mean(np.array(sent_f1))

    return corpus_f1, sent_f1

def main_lr_similarity():
    root = "/home/s1847450/data/vsyntax/"

    name = "v.msc.2nd.40.srnn.b5.{}/{}.pred"
    epochs = (10, 13, 6, 13, 14) 

    name = "msc.2nd.40.{}/{}.pred"
    epochs = (14, 9, 10, 6, 14) 

    name = "vgnsl.{}/{}.pred"
    epochs = (4, 4, 4, 4, 4) 

    name = "vgnsl.hi.{}/{}.pred"
    epochs = (15, 15, 15, 15, 15) 
    
    #left = True
    left = False 

    f1 = []
    for i in range(0, 4):
        fname = root + name.format(seeds[i], epochs[i])
        v = lr_similarity(fname, left=left)
        f1.append(v)
        #break
   
    sign = "LEFT" if left else "RIGHT"
    print("{} -> {}".format(sign, name))

    cf1s = [x for x, _ in f1]
    sf1s = [x for _, x in f1]

    cf1 = np.mean(cf1s)
    sf1 = np.mean(sf1s)

    cstd = np.std(cf1s)
    sstd = np.std(sf1s)

    for x, y in zip(cf1s, sf1s):
        print("{} {}".format(x, y))

    print()
    print(cf1, cstd)
    print(sf1, sstd)


def main_self1():
    root = "/home/s1847450/data/vsyntax/"

    name = "vgnsl.hi.{}/{}.pred"
    epochs = (15, 15, 15, 15, 15) 
    
    name = "vgnsl.{}/{}.pred"
    epochs = (4, 4, 4, 4, 4) 

    name = "msc.2nd.40.{}/{}.pred"
    epochs = (14, 9, 10, 6, 14) 

    name = "v.msc.2nd.40.srnn.b5.{}/{}.pred"
    epochs = (10, 13, 6, 13, 14) 

    f1_per_epoch = []

    f1 = []
    for i in range(0, 4):
        for j in range(i + 1, 4):
            file1 = root + name.format(seeds[i], epochs[i])
            file2 = root + name.format(seeds[j], epochs[j]) 
            v = self1(file1, file2) 
            f1.append(v)
    cf1s = [x for x, _ in f1]
    sf1s = [x for _, x in f1]

    cf1 = np.mean(cf1s)
    sf1 = np.mean(sf1s)

    cstd = np.std(cf1s)
    sstd = np.std(sf1s)

    for x, y in zip(cf1s, sf1s):
        print("{} {}".format(x, y))

    print()
    print(cf1, cstd)
    print(sf1, sstd)


def main_label_by_length():
    name = "test_gold_caps.json"
    root = "/home/s1847450/data/vsyntax/mscoco/" 

    name = "english-test_gold_caps.json"
    root = "/home/s1847450/data/spmrl2014/spmrl.clean/" 
    
    f = root + name
    
    per_label_len = defaultdict(dict) 
    with open(f, "r") as fr:
        for line in fr:
            (caption, spans, labels, tag) = json.loads(line)

            spans = spans[:-1]
            labels = labels[:-1]
            for gold_span, label in zip(spans, labels):

                lspan = gold_span[1] - gold_span[0] + 1
                per_label_len.setdefault(lspan, defaultdict(float)) 

                label = re.split("=|-", label)[0]
                per_label_len[lspan][label] += 1

    f1_ids = ["NP", "VP", "PP", "SBAR", "ADJP", "ADVP"]
    lspans = list(range(2, 20))

    f1_ids = ["NP", "VP", "PP", "SBAR", "ADJP", "ADVP"] #, "S", "QP"]
    lspans = list(range(2, 20))

    nspan = 0 
    for k, v in per_label_len.items():
        x = sum(v.values())
        nspan += x

    for k, v in per_label_len.items():
        for kk in v.keys():
            v[kk] = v[kk] / nspan

    data = []
    for lspan in lspans:
        d = [per_label_len[lspan][label] for label in f1_ids]
        data.append(d)

    data = np.array(data)
    cols = np.sum(data, 0)
    rows = np.sum(data, 1)

    print(np.array2string(data, separator=', '))
    print(np.array2string(cols, separator=', '))
    print(np.array2string(rows, separator=', '))


def main_make_freda_test():
    name = "freda_caps.bin"
    root = "/home/s1847450/data/vsyntax/mscoco/" 

    f = root + name
    fout = root + "freda_gold_caps.json" 

    text = root + "freda_caps.text"
    
    with open(f, 'r') as fr, open(fout, 'w') as fw, open(text, 'w') as ft:
        for line in fr:
            line = line.strip()
            spans = extract_spans(line)

            nspan = len(spans)
            label = ["S"] * nspan

            words = re.sub('\)|\(', '', line)
            #print("\n{}".format(words))
            words = words.strip().split()
            nword = len(words)
            tags = ["T"] * nword

            tree = Tree.fromstring(line)
            #print(tree)
            #print(spans)
            #print(words, nword, nspan)

            data = (" ".join(words), spans, label, tags)

            print(data)
            json.dump(data, fw)
            fw.write("\n")

            ft.write("{}\n".format(data[0]))

def build_parse(spans, caption):
    tree = [[i, word, 0, 0] for i, word in enumerate(caption)]
    for l, r in spans:
        if l != r:
            tree[l][2] += 1
            tree[r][3] += 1
    new_tree = ["".join(["["] * nl) + word + "".join([" ]"] * nr) for i, word, nl, nr in tree] 
    return " ".join(new_tree)

def sample_tree(fname, left=True):
    sent_f1, corpus_f1 = [], [0., 0., 0.] 
    with open(fname, "r") as f:
        for line in f:
            line = json.loads(line)

            words = line[0]
            nword = len(words)

            pred = line[-1]
            gold = line[1]

            pred = [(l, r) for l, r in pred if l != r] 
            gold = [(l, r) for l, r in gold if l != r] 

            gold_set = set(gold[:-1])
            pred_set = set(pred[:-1])

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

            print("{:.2f}".format(f1 * 100), " ".join(words))
            print(gold)
            print(pred)
            print(gold_set - pred_set)
            gold = build_parse(gold, words)
            pred = build_parse(pred, words)
            print(gold)
            print(pred)
            print()

    tp, fp, fn = corpus_f1  
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    corpus_f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.
    sent_f1 = np.mean(np.array(sent_f1))
    
    print(corpus_f1 * 100, sent_f1 * 100)
    return corpus_f1, sent_f1

def main_sample_tree():
    
    root = "/home/s1847450/data/vsyntax/"

    name = "v.msc.2nd.40.srnn.b5.{}/{}.freda.pred"
    epochs = (10, 13, 6, 13, 14) 
    
    i = 0
    fname = root + name.format(seeds[i], epochs[i])
    sample_tree(fname)

def depth():
    pass

def label_depth():
    line = "(NT-10 (T-29 they) (NT-1 (T-37 came) (NT-24 (T-30 by) (NT-2 (T-53 their) (NT-7 (T-43 strangeness) (T-38 honestly))))))"
    tree = Tree.fromstring(line)
    print(tree)
    pass

def main_label_depth():
    label_depth()
    pass

def recompute_f1s(fname, slevel=False):
    sent_f1, corpus_f1 = [], [0., 0., 0.] 
    with open(fname, "r") as f:
        for line in f:
            line = json.loads(line)

            words = line[0]
            nword = len(words)

            pred = line[-1]
            gold = line[1]

            pred = [(l, r) for l, r in pred if l != r] 
            gold = [(l, r) for l, r in gold if l != r] 
            
            if slevel:
                gold_set = set(gold[:])
                pred_set = set(pred[:])
            else:
                gold_set = set(gold[:-1])
                pred_set = set(pred[:-1])

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
    tp, fp, fn = corpus_f1  
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    corpus_f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.
    sent_f1 = np.mean(np.array(sent_f1))
    
    print(corpus_f1 * 100, sent_f1 * 100)
    return corpus_f1, sent_f1

def main_recompute_f1s():
    root = "/home/s1847450/data/vsyntax/"
    name = "vgnsl.{}/{}.pred"
    epochs = (4, 4, 4, 4, 4) 

    i = 3 
    slevel = True
    fname = root + name.format(seeds[i], epochs[i])
    recompute_f1s(fname, slevel)

if __name__ == '__main__':
    main_lr_branching()
    #main_lr_branching_spmrl()
    #main_label_depth()
    #main_random_tree()
    #main_self1()
    #main_lr_similarity()
    #main_label_by_length()
    #main_make_freda_test()
    #main_sample_tree()
    #main_recompute_f1s()
    pass

