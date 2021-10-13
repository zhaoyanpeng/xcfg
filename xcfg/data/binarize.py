import sys, os
import argparse, json

from batchify import (
    get_tags_tokens_lowercase, get_nonbinary_spans, 
    get_nonbinary_spans_label, get_actions
)

parser = argparse.ArgumentParser()

parser.add_argument('--binarize', default=False, type=bool)
parser.add_argument('--ifile', default=None, type=str)
parser.add_argument('--ofile', default=None, type=str)

def space_linear_tree(actions, sent=None, SHIFT=0, REDUCE=1):
    """ Use `( ` instead of `(`. Similarly for ` )`.  
    """
    stack = []
    pointer = 0
    if sent is None:
        sent = list(map(str, range((len(actions)+1) // 2)))
    for action in actions:
        if action == SHIFT:
            word = sent[pointer]
            stack.append(word)
            pointer += 1
        elif action == REDUCE:
            right = stack.pop()
            left = stack.pop()
            stack.append('( ' + left + ' ' + right + ' )')
    assert(len(stack) == 1)
    return stack[-1]

def binarize_linear_tree(args):
  with open(args.ifile, "r") as fr, \
      open(args.ofile, "w") as fw:
      for tree in fr:
          tree = tree.strip()
          action = get_actions(tree)
          tags, sent, sent_lower = get_tags_tokens_lowercase(tree)
          gold_span, binary_actions, nonbinary_actions = get_nonbinary_spans(action)
          pred_tree = space_linear_tree(binary_actions, sent)
          fw.write(pred_tree.strip() + "\n")

def save_labeled_tree(args):
    """ A tree is stored as (sent: str, spans: list, labels, list, tags: list).
    """
    with open(args.ifile, "r") as fr, \
        open(args.ofile, "w") as fw:
        for tree in fr:
            tree = tree.strip()
            action = get_actions(tree)
            tags, sent, sent_lower = get_tags_tokens_lowercase(tree)
            gold_span, binary_actions = get_nonbinary_spans_label(action)

            tags = tags
            sent = ' '.join(sent)
            spans = [(a, b) for a, b, _ in gold_span]
            labels = [l for _, _, l in gold_span] 

            data = (sent, spans, labels, tags)
            json.dump(data, fw)
            fw.write("\n")

if __name__ == '__main__':
    args = parser.parse_args()
    if args.binarize:
        binarize_linear_tree(args)
    else:
        save_labeled_tree(args)
