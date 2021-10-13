import json
import torch
import numpy as np
import regex as xre
import torch.utils.data as data

lang_id_map = {
    'english': 0, 'chinese': 1, 'basque': 2, 'german': 3, 'french': 4, 'hebrew': 5, 
    'hungarian': 6, 'korean': 7, 'polish': 8, 'swedish': 9, '<unk>': 10,
} 

english_tag2idx = """{'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3, 'NNS': 4, 'NN': 5, 'RB': 6, 'NNP': 7, 'VB': 8,
'JJ': 9, 'UH': 10, 'SYM': 11, 'CD': 12, 'VBN': 13, 'FW': 14, 'NNPS': 15, 'WRB': 16, 'WP': 17, 'VBZ': 18, 'DT': 19,
'VBG': 20, 'IN': 21, 'PRP': 22, 'VBD': 23, 'RP': 24, 'CC': 25, 'LS': 26, 'VBP': 27, 'RBR': 28, 'PRP$': 29, 'POS': 30,
'JJS': 31, 'MD': 32, 'TO': 33, 'EX': 34, 'JJR': 35, 'WDT': 36, 'PDT': 37, 'RBS': 38, 'WP$': 39}"""

STR2DICT = lambda x: json.loads(x.replace("\'", "\""))
SWAP_k_V = lambda x: {v: k for k, v in x.items()}

BERT_TOKEN_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    '«': '"',
    '»': '"',
    '‘': "'",
    '’': "'",
    '“': '"',
    '”': '"',
    '„': '"',
    '‹': "'",
    '›': "'",
    "\u2013": "--", # en dash
    "\u2014": "--", # em dash
}

def make_vocab(index_file, extra_keys=[], name=""):
    vocab = Indexer(index_file, extra_keys, name)
    return vocab

def clean_number(w):    
    #new_w = re.sub('[0-9]{1,}([,.]?[0-9]*)*', 'N', w) # cannot handle unicode
    new_w = xre.sub('\p{N}{1,}([,.．]?\p{N}*)*', 'N', w)
    return new_w

def tokenize(text, tokenizer, min_length, max_length, lower_case=True):
    words = text.strip().split()
    if len(words) < min_length or len(words) > max_length:
        return [], [], []
    w2v_words = list()
    mul_words = [tokenizer.cls_token] 
    mul_index = list() 
    for k, word in enumerate(words):
        w2v_word = word.lower() if lower_case else word
        # word ids -> word vector
        w2v_word = clean_number(w2v_word)  
        w2v_words.append(w2v_word)
        # sub-word ids -> bert vector
        word = BERT_TOKEN_MAPPING.get(word, word)
        if word == "n't" and len(mul_words) > 0:
            mul_words[-1] = mul_words[-1] + "n" 
            word = "'t"
        sub_words = tokenizer.tokenize(word) 
        if len(sub_words) == 0:
            sub_words = [tokenizer.unk_token]
        mul_words.extend(sub_words)
        mul_index.append(len(mul_words) - 1)
        #mul_index.extend([k] * len(sub_words))
    mul_words.append(tokenizer.sep_token)
    return w2v_words, mul_words, mul_index 

class SortedBlockSampler(data.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        all_sample = len(self.data_source)
        batch_size = getattr(data_source, "_batch_size", data_source.batch_size)
        nblock = all_sample // batch_size 
        residue = all_sample % batch_size
        nsample = all_sample - residue
        # https://numpy.org/doc/stable/reference/generated/numpy.array_split.html
        # it returns l % n sub-arrays of size l//n + 1 and the rest of size l//n.
        self.groups = np.array_split(range(nsample), nblock)
        self.strip_last = False
        if residue > 0:
            self.strip_last = True
            block = np.array(range(nsample, all_sample))
            self.groups.append(block)

    def __iter__(self):
        self.data_source._shuffle()
        end = -1 if self.strip_last else len(self.groups)
        groups = self.groups[:end]
        #random.shuffle(groups) 
        indice = torch.randperm(len(groups)).tolist() 
        groups = [groups[k] for k in indice]
        if self.strip_last:
            groups.append(self.groups[-1])
        indice = list()
        for i, group in enumerate(groups):
            indice.extend(group)
            #print(i, group)
        assert len(indice) == len(self.data_source)
        return iter(indice)

    def __len__(self):
        return len(self.data_source)

class SortedRandomSampler(data.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        self.data_source._shuffle()
        return iter(torch.randperm(len(self.data_source)).tolist())

    def __len__(self):
        return len(self.data_source)

class SortedSequentialSampler(data.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        self.data_source._shuffle()
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)

class Indexer:
    """
    Build vocabulary from a pre-defined word-index map.

    Args:
        index_file: a file containing <word, index> per line.
            Indices must be a contiguous `int` sequence. The
            first four words must be `PAD`,`UNK`,`BOS`,`EOS`.
    """
    def __init__(self, index_file=None, extra_keys=[], name=""):
        self.PAD, self.UNK, self.BOS, self.EOS = ["<pad>", "<unk>", "<s>", "</s>"]
        self.PAD_IDX, self.UNK_IDX, self.BOS_IDX, self.EOS_IDX = [0, 1, 2, 3]
        self.word2idx = {
            self.PAD: self.PAD_IDX,
            self.UNK: self.UNK_IDX,
            self.BOS: self.BOS_IDX,
            self.EOS: self.EOS_IDX,
        }
        self.idx2word = {}

        self._done = False
        if index_file is not None:
            self.from_file(index_file)
        elif len(extra_keys) > 0:
            self.from_list(extra_keys)
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def word_list(self):
        return [self.idx2word[k] for k in self.idx2word.keys() if k > 3]
	
    def from_file(self, index_file):
        assert not self._done, "the indexer has already been initialized."
        with open(index_file, "r") as fr:
            for line in fr:
                line = line.strip().split()
                word, idx = line[0], int(line[1])
                if self.word2idx.get(word, None) is None:
                    assert idx == len(self.word2idx)
                    self.word2idx[word] = idx
        for word, idx in self.word2idx.items():
            self.idx2word[idx] = word
        self._done = True

    def from_list(self, word_list):
        assert not self._done, "the indexer has already been initialized."
        for _, word in enumerate(word_list):
            if self.word2idx.get(word, None) is None:
                self.word2idx[word] = len(self.word2idx)
        for word, idx in self.word2idx.items():
            self.idx2word[idx] = word
        self._done = True

    def idx(self, token):
        return self.word2idx.get(token, self.word2idx[self.UNK])

    def str(self, idx):
        return self.idx2word.get(idx, self.UNK)

    def has(self, token):
        return token in self.word2idx

    def write(self, ofile):
        with open(ofile, "w") as fw:
            for i in len(self):
                fw.write(f"{self.idx2word[i]} {i}\n")

    def __getitem__(self, idx):
        return self.idx(idx)

    def __call__(self, key):
        if isinstance(key, int):
            return self.str(key)
        elif isinstance(key, str):
            return self.idx(key)
        elif isinstance(key, list):
            return [self(k) for k in key]
        else:
            raise ValueError("type({}) must be `int` or `str`".format(key))

    def __len__(self):
        return len(self.word2idx)

