# XCFGs

Aiming at unifying all extensions of context-free grammars (XCFGs). **X** stands for weighted, (compound) probabilistic, and neural extensions, etc.

## Data

The repo handles [WSJ](https://catalog.ldc.upenn.edu/LDC99T42), [CTB](https://catalog.ldc.upenn.edu/LDC2005T01), and [SPMRL](https://dokufarm.phil.hhu.de/spmrl2014/). Have a look at `treebank.py`.

If you are looking for the data used in [C-PCFGs](https://github.com/zhaoyanpeng/cpcfg). Follow the instructions in `treebank.py` and put all outputs in the same folder, let us say `./data.punct`. The script only removes morphology features and creates data splits. To remove punctuation we will need `clean_tb.py`. For example, I used `python clean_tb.py ./data.punct ./data.clean`. All the cleaned treebanks will reside in `/data.clean`.  Then simply execute the command `./batchify.sh ./data.clean/`, you will have all the data needed to reproduce the results in [C-PCFGs](https://github.com/zhaoyanpeng/cpcfg). Feel free to change parameters in `batchify.sh` if you want to use a different batch size or vocabulary size.

## Evaluation
To convenient evaluation I represent a gold tree as a tuple:
```
TREE: TUPLE(sentence: STR, spans: LIST[SPAN], span_labels: LIST[STR], pos_tags: LIST[STR])
SPAN: TUPLE(left_boundary: int, right_boundary: int)
```
If you have followed the instructions in the last section, this command `./binarize.sh ./data.clean/` could help you convert gold trees into the tuple representation. 

### Trivial baselines

Even for trivial baselines, e.g., left- and right-branching trees, you may find different F1 numbers in literature on grammar induction, partly because authors used (slightly) different procedures for data preprocessing. To encourage truly fair comparison I will release a standard procedure `baseline.py` soon. Actually, you can check it out now. It may be a bit user-unfriendly as the script lacks necessary code comments. Will update soon!

## Citing XCFGs

If you use XCFGs in your research or wish to refer to the results in [C-PCFGs](https://github.com/zhaoyanpeng/cpcfg), please use the following BibTeX entry.
```
@article{zhao2020xcfg,
  author = {Zhao, Yanpeng},
  title  = {An Empirical Study of Compound PCFGs},
  journal= {https://github.com/zhaoyanpeng/cpcfg},
  url    = {https://github.com/zhaoyanpeng/cpcfg},
  year   = {2020}
}

```
## Acknowledgements
`batchify.py` is borrowed from [C-PCFGs](https://github.com/harvardnlp/compound-pcfg).

## License
MIT
