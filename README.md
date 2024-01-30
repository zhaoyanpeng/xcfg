# XCFGs

Aiming at unifying all extensions of context-free grammars (XCFGs). **X** stands for weighted, (compound) probabilistic,
and neural extensions, etc. Currently only the data preprocessing module has been implemented though.

**Update (08/06/2023):** Support [Brown Corpus](https://catalog.ldc.upenn.edu/LDC99T42) and [English Web Treebank](https://catalog.ldc.upenn.edu/LDC2012T13) that are used in this [study](https://aclanthology.org/2023.findings-emnlp.530/).

**Update (06/02/2022):** Parse MSCOCO and Flickr30k captions, create data splits, and encode images for [VC-PCFG](https://github.com/zhaoyanpeng/vpcfg).

**Update (03/10/2021):** Parallel Chinese-English data is supported.

## Data

The repo handles [WSJ](https://catalog.ldc.upenn.edu/LDC99T42), [CTB](https://catalog.ldc.upenn.edu/LDC2005T01), [SPMRL](https://dokufarm.phil.hhu.de/spmrl2014/), [Brown Corpus](https://catalog.ldc.upenn.edu/LDC99T42), and [English Web Treebank](https://catalog.ldc.upenn.edu/LDC2012T13). Have a look at `treebank.py`.

If you are looking for the data used in [C-PCFGs](https://arxiv.org/abs/2103.02298). Follow the instructions in `treebank.py` and put all outputs in the same folder, let us say `./data.punct`. The script only removes morphology features and creates data splits. To remove punctuation we will need `clean_tb.py`. For example, I used `python clean_tb.py ./data.punct ./data.clean`. All the cleaned treebanks will reside in `/data.clean`.  Then simply execute the command `./batchify.sh ./data.clean/`, you will have all the data needed to reproduce the results in [C-PCFGs](https://arxiv.org/abs/2103.02298). Feel free to change parameters in `batchify.sh` if you want to use a different batch size or vocabulary size.

## Evaluation
To ease evaluation I represent a gold tree as a tuple:
```
TREE: TUPLE(sentence: STR, spans: LIST[SPAN], span_labels: LIST[STR], pos_tags: LIST[STR])
SPAN: TUPLE(left_boundary: INT, right_boundary: INT)
```
If you have followed the instructions in the last section, this command `./binarize.sh ./data.clean/` could help you convert gold trees into the tuple representation. 

### Trivial baselines

Even for trivial baselines, e.g., left- and right-branching trees, you may find different F1 numbers in literature on grammar induction, partly because the authors used (slightly) different procedures for data preprocessing. *To encourage truly fair comparison I also released a standard procedure `baseline.py`.* Hopefully, this will help with the situation.

| Model | WSJ | CTB | Basque | German | French | Hebrew | Hungarian | Korean | Polish | Swedish |
|:-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
| LB | 8.7 | 7.2 | 17.9 | 10.0 | 5.7 | 8.5 | 13.3 | 18.5 | 10.9 | 8.4 |
| RB | 39.5 | 25.5 | 15.4 | 14.7 | 26.4 | 30.0 | 12.7 | 19.2 | 34.2 | 30.4 |

### An evaluation checklist for phrase-structure grammar induction

Below is a comparison of several cirtical training / evaluation settings of recent unsupervised parsing models.

| Model | Sent. F1 | Corpus F1 | Variance | Word repr. | Punct. rm | Length | Dataset |
|:-:|-:|-:|-:|-:|-:|-:|-:|
| [PRPN](https://openreview.net/forum?id=rkgOLb-0W) | &check; |  |  | RAW | &check; |  | WSJ | |
| [ON](https://openreview.net/forum?id=B1l6qiR5F7) | &check; |  |  | RAW | &check; |  | WSJ |  |
| [DIORA](https://doi.org/10.18653/v1/N19-1116) | &check; |  |  | ELMo |  |  | WSJ |  |
| [URNNG](https://doi.org/10.18653/v1/N19-1114) | &check; |  |  | RAW | &cross; |  | WSJ |  |
| [N-PCFG](https://doi.org/10.18653/v1/P19-1228) | &check; |  |  | RAW | &check; |  | WSJ / CTB |  |
| [C-PCFG](https://doi.org/10.18653/v1/P19-1228) | &check; |  |  | RAW | &check; |  | WSJ / CTB |  |
| [VG-NSL](https://doi.org/10.18653/v1/P19-1180) | &check; |  | &check; | RAW / FastText | &cross; |  | MSCOCO |  |
| [LN-PCFG](http://arxiv.org/abs/2007.15135) | &check; |  |  | RAW |  |  | WSJ |  |
| [CT](https://www.aclweb.org/anthology/2020.emnlp-main.389) | &check; |  |  | RoBERTa |  |  | WSJ |  |
| [S-DIORA](https://www.aclweb.org/anthology/2020.emnlp-main.392) | &check; |  |  | ELMo |  |  | WSJ |  |
| [VC-PCFG](https://www.aclweb.org/anthology/2020.emnlp-main.354) | &check; | &check; | &check; | RAW | &check; |  | MSCOCO |  |
| [C-PCFG (Zhao 2020)](https://arxiv.org/abs/2103.02298) | &check; | &check; | &check; | RAW | &check; |  | WSJ / CTB / SPMRL |  |


## Citing XCFGs

If you use XCFGs in your research or wish to refer to the results in [C-PCFGs](https://arxiv.org/abs/2103.02298), please use the following BibTeX entry.
```
@inproceedings{zhao-titov-2023-transferability,
    title = "On the Transferability of Visually Grounded {PCFGs}",
    author = "Zhao, Yanpeng  and Titov, Ivan",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
}
```
```
@inproceedings{zhao-titov-2021-empirical,
    title = "An Empirical Study of Compound {PCFG}s",
    author = "Zhao, Yanpeng and Titov, Ivan",
    booktitle = "Proceedings of the Second Workshop on Domain Adaptation for NLP",
    month = apr,
    year = "2021",
    address = "Kyiv, Ukraine",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.adaptnlp-1.17",
    pages = "166--171",
}
```
## Acknowledgements
`batchify.py` is borrowed from [C-PCFGs](https://github.com/harvardnlp/compound-pcfg).

## License
MIT
