# Implementation of Cross-Sentence N-ary Relation Extraction using Lower-Arity Universal Schemas

This repository contains implementations of the proposed method and baseline methods tested in our paper "Cross-Sentence N-ary Relation Extraction using Lower-Arity Universal Schemas" (To be appeared in EMNLP 2019).

## Data

Following data files are required to run codes.
Also, see [data/README.md](data/README.md) for the dataset format.

Wiki-90k and WF-20k dataset is available [here](https://github.com/aurtg/n-ary-dataset).

```
data/glove.6B.300d.txt # You can download it from here (http://nlp.stanford.edu/data/glove.6B.zip).
data/Wiki-90k/train
data/Wiki-90k/dev
data/Wiki-90k/test
data/WF-20k/train.json
data/WF-20k/dev.json
data/WF-20k/test.json
```

## Proposed method and baseline methods

### Required environment

Codes of our proposed method are tested in the following environment.


* python (3.6.8)
  * numpy (1.16.2)
  * scikit-learn (0.20.3)
  * pytorch (1.1.0)
  * torch_scatter (1.3.1, https://github.com/rusty1s/pytorch_scatter)
  * networkx (2.2)

### How to run

Before running codes, create `logs` directory. Results of experiments will be output in log files `logs/ExpLog_<suffix>_<exp_number>.log`. You can set `suffix` and `exp_number` by options when you start experiments.

To run experiments with the same settings in the paper, execute commands described in `example.sh`.

> NOTES:
>
> * You can use GPU by specifying its id by `--gpu <id>` option, or the codes will use CPU (slow).
> * You can set a name and number of an experiment by using `--suffix <NAME>` and `--exp_number <NUM>` options.
