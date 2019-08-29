#!/bin/bash

# Add `--gpu <GPU_ID>` option to use GPU.

### Experiments of Table. 1
## Proposed method
# Wiki-90k dataset
python main_graph.py --K 10 --weight_aggregation --init_wordvec

# WF-20k dataset
python main_graph.py --K 5 --weight_aggregation --init_wordvec --data wf20k.data.json

## (Song et al., 2018) baseline
# Wiki-90k dataset
python baseline_gslstm_reimpl.py --n_layers 5 --init_wordvec

# WF-20k dataset
python baseline_gslstm_reimpl.py --dim_node 50 --dropout 0.0 --n_layers 3 --data wf20k.data.json

## Model F (Toutanova et al., 2015) baseline
# Wiki-90k dataset
python baseline_gslstm_modelf.py --lr 1e-3 --decay 1e-8 --dim_state 300 --dim_node 50 \
  --dim_link 3 --dropout 0 --node_dropout 0 --K 5 --n_layers 5 --epoch 50 --init_wordvec

# WF-20k dataset
python baseline_gslstm_modelf.py --lr 1e-3 --decay 1e-6 --dim_state 300 --dim_node 50 \
  --dim_link 3 --dropout 0.1 --node_dropout 0.1 --K 5 --n_layers 5 --epoch 50 --init_wordvec \
  --normalize --data wf20k.data.json

## Model E (Toutanova et al., 2015) baseline
# Wiki-90k dataset
python baseline_gslstm_modele.py --lr 1e-4 --decay 1e-8 --dim_state 100 --dim_node 100 \
  --dim_link 3 --dropout 0.1 --node_dropout 0.0 --K 5 --n_layers 5 --epoch 50 --init_wordvec \
  --normalize

# WF-20k dataset
python baseline_gslstm_modele.py --lr 1e-4 --decay 1e-6 --dim_state 100 --dim_node 100 \
  --dim_link 10 --dropout 0.1 --node_dropout 0.2 --K 5 --n_layers 5 --epoch 50 --init_wordvec \
  --normalize --data wf20k.data.json

## Model F (Verga et al., 2017) baseline
# Wiki-90k dataset
python baseline_verga_modelf.py --lr 1e-4 --decay 1e-4 --dim_state 100 --dim_node 50 \
  --dim_link 10 --dropout 0.1 --node_dropout 0.1 --K 5 --n_layers 5 --epoch 50 --init_wordvec \
  --normalize

# WF-20k dataset
python baseline_verga_modelf.py --lr 1e-4 --decay 1e-4 --dim_state 300 --dim_node 50 \
  --dim_link 10 --dropout 0 --node_dropout 0.1 --K 5 --n_layers 5 --epoch 50 --init_wordvec \
  --normalize --data wf20k.data.json

### Experiments of Table. 2
## U
python main_graph.py --K 20 --disableB --init_wordvec

## B
python main_graph.py --K 20 --disableU --init_wordvec

## N
python main_graph.py --K 20 --weight_aggregation --disableB --disableU --init_wordvec

## U+B
python main_graph.py --K 20 --init_wordvec

## U+B+N
K=20
python main_graph.py --K ${K} --weight_aggregation --init_wordvec

## U+B+N (fix aggregation weights)
python main_graph.py --K 20 --weight_aggregation --fix_weight --init_wordvec

### Experiments of Table. 3
## \alpha = 0
LABEL_RATIO=0.4
python main_graph.py --K 10 --M 2 --init_wordvec --label_ratio ${LABEL_RATIO}

## 0 < \alpha < \infty
LABEL_RATIO=0.4
ALPHA=1.0
python main_graph.py --weight_aggregation --wa_weight ${ALPHA} --K 10 --M 2 \
  --init_wordvec --label_ratio ${LABEL_RATIO} --mid_eval 3 --epoch 4

## \alpha = \infty
LABEL_RATIO=0.4
python main_graph.py --weight_aggregation --wa_weight 1 --K 10 --M 2 --init_wordvec \
  --disableB --disableU --label_ratio ${LABEL_RATIO} --mid_eval 3 --epoch 4
