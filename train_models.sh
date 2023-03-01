#!/bin/sh
python main.py train_clf --sample_strat dynamic_ros  --learning_mode jigsaw --accelerator gpu --devices 50 --device cuda --epochs 50 --criterion MSFE --lr .00001  --track_grad_norm 2 --balancing_beta .5;
python main.py train_clf --sample_strat dynamic_kmeans_ros  --learning_mode jigsaw --accelerator gpu --devices 50 --device cuda --epochs 50 --criterion MSFE --lr .00001  --track_grad_norm 2 --balancing_beta .5;
