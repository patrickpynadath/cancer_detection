#!/bin/sh
python main.py train_clf --sample_strat none  --learning_mode jigsaw --accelerator gpu --devices 1 --device cuda --max_epochs 200 --criterion CE --track_grad_norm 2 --auto_lr_find True;
python main.py train_clf --sample_strat none  --learning_mode jigsaw --accelerator gpu --devices 1 --device cuda --max_epochs 200 --criterion MSFE --lr .00001  --track_grad_norm 2;
python main.py train_clf --sample_strat none  --learning_mode jigsaw --accelerator gpu --devices 1 --device cuda --max_epochs 200 --criterion CE --track_grad_norm 2 --use_encoder_params True --auto_lr_find True;
python main.py train_clf --sample_strat none  --learning_mode jigsaw --accelerator gpu --devices 1 --device cuda --max_epochs 200 --criterion MSFE --lr .00001  --track_grad_norm 2 --use_encoder_params True;
python main.py train_clf --sample_strat ros  --learning_mode jigsaw --accelerator gpu --devices 1 --device cuda --max_epochs 200 --criterion MSFE --lr .00001  --track_grad_norm 2;
python main.py train_clf --sample_strat rus  --learning_mode jigsaw --accelerator gpu --devices 1 --device cuda --max_epochs 200 --criterion MSFE --lr .00001  --track_grad_norm 2;
python main.py train_clf --sample_strat ros  --learning_mode jigsaw --accelerator gpu --devices 1 --device cuda --max_epochs 200 --criterion CE  --track_grad_norm 2 --auto_lr_find True;
python main.py train_clf --sample_strat rus  --learning_mode jigsaw --accelerator gpu --devices 1 --device cuda --max_epochs 200 --criterion CE  --track_grad_norm 2 --auto_lr_find True;
python main.py train_clf --sample_strat dynamic_ros  --learning_mode jigsaw --accelerator gpu --devices 1 --device cuda --max_epochs 200 --criterion MSFE --lr .00001  --track_grad_norm 2;
python main.py train_clf --sample_strat dynamic_ros  --learning_mode jigsaw --accelerator gpu --devices 1 --device cuda --max_epochs 200 --criterion CE  --track_grad_norm 2 --auto_lr_find True;
python main.py train_clf --sample_strat dynamic_kmeans_ros  --learning_mode jigsaw --accelerator gpu --devices 1 --device cuda --max_epochs 200 --criterion MSFE --lr .00001  --track_grad_norm 2;
python main.py train_clf --sample_strat dynamic_kmeans_ros  --learning_mode jigsaw --accelerator gpu --devices 1 --device cuda --max_epochs 200 --criterion CE --track_grad_norm 2 --auto_lr_find True;
