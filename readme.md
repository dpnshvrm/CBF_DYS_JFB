# Command to run script

```
python train.py --problem double_integrator_single  --epochs 1000 --lr_decay 400 --weight_decay 0.0001

python train.py --problem double_integrator_multi   --epochs 5000 --lr_decay 800 --hidden_dim 128  --weight_decay 0.005

python train.py --problem single_integrator_swarm   --epochs 3500 --lr_decay 800 --hidden_dim 192 --n_blocks 4

python train.py --problem quadcopter_5   --epochs 2800 --lr_decay 600 --hidden_dim 128

python train.py --problem quadcopter_30  --epochs 4500 --lr_decay 800 --hidden_dim 128 --n_blocks 4

python train.py --problem quadcopter_100  --epochs 5000 --lr_decay 800 --hidden_dim 192 --n_blocks 4

python train_finetune.py --problem quadcopter_30_finetune --epochs 2000 --alpha_terminal_final  75.0

python train_finetune.py --problem quadcopter_100_finetune  --epochs 2000 --alpha_terminal_final 50.0 --plot

```