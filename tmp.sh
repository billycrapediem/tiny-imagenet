torchrun --nproc_per_node=1  classification/train.py \
    --model 'resnet18' \
    --batch-size 256 \
    --epochs 50 \
    --opt 'sgd' \
    --lr 0.2 \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --lr-scheduler 'cosineannealinglr' \
    --lr-warmup-epochs 5 \
    --lr-warmup-method 'linear' \
    --lr-warmup-decay 0.01 \
    --output-dir 'save/rn18_50epdistill'\
    --syn-data-path '/scratch/bzhang44/cv/tiny-imagenet/data/sre2l_tiny_rn18_4k_ipc100' \


    torchrun --nproc_per_node=1  classification/train.py \
    --model 'resnet18' \
    --batch-size 256 \
    --epochs 100 \
    --opt 'sgd' \
    --lr 0.2 \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --lr-scheduler 'cosineannealinglr' \
    --lr-warmup-epochs 5 \
    --lr-warmup-method 'linear' \
    --lr-warmup-decay 0.01 \
    --output-dir 'save/rn18_100ep'

torchrun --nproc_per_node=1  classification/train.py \
    --model 'resnet18' \
    --batch-size 256 \
    --epochs 200 \
    --opt 'sgd' \
    --lr 0.2 \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --lr-scheduler 'cosineannealinglr' \
    --lr-warmup-epochs 5 \
    --lr-warmup-method 'linear' \
    --lr-warmup-decay 0.01 \
    --output-dir 'save/rn18_200ep'


    # log file: log/sre2l_tiny_4k_ipc100_rn18.log
python classification/train_kd.py \
    --model 'resnet18' \
    --teacher-model 'resnet18' \
    --teacher-path '/path/to/resnet18_E50/checkpoint.pth' \
    --batch-size 256 \
    --epochs 100 \
    --opt 'sgd' \
    --lr 0.2 \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --lr-scheduler 'cosineannealinglr' \
    --lr-warmup-epochs 5 \
    --lr-warmup-method 'linear' \
    --lr-warmup-decay 0.01 \
    --syn-data-path '/path/to/sre2l_tiny_rn18_4k_ipc100'
    -T 20 \
    --image-per-class 100 \
    --output-dir 'save_kd/T18S18_T20_[4K].ipc_100'