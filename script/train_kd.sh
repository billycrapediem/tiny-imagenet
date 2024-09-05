cd ..

# log file: log/sre2l_tiny_4k_ipc50_rn18.log
python classification/train_kd.py \
    --model 'resnet18' \
    --teacher-model 'resnet18' \
    --teacher-path '/scratch/bzhang44/cv/tiny-imagenet/save/rn18_50ep/checkpoint_best.pth' \
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
    --syn-data-path '/scratch/bzhang44/cv/tiny-imagenet/data/sre2l_tiny_rn18_4k_ipc100' \
    -T 20 \
    --image-per-class 50 \
    --output-dir 'save_kd/T18S18_T20_[4K].ipc_50'


