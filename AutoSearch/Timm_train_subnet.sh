

MODEL=SubFormer

#imagenet
#DIR=/home/tsc/data/ImageNet
DIR=/mnt/data/ImageNet
#DATASET='imagenet-200'
DATASET='imagenet'
CUDA_VISIBLE_DEVICES=0 nohup ./distributed_train.sh 1 $DIR --dataset $DATASET --opt-eps 1e-8 --model $MODEL  --opt 'adam' --clip-grad 5 --amp --checkpoint-hist 1 --lr 5e-4 --epochs 300 --input-size 3 224 224 -b 1024 --weight-decay 4e-5 --pin-mem > train_log/IMG/DeiT_tiny_3.log 2>&1 &

#MODEL=resnet20_cifar10_Q
#cifar10
#DIR=/home/tsc/data/cifar_data
#CUDA_VISIBLE_DEVICES=0,2,3 nohup ./distributed_train.sh 3 $DIR --dataset 'cifar10' --model $MODEL --checkpoint-hist 1 --num-classes 10 --input-size 3 32 32 -b 128 --opt 'adam' --epochs 200 --weight-decay 1e-5 --lr 5e-4 > log/cifar/dk_soft_2op.txt 2>&1 &


# MODEL=regnety_040

# #imagenet
# #DIR=/home/tsc/data/ImageNet
# DATASET='imagenet-100'
# DIR=/mnt/data/ImageNet
# CUDA_VISIBLE_DEVICES=0 ./distributed_train.sh 1 $DIR --model $MODEL --dataset $DATASET --pretrain --num-classes 100 --opt 'adam' --clip-grad 5 --checkpoint-hist 1 --lr 5e-5 --epochs 1  -b 256 --weight-decay 4e-5 --pin-mem > regnety_finetune.log 2>&1 &
