CUDA_VISIBLE_DEVICES=2 nohup python train_search.py > search_log/search-RN50-2.txt 2>&1 &
#CUDA_VISIBLE_DEVICES=0 nohup python train_search.py > log/search-2depth6_lr_GS_one.txt 2>&1 &