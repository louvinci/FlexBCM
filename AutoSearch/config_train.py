import sys
import time
import numpy as np

from easydict import EasyDict as edict

C = edict()
config = C
cfg = C


# C.dataset = 'imagenet'
# C.backbone='RN18'

C.dataset = 'cifar100'
C.backbone='RN34'

if 'cifar' in C.dataset:
    """Data Dir and Weight Dir"""
    #C.dataset_path = "F:\Paper\Pytorch\cifar_data"
        
    if C.dataset == 'cifar10':
        C.num_classes = 10
        C.dataset_path = "/home/lwq/cifar_data"
    elif C.dataset == 'cifar100':
        C.num_classes = 100
        C.dataset_path = "/home/lwq/cifar100"
    else:
        print('Wrong dataset.')
        sys.exit()

    """Image Config"""
    C.image_height = 32 # this size is after down_sampling
    C.image_width = 32
    C.dali = False
    C.dali_cpu=False
    ####################  Modle Config #################### 

    C.load_path = 'ckpt/cifar/search'
   

    C.bn_eps = 1e-5
    C.bn_momentum = 0.1
   ####################  Train Config #################### 
    C.opt = 'Sgd'
    C.momentum = 0.9
    C.weight_decay = 5e-4
    C.betas=(0.5, 0.999)
    C.num_workers = 4
    C.pretrain = "ckpt/cifar/finetune-{0}".format(C.backbone)

    C.batch_size = 256
    C.image_height = 32 # this size is after down_sampling
    C.image_width = 32

    C.save = "finetune-{0}".format(C.backbone)
    C.nepochs = 200 #600->300
    C.eval_epoch = 1
    C.lr_schedule = 'cosine'
    C.lr = 0.1
    # linear 
    C.decay_epoch = 100
    # exponential
    C.lr_decay = 0.97
    # multistep
    C.milestones = [80, 120, 160]
    C.gamma = 0.1
    # cosine
    C.learning_rate_min = 0.001
    C.eval_only = False
    C.efficiency_metric = 'flops'

    ####
    C.grad_clip = 5
    C.autoaugment = False

    C.label_smoothing = False

    C.cutmix = False
    C.beta = 1
    C.cutmix_prob = 1
    
elif C.dataset == 'imagenet':
    #C.dataset_path = "/root/datasets/imagenet"
    C.dataset_path = "/mnt/data/ImageNet"
    C.num_workers = 8 # workers per gpu
    C.batch_size = 512#512#384 for RN34
    C.dali = True
    C.dali_cpu=False
    C.num_classes = 1000
    C.image_height = 224 # this size is after down_sampling
    C.image_width = 224

    ####################  Modle Config #################### 
    C.load_path = 'ckpt/IMG/search'


    ####################  Train Config #################### 

    C.opt = 'Sgd'

    C.momentum = 0.9
    C.weight_decay = 4e-5

    C.betas=(0.5, 0.999)


    """ Search Config """

    C.pretrain = "ckpt/IMG/finetune-{0}".format(C.backbone)

        
    ########################################

    C.save = "finetune-{0}".format(C.backbone)
    ########################################

    # C.nepochs = 360
    C.nepochs = 250

    C.eval_epoch = 1

    C.lr_schedule = 'cosine'
    C.lr = 0.1

    # linear 
    C.decay_epoch = 100
    # exponential
    C.lr_decay = 0.97
    # multistep
    C.milestones = [90, 180, 270]
    C.gamma = 0.1
    # cosine
    C.learning_rate_min = 0

    C.eval_only = False

    C.efficiency_metric = 'flops'
    ####
    C.grad_clip = 5
    C.autoaugment = False

    C.label_smoothing = True

    C.cutmix = True
    C.beta = 1
    C.cutmix_prob = 1
    
    C.efficiency_metric = 'latency'
    C.platform = 'zcu102'
else:
    print('Wrong dataset.')
    sys.exit()