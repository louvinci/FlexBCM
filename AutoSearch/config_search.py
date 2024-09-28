# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
import time
import numpy as np

from easydict import EasyDict as edict


C = edict()
config = C
cfg = C
C.seed = 12345

"""please config ROOT_dir and user when u first using"""
C.repo_name = 'AMB-CNN'


C.gpu = None

#C.dataset = 'cifar10'
#C.dataset = 'imagenet'
C.dataset = 'cifar100'
#C.backbone='RN34'

if 'cifar' in C.dataset:
    #C.dataset_path = "F:\Paper\Pytorch\cifar_data"
    #C.dataset_path = "/root/workspace/cifar_data"
    #C.dataset_path = "/home/lwq/workspace/cifar_data"
    
    #C.dataset_path = "/data/cifar_data"
    if C.dataset == 'cifar10':
        C.num_classes = 10
        #C.dataset_path = "/home/tsc/data/cifar_data"
        C.dataset_path = "/data/cifar_data"
        C.backbone='RN18'
    elif C.dataset == 'cifar100':
        C.num_classes = 100
        C.dataset_path = "/home/lwq/cifar100"
        C.backbone='RN34'
    else:
        print('Wrong dataset.')
        sys.exit()


    """ Settings for network, this would be different for each kind of model"""
    C.bn_eps = 1e-5
    C.bn_momentum = 0.1

    # backbone RN18 or RN34
    

    """Train Config"""

    C.opt = 'Sgd'

    C.momentum = 0.9
    C.weight_decay = 5e-4

    C.betas=(0.5, 0.999)
    C.num_workers = 8

    C.nb_classes= 10

    """ Search Config """
    C.grad_clip = 5

    C.pretrain = 'ckpt/cifar/search'


    C.train_mode = 'iterwise'  # 'epochwise'

    C.sample_func = 'gumbel_softmax'
    C.temp_init = 5
    C.temp_decay = 0.975

    ## Gumbel Softmax settings for operator
    C.mode = 'soft'  # "soft", "hard", "fake_hard", "random_hard", "proxy_hard"

    C.offset = True and C.mode == 'proxy_hard'
    C.act_num = 1



    C.pretrain_epoch = 30 # initial-30
    C.pretrain_aline = True

    if C.pretrain_aline:
        C.pretrain_mode = C.mode
        C.pretrain_act_num = C.act_num
    else:
        C.pretrain_mode = 'soft'
        C.pretrain_act_num = 1

    C.arch_one_hot_loss_weight = None
    C.arch_mse_loss_weight = None


    C.hw_aware_nas = False
    ###train set####

    C.batch_size = 64 # 64->128

    C.save = "cifar/search"

    C.nepochs = 45 + C.pretrain_epoch
    C.eval_epoch = 1

    C.lr_schedule = 'cosine'
    C.lr = 0.1 # 0.025-> 0.05
    # linear 
    C.decay_epoch = 20
    # exponential
    C.lr_decay = 0.97
    # multistep
    C.milestones = [50, 100, 200]
    C.gamma = 0.1
    # cosine
    C.learning_rate_min = 0.001

    #########

    C.train_portion = 0.5  # 0.8

    C.unrolled = False

    # this parameter is lr of architecture parameter optimizer
    C.arch_learning_rate = 3e-4#C.lr #3e-4 

    C.arch_update_frec = 1

    # hardware cost
    C.efficiency_metric = 'flops' # 'latency'/'flops'

    # hardware cost weighted coefficients
    
    # FLOPs, 332608896.0 (330M), 5182090.0 (5.18Million) cifar_vit
    C.flops_mode = 'single_path' # 'single_path', 'multi_path'
    C.flops_weight = 0 #1e-10 #1e-10 #0->1e-10
    C.flops_max = 2e8 # 
    C.flops_min = 4e7 # 
    C.flops_decouple = False

    #FPS
    # the parameter is set for adjusting the weight of flops/fps value 
    C.latency_weight = 0  # 1e-7 - 1e-14 set 0 
    C.hw_update_freq = 10   # update hw every xx arch param update
    C.fps_max = 120
    C.fps_min = 60
    C.Populations=250#200
    C.Generations=50#20


##########################################################imagenet###############################
elif  'imagenet' in C.dataset:
    """Data Dir and Weight Dir"""
    #C.dataset_path = "/home/tsc/data/ImageNet"
    C.dataset_path = "/mnt/data/ImageNet" # Specify path to ImageNet-100
    #C.dataset_path = "/data/ImageNet"
    C.batch_size = 128#196
    C.num_workers = 8

    """Image Config"""
    # imageNet
    C.image_height = 224
    C.image_width = 224

    C.num_classes = 100
    C.image_channels = 3
    C.backbone='RN50'

    C.train_mode = 'iterwise'  # 'epochwise'
    """ Settings for network, this would be different for each kind of model"""
    C.bn_eps = 1e-5
    C.bn_momentum = 0.1

    """Train Config"""

    C.opt = 'Sgd'

    C.momentum = 0.9
    C.weight_decay = 5e-4 # sgd

    C.betas=(0.5, 0.999) # adam optimizer

    """ Search Config """
    C.grad_clip = 5

    C.pretrain = 'ckpt/IMG/search'


    C.sample_func = 'gumbel_softmax'
    C.temp_init = 5
    C.temp_decay = 0.956

    ## Gumbel Softmax settings for operator
    C.mode = 'hard'  # "soft", "hard",  "random_hard", "proxy_hard"(no support)
    C.offset = True and C.mode == 'proxy_hard'
    C.act_num = 2


    C.pretrain_epoch = 20#45
    C.pretrain_aline = True

    if C.pretrain_aline:
        C.pretrain_mode = C.mode
        C.pretrain_act_num = C.act_num
    else:
        C.pretrain_mode = 'soft'
        C.pretrain_act_num = 1

    C.arch_one_hot_loss_weight = None
    C.arch_mse_loss_weight = None

    C.num_sample = 10


    C.hw_aware_nas = False
    ########################################
    
    C.save = "IMG/search"

    C.nepochs = C.pretrain_epoch + 40#30 # 75
    C.eval_epoch = 1

    C.lr_schedule = 'cosine'
    C.lr = 0.1
    # linear 
    C.decay_epoch = 20
    # exponential
    C.lr_decay = 0.97
    # multistep
    C.milestones = [50, 100, 200]
    C.gamma = 0.1
    # cosine
    C.learning_rate_min = 0


    ########################################

    C.train_portion = 0.8

    C.unrolled = False

    C.arch_learning_rate = 3e-4

    C.arch_update_frec = 1

    # hardware cost
    C.efficiency_metric = 'latency' # 'latency'
    
    # FLOPs
    C.flops_mode = 'single_path' # control the forward flops mode{'sum','single path','multi path'}
    C.flops_weight = 1e-10#0 # 
    C.flops_max = 1.4e9
    C.flops_min = 1e9
    C.flops_decouple = False
    
    
    C.platform = "zcu102"
    if "706" in C.platform:
        C.Mhz = 150
    else:
        C.Mhz = 200
    
    #FPS
    # the parameter is set for adjusting the weight of flops/fps value 
    C.Populations=200#200
    C.Generations=40#20
    C.latency_weight = 1e-8  # 1e-7 - 1e-14 set 0 
    C.hw_update_freq = 10   # update hw every xx arch param update
    if config.backbone == 'RN18':
        C.fps_max = 180
        C.fps_min = 150
    elif config.backbone == 'RN34':
        C.fps_max = 110
        C.fps_min = 90
    elif config.backbone == 'RN50':
        C.fps_max = 70
        C.fps_min = 30
    else:
        C.fps_max = 120
        C.fps_min = 60

else:
    print('Wrong dataset.')
    sys.exit()