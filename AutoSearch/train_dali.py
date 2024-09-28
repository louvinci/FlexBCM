from __future__ import division
import sys
import time
import glob
import logging
#from tqdm import tqdm
import os

import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as dset
from torch.autograd import Variable

import torchvision

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

from tensorboardX import SummaryWriter

from config_train import config

from datasets import prepare_train_data, prepare_train_data_autoaugment, prepare_test_data
from datasets import prepare_train_data_for_search, prepare_test_data_for_search
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

import numpy as np

from config_train import config
import genotypes

from model_infer import SubNet_Infer

from lr import LambdaLR

from thop import profile
from thop.vision.basic_hooks import count_convNd

from operations import count_BCMConv
from circulant_2d import BCMConv2d_FFT_Mconv
custom_ops = {BCMConv2d_FFT_Mconv: count_BCMConv}
import argparse


parser = argparse.ArgumentParser(description='AMB')
parser.add_argument('--dataset_path', type=str, default=None,
                    help='path to ImageNet-100')
parser.add_argument('-b', '--batch_size', type=int, default=None,
                    help='batch size')
parser.add_argument('--num_workers', type=int, default=None,
                    help='number of workers per gpu')
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()


best_acc = 0
best_epoch = 0

@pipeline_def
def create_dali_pipeline(data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True):
    images, labels = fn.readers.file(file_root=data_dir,
                                     shard_id=shard_id,
                                     num_shards=num_shards,
                                     random_shuffle=is_training,
                                     pad_last_batch=True,
                                     name="Reader")
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
    if is_training:
        images = fn.decoders.image_random_crop(images,
                                               device=decoder_device, output_type=types.RGB,
                                               device_memory_padding=device_memory_padding,
                                               host_memory_padding=host_memory_padding,
                                               preallocate_width_hint=preallocate_width_hint,
                                               preallocate_height_hint=preallocate_height_hint,
                                               random_aspect_ratio=[0.8, 1.25],
                                               random_area=[0.1, 1.0],
                                               num_attempts=100)
        images = fn.resize(images,
                           device=dali_device,
                           resize_x=crop,
                           resize_y=crop,
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
        images = fn.resize(images,
                           device=dali_device,
                           size=size,
                           mode="not_smaller",
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = False

    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                      std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                      mirror=mirror)
    labels = labels.gpu()
    return images, labels

def main():
    if args.dataset_path is not None:
        config.dataset_path = args.dataset_path
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers

    # Simply call main_worker function
    main_worker(config)


def main_worker(config):
    global best_acc
    global best_epoch

    pretrain = config.pretrain


    # logging.info("config = %s", str(config))
    # # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    #torch.set_float32_matmul_precision('medium')
    # seed = config.seed
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)


    
    if type(pretrain) == str:
        config.save = pretrain
    else:
        config.save = 'train/{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))

    logger = SummaryWriter(config.save)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info("args = %s", str(config))



    # Model #######################################

    state = torch.load(os.path.join(config.load_path, 'arch.pt'))
    alpha = state['alpha']

    #model = SubNet_Infer(alpha, config=config)
    model = SubNet_Infer(alpha, config=config)
    #model = torch.compile(p_model) # mode='reduce-overhead'
    logging.info("model = %s",str(model))
    #print(model)
    #return
    
    
    flops, params = profile(model, inputs=(torch.randn(1, 3, config.image_height, config.image_width),), custom_ops=custom_ops)
        
    logging.info("params = %fM, FLOPs = %fM", params / 1e6, flops / 1e6)
    #return
    if config.efficiency_metric == 'latency':
        fps, searched_hw = model.eval_latency(config.backbone, Populations = 250,Generations= 50, platform = config.platform,
                                                        hardware= None)
        logging.info("FPS of Searched Arch:" + str(fps))


    model = torch.nn.DataParallel(model).cuda()


    # for param, val in model.named_parameters():
    #     print(param, val.device)
        
    #     if val.device.type == 'cpu':
    #         print('This tensor is on CPU.')
    #         sys.exit()

    trainable_parameters = filter(lambda x : x.requires_grad, model.parameters())
    
    if config.opt == 'Adam':
        optimizer = torch.optim.Adam(
            trainable_parameters,
            lr=config.lr,
            betas=config.betas)
    elif config.opt == 'Sgd':
        optimizer = torch.optim.SGD(
            trainable_parameters,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay)
    else:
        print("Wrong Optimizer Type.")
        sys.exit()

    # lr policy ##############################
    # total_iteration = config.nepochs * config.niters_per_epoch
    
    if config.lr_schedule == 'linear':
        lr_policy = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(config.nepochs, 0, config.decay_epoch).step)
    elif config.lr_schedule == 'exponential':
        lr_policy = torch.optim.lr_scheduler.ExponentialLR(optimizer, config.lr_decay)
    elif config.lr_schedule == 'multistep':
        lr_policy = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)
    elif config.lr_schedule == 'cosine':
        lr_policy = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(config.nepochs), eta_min=config.learning_rate_min)
    else:
        print("Wrong Learning Rate Schedule Type.")
        sys.exit()

    cudnn.benchmark = True


    # if use multi machines, the pretrained weight and arch need to be duplicated on all the machines
    if type(pretrain) == str and os.path.exists(pretrain + "/weights_best.pt"):
        pretrained_model = torch.load(pretrain + "/weights_best.pt")
        partial = pretrained_model['state_dict']

        state = model.state_dict()
        pretrained_dict = {k: v for k, v in partial.items() if k in state and state[k].size() == partial[k].size()}
        state.update(pretrained_dict)
        model.load_state_dict(state)

        # TODO this code need be proved
        optimizer.load_state_dict(pretrained_model['optimizer'])
        lr_policy.load_state_dict(pretrained_model['lr_scheduler'])
        start_epoch = pretrained_model['epoch'] + 1
        #start_epoch = 0

        best_acc = pretrained_model['acc']
        best_epoch = pretrained_model['epoch']

        print('Resume from Epoch %d. Load pretrained weight.' % start_epoch)

    else:
        start_epoch = 0
        print('No checkpoint. Train from scratch.')


    # data loader ############################
    if 'cifar' in config.dataset:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        if config.dataset == 'cifar10':
            train_data = dset.CIFAR10(root=config.dataset_path, train=True, download=True, transform=transform_train)
            test_data = dset.CIFAR10(root=config.dataset_path, train=False, download=True, transform=transform_test)
        elif config.dataset == 'cifar100':
            train_data = dset.CIFAR100(root=config.dataset_path, train=True, download=True, transform=transform_train)
            test_data = dset.CIFAR100(root=config.dataset_path, train=False, download=True, transform=transform_test)
        else:
            print('Wrong dataset.')
            sys.exit()


    elif config.dataset == 'imagenet':
        if config.dali:            
            crop_size,val_size = 224,256
            pipe = create_dali_pipeline(batch_size=config.batch_size,
                                num_threads=8,
                                device_id=args.local_rank,
                                seed=12 + args.local_rank,
                                data_dir=config.dataset_path+'/train',
                                crop=crop_size,
                                size=val_size,
                                dali_cpu=config.dali_cpu,
                                shard_id=args.local_rank,
                                num_shards=1,
                                is_training=True)
            pipe.build()
            train_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)
    
            pipe = create_dali_pipeline(batch_size=config.batch_size,
                                        num_threads=8,
                                        device_id=args.local_rank,
                                        seed=12 + args.local_rank,
                                        data_dir=config.dataset_path+'/validation',
                                        crop=crop_size,
                                        size=val_size,
                                        dali_cpu=config.dali_cpu,
                                        shard_id=args.local_rank,
                                        num_shards=1,
                                        is_training=False)
            pipe.build()
            test_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)
        else:
            # if config.autoaugment:
            #     train_data = prepare_train_data_autoaugment(dataset=config.dataset,
            #                                   datadir=config.dataset_path+'/train')   
            # else:        
            #     train_data = prepare_train_data(dataset=config.dataset,
            #                                   datadir=config.dataset_path+'/train')
            
            # test_data = prepare_test_data(dataset=config.dataset,
            #                                 datadir=config.dataset_path+'/validation')
            train_data = prepare_train_data_for_search(dataset=config.dataset,
                                          datadir=config.dataset_path+'/train', num_class=config.num_classes)
            test_data = prepare_test_data_for_search(dataset=config.dataset,
                                        datadir=config.dataset_path+'/validation', num_class=config.num_classes)

    else:
        print('Wrong dataset.')
        sys.exit()

    train_sampler = None

    if config.dali == False:
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=config.batch_size, shuffle=(train_sampler is None),
            pin_memory=True, num_workers=config.num_workers, sampler=train_sampler)
    
        test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=config.batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=config.num_workers)

    if config.eval_only:
        acc1,acc5 = infer(0, model, test_loader, logger)
        logging.info('Eval: acc1 = {:.3f},acc5 = {:.3f}'.format(acc1,acc5) )
        sys.exit(0)

    # tbar = tqdm(range(config.nepochs), ncols=80)
    for epoch in range(start_epoch, config.nepochs):
       
        # tbar.set_description("[Epoch %d/%d][train...]" % (epoch + 1, config.nepochs))
        logging.info("[Epoch %d/%d],  lr=%f" % (epoch, config.nepochs, optimizer.param_groups[0]['lr']))

        start_t = time.time()
        train(train_loader, model, optimizer, lr_policy, logger, epoch, config)
        total_t = time.time()-start_t
        logging.info('Consuming {0:.2f}s'.format(total_t))
        torch.cuda.empty_cache()
        lr_policy.step()
        
        eval_epoch = config.eval_epoch

        #validation
        if (epoch+1) % eval_epoch == 0: # eval_epoch=1

            with torch.no_grad():
                acc,acc5 = infer(epoch, model, test_loader, logger)


            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                state = {}
                state['state_dict'] = model.state_dict()
                state['optimizer'] = optimizer.state_dict()
                state['lr_scheduler'] = lr_policy.state_dict()
                state['epoch'] = epoch 
                state['acc'] = acc
                torch.save(state, os.path.join(config.save, 'weights_best.pt'))

            
            logger.add_scalar('acc/val', acc, epoch)
            logging.info("Test Acc1:%.3f,Acc5:%.3f, Best Acc1:%.3f,Best Epoch:%d\n" % (acc,acc5, best_acc,best_epoch))
        if config.dali == True:
            train_loader.reset()
            test_loader.reset()



def train(train_loader, model, optimizer, lr_policy, logger, epoch, config):
    model.train()
    #train_loss, correct,total = 0,0,0
    #lambda_alpha = 0.0002
    #dataloader_model = iter(train_loader)
    for batch_idx, data in enumerate(train_loader):
        
        dstart_time = time.time()
        
        
        if config.dali:
            input = data[0]["data"].cuda(non_blocking=True)
            target = data[0]["label"].squeeze().long().cuda(non_blocking=True)
        else:
            input = data[0].cuda(non_blocking=True)
            target = data[1].cuda(non_blocking=True)
        dend_time = time.time()
        


        comp_st = time.time()
        if config.label_smoothing:
            criterion = loss_label_smoothing
        else:
            criterion = model.module._criterion

        r = np.random.rand(1)
        if config.cutmix and config.beta > 0 and r < config.cutmix_prob:
            # generate mixed sample
            prostart_time = time.time()
            lam = np.random.beta(config.beta, config.beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            # compute output
            logit = model(input)
            loss = criterion(logit, target_a) * lam + criterion(logit, target_b) * (1. - lam)
            del rand_index
        else:
            logit = model(input)
            loss = criterion(logit, target)
        comp_et = time.time()
      
        # l2_alpha = 0.0
        # for name, param in model.named_parameters():
        #     if "scale_coef" in name:
        #         l2_alpha += torch.pow(param, 2)
        # alpha_loss = lambda_alpha * l2_alpha
        # total_loss = loss+alpha_loss
       
        loss.backward()
        #total_loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

 
        if batch_idx % 100 == 0:
            
            #logging.info("[Epoch %d/%d][batch_idx %d/%d] TLoss=%.3f  WLoss=%.3f DataTime=%.3fs ProcessComputTime=%.3fs" % (epoch + 1, config.nepochs, batch_idx + 1, len(train_loader), total_loss.item(),loss.item(),(dend_time-dstart_time), comp_et-comp_st))
            logging.info("[Epoch %d/%d][batch_idx %d/%d]  WLoss=%.3f DataTime=%.3fs ProcessComputTime=%.3fs" % (epoch + 1, config.nepochs, batch_idx + 1, len(train_loader),loss.item(),(dend_time-dstart_time), comp_et-comp_st))
            logger.add_scalar('loss/train', loss, epoch*len(train_loader)+batch_idx)
            #logger.add_scalar('Accuracy/train', 100. * correct / total, epoch * len(train_loader) + batch_idx)
            logger.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch * len(train_loader) + batch_idx)
    
    #logging.info('Train Acc:{0:.3f}, Train loss:{1:.4f}'.format(100.*correct/total,train_loss/len(train_loader)))
    torch.cuda.empty_cache()
    del loss
    del input,target,logit


def loss_label_smoothing(outputs, labels):
    """
    loss function for label smoothing regularization
    """
    alpha = 0.1
    N = outputs.size(0)  # batch_size
    C = outputs.size(1)  # number of classes
    smoothed_labels = torch.full(size=(N, C), fill_value= alpha / (C - 1)).cuda()
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=1-alpha)

    log_prob = torch.nn.functional.log_softmax(outputs, dim=1)
    loss = -torch.sum(log_prob * smoothed_labels) / N

    return loss


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def infer(epoch, model, test_loader, logger):
    model.eval()
    prec1_list ,prec5_list= [],[]
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if config.dali:
                input_var = data[0]["data"].cuda(non_blocking=True)
                target_var = data[0]["label"].squeeze().long().cuda(non_blocking=True)
            else:
              input_var = Variable(data[0]).cuda()
              target_var = Variable(data[1]).cuda()

            output = model(input_var)
            prec1, prec5 = accuracy(output.data, target_var, topk=(1,5))
            prec1_list.append(prec1)
            prec5_list.append(prec5)

        acc  = sum(prec1_list)/len(prec1_list)
        acc5 = sum(prec5_list)/len(prec5_list)
    
     
    del input_var,target_var,output
    del prec1_list,prec5_list
    torch.cuda.empty_cache() 
    return acc,acc5


def reduce_tensor(rt, n):
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save(model, model_path):
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    main() 
