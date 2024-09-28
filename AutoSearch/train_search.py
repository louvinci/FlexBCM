from __future__ import division
import os
import sys
import time
import logging
import numpy as np


import time
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as dset
from torch.autograd import Variable

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

from losses import DistillationLoss
from datasets import prepare_train_data_for_search, prepare_test_data_for_search

from tensorboardX import SummaryWriter

from config_search import config
from architect import Architect
from model_search import SuperNet as Network
from model_infer import SubNet_Infer
from timm.models import create_model

from thop import profile
from operations import count_BCMConv
from circulant_2d import BCM_Conv2d_fft
custom_ops = {BCM_Conv2d_fft: count_BCMConv}

from lr import LambdaLR
import argparse
parser = argparse.ArgumentParser(description='AMB-CNN')
parser.add_argument('--dataset_path', type=str, default=None,
                    help='path to ImageNet-100')
parser.add_argument('-b', '--batch_size', type=int, default=None,
                    help='batch size')
parser.add_argument('--num_workers', type=int, default=None,
                    help='number of workers')
parser.add_argument('--flops_weight', type=float, default=None,
                    help='weight of FLOPs loss')
parser.add_argument('--gpu', nargs='+', type=int, default=None,
                    help='specify gpus')
parser.add_argument('--seed', type=int, default=12345,
                    help='random seed')


args = parser.parse_args()


np.random.seed(args.seed)
torch.manual_seed(args.seed)

def save_files(fdir,fname,data):
    with open(fdir+fname,"a+") as f:
        f.write(data)
        f.write('\n')

def main():
    if args.dataset_path is not None:
        config.dataset_path = args.dataset_path
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.flops_weight is not None:
        config.flops_weight = args.flops_weight

    
    
    # Simply call main_worker function
    main_worker(config)


def main_worker(config):
    pretrain = config.pretrain


    if type(pretrain) == str:
        config.save = pretrain
    else:
        config.save = 'ckpt/{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))

    logger = SummaryWriter(config.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))# keep the config info
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info("args = %s", str(config))


    #* lwq return the SuperNet
    model = Network(config=config)
    
    model = torch.nn.DataParallel(model).cuda()# default add 'module after the model.'

    
    #* Train the weight parameters of the SuperNet, excluding the architecture parameters
    model_weight_parameters = []
    model_weight_parameters += list(model.module.stem.parameters())
    model_weight_parameters += list(model.module.layer1.parameters())
    model_weight_parameters += list(model.module.layer2.parameters())    
    model_weight_parameters += list(model.module.layer3.parameters())    
    model_weight_parameters += list(model.module.layer4.parameters())        
    model_weight_parameters += list(model.module.linear.parameters())

    
    if config.opt == 'Adam':
        optimizer = torch.optim.Adam(
            model_weight_parameters,
            lr=config.lr,
            betas=config.betas)
    elif config.opt == 'Sgd':
        optimizer = torch.optim.SGD(
            model_weight_parameters,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay)
    else:
        print("Unsupport Optimizer Type.")
        sys.exit()
        
    #*train the architect parameters
    architect = Architect(model, config)
    
    # lr policy ##############################
    total_iteration = config.nepochs
    
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


    # if use multi machines, the pretrained weight and arch need to be duplicated on all the machines
    if type(pretrain) == str and os.path.exists(pretrain + "/weights_latest.pt"):
        
        partial = torch.load(pretrain + "/weights_latest.pt")
        state = model.state_dict()
        pretrained_dict = {k: v for k, v in partial.items() if k in state and state[k].size() == partial[k].size()}
        state.update(pretrained_dict)
        model.load_state_dict(state)

        pretrain_arch = torch.load(pretrain + "/arch_checkpoint.pt")
        
        model.module.alpha.data = pretrain_arch['alpha'].data
        start_epoch = pretrain_arch['epoch']

        optimizer.load_state_dict(pretrain_arch['optimizer'])
        lr_policy.load_state_dict(pretrain_arch['lr_scheduler'])
        architect.optimizer.load_state_dict(pretrain_arch['arch_optimizer'])

        logging.info('Resume from Epoch %d. Load pretrained weight and arch.' % start_epoch)
    else:
        start_epoch = 0
        logging.info('No checkpoint. Search from scratch.')


    # # data loader ###########################
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


    elif 'imagenet' in config.dataset:
        train_data = prepare_train_data_for_search(dataset=config.dataset,
                                          datadir=config.dataset_path+'/train', num_class=config.num_classes,is_random=True)
        test_data = prepare_test_data_for_search(dataset=config.dataset,
                                        datadir=config.dataset_path+'/validation', num_class=config.num_classes,is_random=True)

    else:
        print('Wrong dataset.')
        sys.exit()

    criterion = nn.CrossEntropyLoss()
    num_train = len(train_data)# train data numbers
    indices = list(range(num_train))
    split = int(np.floor(config.train_portion * num_train))

    
    train_sampler_model = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    train_sampler_arch = torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train])

    train_loader_model = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size, 
        sampler=train_sampler_model, shuffle=(train_sampler_model is None),
        pin_memory=False, num_workers=config.num_workers, drop_last=True)

    
    train_loader_arch = torch.utils.data.DataLoader(
            train_data, batch_size=config.batch_size,
            sampler=train_sampler_arch, shuffle=(train_sampler_arch is None),
            pin_memory=False, num_workers=config.num_workers, drop_last=True)

    test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=config.batch_size,
                                          shuffle=False,
                                          num_workers=config.num_workers)

    best_acc = 0
    bt = time.time()
    for epoch in range(start_epoch, config.nepochs):

        #*lwq Enable the pretrain_epoch(30) with different mode.The Default setting for both is "proxy_hard"
        if epoch < config.pretrain_epoch:
            update_arch = False
        else:
            update_arch = True


        temp = config.temp_init * config.temp_decay ** epoch
        
        logging.info("Temperature: " + str(temp))
        logging.info("[Epoch %d/%d] lr=%f" % (epoch, config.nepochs, optimizer.param_groups[0]['lr']))
        logging.info("update arch: " + str(update_arch))


    
        train_iterwise(train_loader_model, train_loader_arch, model, architect, optimizer, criterion, logger, epoch, 
                    update_arch=update_arch, temp=temp, arch_update_frec=config.arch_update_frec)
   

        lr_policy.step()
        # if update_arch:
        #     architect.lr_policy.step()
        torch.cuda.empty_cache()

        # validation, except the 0-th epoch
        alpha_log = model.module._arch_params['alpha'].data.clone()
        alpha_log = alpha_log.softmax(-1)
        save_alpha('alpha.txt',alpha_log)

        if not (epoch+1) % config.eval_epoch:
            
            if pretrain == True:
                acc = infer(model, test_loader,hw_gen=False)
                logger.add_scalar('acc/val', acc, epoch)
                logging.info("Epoch %d: acc %.3f"%(epoch, acc))

            else:
                if (epoch == start_epoch) or (update_arch==True):
                    acc, metric = infer( model, test_loader,hw_gen=True)
                else:
                    # use the metric of the first time
                    acc = infer( model, test_loader,hw_gen=False)

                logger.add_scalar('acc/val', acc, epoch)
                logging.info("Epoch %d: acc %.3f"%(epoch, acc))

                state = {}

                if acc > best_acc:
                    best_acc = acc
                    state['alpha'] = getattr(model.module, 'alpha')
                    state['acc'] = acc
                    state['epoch'] = epoch
                    torch.save(state, os.path.join(config.save, "arch.pt"))
                    state['optimizer'] = optimizer.state_dict()
                    state['lr_scheduler'] = lr_policy.state_dict()
                    state['arch_optimizer'] = architect.optimizer.state_dict()
                    torch.save(state, os.path.join(config.save, "arch_checkpoint.pt"))
                    save(model, os.path.join(config.save, 'weights_latest.pt'))

                if metric!= None:
                    if config.efficiency_metric == 'flops':
                        logger.add_scalar('flops/val', metric, epoch)
                        logging.info("Epoch %d: FLOPs %.3f"%(epoch, metric))
                        
                        #*adjusting the flops_weight
                        if config.flops_weight > 0 and update_arch:
                            if metric < config.flops_min:
                                architect.flops_weight /= 2
                            elif metric > config.flops_max:
                                architect.flops_weight *= 2

                            logger.add_scalar("arch/flops_weight", architect.flops_weight, epoch+1)
                            logging.info("arch_flops_weight = " + str(architect.flops_weight))

                    elif config.efficiency_metric == 'latency':
                        logger.add_scalar('fps/val', metric, epoch)
                        logging.info("Epoch %d: FPS %.3f"%(epoch, metric))

                        if config.latency_weight > 0 and update_arch:
                            if metric < config.fps_min:
                                architect.latency_weight *= 2
                            elif metric > config.fps_max:
                                architect.latency_weight /= 2
    
                            logger.add_scalar("arch/latency_weight", architect.latency_weight, epoch+1)
                            logging.info("arch_latency_weight = " + str(architect.latency_weight))


    if config.efficiency_metric == 'latency':
        #get the searched model.
        best_arch = torch.load(os.path.join(config.save, "arch.pt"))
    
        b_alpha = best_arch['alpha'].data

        model_infer = SubNet_Infer(b_alpha, config=config)
    
        fps, searched_hw = model_infer.eval_latency(config.backbone, Populations = config.Populations,Generations= config.Generations,platform = config.platform,
                                                        hardware=model.module.searched_hw if config.hw_aware_nas else None)

        opt_hw_final = {'opt_hw': searched_hw, 'FPS': fps}
        torch.save(opt_hw_final, os.path.join(config.save, "opt_hw.pt"))

        flops, params = profile(model_infer, inputs=(torch.randn(1, 3, config.image_height, config.image_width),), custom_ops=custom_ops)
        
        logging.info("params = %fM, FLOPs = %fM", params / 1e6, flops / 1e6)
        logging.info("FPS of Final Arch: %f", fps)
    logging.info("Total Search Time: {:.3f}h".format( (time.time()-bt)/3600 ))



def train_iterwise(train_loader_model, train_loader_arch, model, architect, optimizer, criterion, logger, epoch, update_arch=True, temp=1, arch_update_frec=1):
    model.train()

    # bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    # pbar = tqdm(range(len(train_loader_model)), file=sys.stdout, bar_format=bar_format, ncols=80)
    dataloader_model = iter(train_loader_model)
    dataloader_arch = iter(train_loader_arch)
    # cifar10 default len(train_loader_model) =390 *64 = 25000, equally divided 

    for step in range(len(train_loader_model)):
        start_time = time.time()
        optimizer.zero_grad()

        input, target = dataloader_model.next()

        data_time = time.time() - start_time

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)


        if update_arch and step % arch_update_frec == 0:

            try:
                input_search, target_search = dataloader_arch.next()
            except:
                dataloader_arch = iter(train_loader_arch)
                input_search, target_search = dataloader_arch.next()

            input_search = input_search.cuda(non_blocking=True)
            target_search = target_search.cuda(non_blocking=True)

            
            #* architect loss, including hardware information
            if config.efficiency_metric == 'latency':
                loss_arch,loss_proxy, op_index_layerwise= architect.step(input_search, target_search, criterion,temp=temp)
            else:
                loss_arch,loss_proxy = architect.step(input_search, target_search, criterion,temp=temp)

            if (step+1) % 25 == 0:
                
                alpha = model.module._arch_params['alpha'].data.clone()
                
                alpha0 = alpha[0].argmax(-1)
                alpha2 = alpha[2].argmax(-1)
                alpha3 = alpha[3].argmax(-1)
                alpha6 = alpha[6].argmax(-1)
                        
                logger.add_scalar('arch/alpha0', alpha0, epoch*len(train_loader_arch)+step)
                logger.add_scalar('arch/alpha2', alpha2, epoch*len(train_loader_arch)+step)
                logger.add_scalar('arch/alpha3', alpha3, epoch*len(train_loader_arch)+step)
                logger.add_scalar('arch/alpha6', alpha6, epoch*len(train_loader_arch)+step)
                if config.efficiency_metric == 'flops':
                    logger.add_scalar('arch/flops_subnet', architect.flops_supernet, epoch*len(train_loader_arch)+step)
                elif config.efficiency_metric == 'latency':
                    logger.add_scalar('arch/latency_subnet', architect.latency_supernet, epoch*len(train_loader_arch)+step)

        outputs = model(input, temp)
        
        #loss = model.module._criterion(outputs, target)
        loss = criterion(outputs,target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()# here  the optimizer doesn't update the alpha parameters.

        total_time = time.time() - start_time

        if (step+1) % 25 == 0:
            logging.info("[Epoch %d/%d][Step %d/%d] WTLoss=%.3f Time=%.3f Data Time=%.3f" % 
                        (epoch + 1, config.nepochs, step + 1, len(train_loader_model), loss.item(), total_time, data_time))
            logger.add_scalar('loss_weight/train', loss, epoch*len(train_loader_model)+step)


    torch.cuda.empty_cache()
    del loss
    if update_arch: del loss_arch




# default finalize is ture all the time
def infer(model, test_loader, hw_gen=False):
    model.eval()
    prec1_list = []

    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            input_var = Variable(input).cuda()
            target_var = Variable(target).cuda()

            output = model(input_var)
            prec1, = accuracy(output.data, target_var, topk=(1,))
            prec1_list.append(prec1)

        acc = sum(prec1_list)/len(prec1_list)

    if hw_gen:
        model_infer = SubNet_Infer(getattr(model.module, 'alpha'), config=config)
        if config.efficiency_metric == 'flops':
            flops = model_infer.forward_flops()
            return acc, flops

        elif config.efficiency_metric == 'latency':
            fps, searched_hw = model_infer.eval_latency(config.backbone, Populations = config.Populations,Generations= config.Generations,platform = config.platform,
                                                        hardware=model.module.searched_hw if config.hw_aware_nas else None)
            return acc, fps
    else:
        return acc


def reduce_tensor(rt, n):
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_alpha(dir,arr):
    with open(dir,'a+') as f:
        f.write("alpha log:\n")
        np.savetxt(f,arr.detach().cpu().numpy(),fmt='%.3f')

def save(model, model_path):
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    main() 
