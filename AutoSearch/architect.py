import torch
import numpy as np
import sys
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
from pdb import set_trace as bp



def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, args):

        self.model = model
        self._args = args
        #lwq get the architecture parameters
        self.optimizer = torch.optim.Adam(list(self.model.module._arch_params.values()), lr=args.arch_learning_rate, betas=(0.5, 0.999))#, weight_decay=args.arch_weight_decay)
       
        self.flops_weight = args.flops_weight

        self.flops_decouple = args.flops_decouple

        self.latency_weight = args.latency_weight

        self.mode = args.mode

        self.offset = args.offset

        self.hw_update_cnt = 0
        self.hw_aware_nas = args.hw_aware_nas
        self.hw_update_freq = args.hw_update_freq

        self.Population = args.Populations
        self.Gene = args.Generations
        self.platform = args.platform

        print("architect initialized!")



    def step(self, input_valid, target_valid, criterion, temp=1):
        self.optimizer.zero_grad()
        # #!log

        if self.mode == 'proxy_hard' and self.offset:
            alpha_old = self.model.module._arch_params['alpha'].data.clone()

        if self._args.efficiency_metric == 'flops':
            loss, loss_flops = self._backward_step_flops(input_valid, target_valid, criterion, temp)

        elif self._args.efficiency_metric == 'latency':
            loss, loss_latency,op_index_layerwise= self._backward_step_latency(input_valid, target_valid, temp)
    
        else:
            print('Wrong efficiency metric.')
            sys.exit()

        loss.backward()


        ## decouple the efficiency loss of alpha and beta
        if self._args.efficiency_metric == 'flops':
            if self.flops_weight > 0:
                loss_flops.backward()

        elif self._args.efficiency_metric == 'latency':
            if self.latency_weight > 0:
                loss_latency.backward()
        else:
            print('Unsupport efficiency metric:', self._args.efficiency_metric)
            sys.exit()

        
        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.mode == 'proxy_hard' and self.offset:
            pass
            # alpha_new = self.model.module._arch_params['alpha'].data

            # for i, cell in enumerate(self.model.module.transformer):

            #     offset = torch.log(sum(torch.exp(alpha_old[i][cell.active_list])) / sum(torch.exp(alpha_new[i][cell.active_list])))

            #     for active_op in cell.active_list:
            #         self.model.module._arch_params['alpha'][i][active_op].data += offset.data

        
        if self._args.efficiency_metric == 'latency':
            return loss,loss_latency,op_index_layerwise
        else:
            return loss,loss_flops
 

    ##generate the flops, mutily the flops weight get the flops_weight
    def _backward_step_flops(self, input_valid, target_valid, criterion,temp=1):
        # print('Param on CPU:', [name for name, param in self.model.named_parameters() if param.device.type == 'cpu'])
        # print('Buffer on CPU:', [name for name, param in self.model.named_buffers() if param.device.type == 'cpu'])
        

        logit = self.model(input_valid, temp)
        loss = criterion(logit,target_valid)
        #loss = self.model.module._criterion(logit, target_valid)

        if self.flops_weight > 0:
            flops = self.model.module.forward_flops(temp=temp)
        else:
            flops = 0

        self.flops_supernet = flops
        loss_flops = self.flops_weight * flops

        # print(flops, loss_flops, loss)
        return loss, loss_flops

    #generate the latency, mutily the latency weight get the latency_loss
    def _backward_step_latency(self, input_valid, target_valid, temp=1):
        logit = self.model(input_valid, temp)
        loss = self.model.module._criterion(logit, target_valid) # acc loss

        if self.latency_weight > 0:
        
            if self.hw_aware_nas:#default false
                if self.hw_update_cnt == 0:
                    self.model.module.search_for_hw(Populations=self.Population,Generations=self.Gene,platform=self.platform)
            else:
                #
                if self.hw_update_cnt == 0 or self.hw_update_cnt % self.hw_update_freq == 0:
                    self.model.module.search_for_hw(Populations=self.Population,Generations=self.Gene,platform=self.platform)
            self.hw_update_cnt += 1
            latency,op_indx_layerwise = self.model.module.forward_hw_latency(self.platform)
        else:
            latency = 0
    
        self.latency_supernet = latency
        loss_latency = self.latency_weight * latency

        return loss, loss_latency,op_indx_layerwise