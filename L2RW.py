import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
#file with L2HMC object that manages sampling and training

def adjust_lr(RW_C,i,warmup=True,min_lr=1e-3):
    #min_lr is the fraction of lr 
    warmup_steps = 100
    lr = RW_C.lr
    
    if warmup:
        if i < warmup_steps:
            lr = lr*i/warmup_steps
    lr = lr*((1 + np.cos(np.pi * (i - warmup_steps) / (RW_C.N_steps - warmup_steps))) / 2 + min_lr) 
    
    for param_group in RW_C.optimizer.param_groups:
        param_group['lr'] = lr

def clip_grad(optimizer,grad_norm,norm_type):
    for group in optimizer.param_groups:
        nn.utils.clip_grad_norm_(group['params'], grad_norm,norm_type)

class L2GradRW_M(object):
    def __init__(self,RW,beta=1e-3,beta_max=1e3,rho_beta=0.03,adapt_beta=True,a_target=0.9,
                 lr=1e-3,momo=0.9,rms_momo=0.999,grad_norm=10,N_steps=10000,lrmin=1e-5):
        #high-level object, agnostic to shape of x and v
        #controls flow of sampling and training process...
        self.RW = RW #learn to leapfrog module, contains netE and netX, netV
#         V_params = {"lr": lrv, "betas": (0.5, 0.95),'eps': 5e-2}
#         x_params = {"lr": lrx, "betas": (0.5, 0.95),'eps': 5e-2}
        self.lr = lr
        self.N_steps = N_steps
        Adam_params = {"lr": lr, "betas": (momo, rms_momo)}
        self.optimizer = torch.optim.Adam(self.RW.get_para(),**Adam_params,amsgrad=True)
        #self.optimizer = torch.optim.SGD(self.RW.get_para(),lr=lr)
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,N_steps,eta_min=lrmin)
        
        self.a_target = a_target
        self.beta = beta #initialize beta parameter
        self.beta_min = beta #minimal beta value since accept rate takes a while to recover...
        self.beta_max = beta_max #maximum beta
        self.adapt_beta = adapt_beta
        self.rho_beta = rho_beta #beta adaptation rate coefficient
        self.grad_norm = grad_norm #gradient clipping range...
        
    def sample(self,x,Nsteps,T=1,grad=True,print_log=False):
        #x, v has shape of image, d has length bs
        #train_every: number of steps per leapfrog training step
        #E_last = self.E(x,v).detach()
        E_list = []
        for i in range(Nsteps):
            #proposal
            xp = x.clone()
            xp, log_a, sldjxx, Ex, Exp = self.RW.sample(xp,T=T,grad=grad)
            #accept reject
            
            accept_vect = torch.rand_like(log_a) < log_a.clamp(max=0).exp()
            x[accept_vect] = xp[accept_vect].detach()
            Ex, Exp = Ex.detach(), Exp.detach()
            Ex[accept_vect] = Exp[accept_vect].detach()
            E_list.append(Ex.detach()) 
            
            if print_log:
                print('accept {}'.format(accept_vect.float().mean()))

        
        return x.detach(), xp, accept_vect, E_list
    
    def sample_train(self,x,T=1,train=True,print_log=False):
        #train with external supplied samples
        #T: temperature to apply to the energy function
        xp, log_a, sldjxx, Ex, Exp = self.RW.sample(x,T=T,grad=True)
        
        log_a_clip = log_a.clamp(max=0)
        a_avg = log_a_clip.exp().mean().detach()
        L2_jump = (xp-x).pow(2).flatten(1).mean(1).detach()
            
        beta = self.beta
        
        accept_vect =  torch.rand_like(log_a_clip) < log_a_clip.exp()
        
        L2_jump_expected = L2_jump*accept_vect.float()
        
        Loss= - log_a_clip - beta*sldjxx #standard entropy regulated loss for sampling equilibrium distribution
        
        if train:
            if self.adapt_beta:
                self.beta = self.beta*(1+self.rho_beta*(a_avg-self.a_target))
                self.beta = min(max(self.beta,self.beta_min),self.beta_max)
            
            self.optimizer.zero_grad()
            Loss.mean().backward()
            clip_grad(self.optimizer,self.grad_norm,2)
            self.optimizer.step()
            #self.scheduler.step()
        
        if print_log:
            print('jump {:.3e}, Exp jump{:.3e}, accept {:.3f},log_a {:.3e}, sldjxx {:.3f},beta {:.3e}, epsi {:.3e}'.format(L2_jump.mean().item(),L2_jump_expected.mean().item(),a_avg.item(),log_a.mean().item(), sldjxx.mean().item(),self.beta,self.RW.epsi.exp().item()))
        
        xp = xp.clone()
        x = x.clone()
        x[accept_vect] = xp[accept_vect].detach() #burn in samples don't go through MH step.
        
        return x, a_avg, sldjxx.mean().item(), L2_jump.mean().item()
        
