import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
#neural net used in L2HMC
class Swish(nn.Module):
    def __init__(self,Lipswish=False):
        super(Swish,self).__init__()
        self.Lipswish = Lipswish
        pass
    
    def forward(self,in_vect):
        out = in_vect*torch.sigmoid(in_vect)
        out = out/1.1 if self.Lipswish else out
        return out
    
def gen_mask(Nside,mask_type):
    #generate checkerboard mask for x with size Nside*Nside and 3 channels
    if mask_type=='conv':
        lin_vect = np.linspace(0,Nside*(Nside+1)-1,Nside*(Nside+1))
        bi_mask = (lin_vect%2 == 1).reshape(Nside,Nside+1)[:,:-1]
        bi_mask_conj = ~bi_mask
        x_mask = np.stack([bi_mask,bi_mask_conj,bi_mask],axis=0).reshape(1,3,Nside,Nside)
        x_mask_conj = ~x_mask
    elif mask_type=='MLP':
        lin_vect = np.linspace(0,1,Nside)
        x_mask = np.random.permutation(lin_vect>0.5).reshape(1,Nside)
        x_mask_conj = ~x_mask
    
    return torch.from_numpy(x_mask), torch.from_numpy(x_mask_conj)

def batch_forward(module_list,x,i):
    # this function avoid evaluating the entire module in series multiple times, should save a bit of time...
    # i should be a number of a vector of indexes from 0, if a vector it should correspond to x in batch direction
    if not torch.is_tensor(i):
        out = module_list[i](x)
    else:
        out_list, order_list = [], []
        for j in range(int(i.max()+1)):
            indx = (i == j)
            out_list.append(module_list[j](x[indx]))
            order_list.append(torch.nonzero(indx,as_tuple=True)[0])
        out = torch.cat(out_list,dim=0)[torch.argsort(torch.cat(order_list,dim=0))] 
    return out

def get_act(act):
    if act=='relu':
        af = nn.ReLU(inplace=True)
    elif act=='elu':
        af = nn.ELU(inplace=True)
    elif act=='leakyrelu':
        af = nn.LeakyReLU(0.01,inplace=True)
    elif act=='swish':
        af = Swish()
    return af
    
class MeanPoolConv3x3(nn.Module):
    #block for minimal down sampling.
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel,out_channel,3,stride=1,padding=1,bias=False)
    
    def forward(self,inputs):
        inputs = F.avg_pool2d(inputs,2)
        return self.conv(inputs)

class Conv_SQT(nn.Module):
    def __init__(self,in_chan,dim,out_layers,in_layers,mode='SQT',act ='relu',N_side=32):
        super().__init__()
        #small Unet with convolution architecture
        #in_chan: channels in the image data,input will be 2*in_chan, out channel will 3*in_chan
        #dim: width of intemediate layers
        #max_steps: maximum number of steps of leapfrog integration
                #   will create max_steps number of output layers
        self.af = get_act(act)
        
        self.mode = mode 
        if self.mode == 'SQT':
            in_stacks, self.out_stacks = 2, 3
        elif self.mode == 'SQT_X':
            in_stacks, self.out_stacks = 1, 3
        elif self.mode == 'SQT_RW':
            in_stacks, self.out_stacks = 3, 3 
        elif self.mode == 'R':
            in_stacks, self.out_stacks = 2, 1
            
        self.conv1_list =nn.ModuleList([nn.Conv2d(in_stacks*in_chan,dim,3,1,1) for i in range(in_layers)]) #32x32xdim
        self.conv2 = MeanPoolConv3x3(dim,2*dim) #16x16x2*dim
        
        self.conv_mid1 = nn.Conv2d(2*dim,2*dim,5,1,2)
        self.conv_mid2 = nn.Conv2d(2*dim,2*dim,5,1,2)
        
        self.convup2 = nn.ConvTranspose2d(2*dim,dim,3,2,1,output_padding=1) #32x32xdim
        #condition on step number with different output layer modules..
        self.convup1_list = nn.ModuleList([nn.ConvTranspose2d(dim,self.out_stacks*in_chan,3,1,1,output_padding=0) for i in range(out_layers)])        
        for m in self.convup1_list:
            m.weight.data.zero_()
            #m.bias.data.zero_()
        #self.g_bias = nn.Parameter(torch.zeros((1,self.out_stacks*in_chan,N_side,N_side)))
        
    def forward(self,in_list,o,i):
        #x, v are the 2 variables input, v can be gradE sometimes
        #i is the step number...
        x = torch.cat(in_list,dim=1)
        out = self.af(batch_forward(self.conv1_list,x,i))
        out = self.af(self.conv2(out))
        out = self.af(self.conv_mid1(out))
        out = self.af(self.conv_mid2(out))
        if self.mode == 'R':
            out = self.af(self.convup2(out))
        else:
            out = torch.tanh(self.convup2(out))
        
        out = batch_forward(self.convup1_list,out,o)
        #out = out + self.g_bias
                                   
        if self.out_stacks>1:
            out_list = list(out.chunk(self.out_stacks,dim=1))
        else:
            out_list = out
        
        return out_list    
    
    
class Conv_U_SQT(nn.Module):
    def __init__(self,in_chan,dim,out_layers,in_layers,mode='SQT',act ='relu',N_side=32):
        super().__init__()
        #small Unet with convolution architecture
        #in_chan: channels in the image data,input will be 2*in_chan, out channel will 3*in_chan
        #dim: width of intemediate layers
        #max_steps: maximum number of steps of leapfrog integration
                #   will create max_steps number of output layers
        self.af = get_act(act)
        
        self.mode = mode 
        if self.mode == 'SQT':
            in_stacks, self.out_stacks = 2, 3
        elif self.mode == 'SQT_X':
            in_stacks, self.out_stacks = 1, 3
        elif self.mode == 'SQT_RW':
            in_stacks, self.out_stacks = 3, 3 
        elif self.mode == 'R':
            in_stacks, self.out_stacks = 2, 1
            
        
        
        self.conv1_list =nn.ModuleList([nn.Conv2d(in_stacks*in_chan,dim,3,1,1) for i in range(in_layers)]) #32x32xdim
        self.conv2 = MeanPoolConv3x3(dim,2*dim) #16x16x2*dim
        self.conv3 = MeanPoolConv3x3(2*dim,4*dim) #8x8x4*dim
        self.conv4 = MeanPoolConv3x3(4*dim,8*dim) #4x4x8*dim
        
        self.convup4 = nn.ConvTranspose2d(8*dim,4*dim,3,2,1,output_padding=1) #8x8x4*dim
        self.convup3 = nn.ConvTranspose2d(8*dim,2*dim,3,2,1,output_padding=1) #16x16x2*dim
        self.convup2 = nn.ConvTranspose2d(4*dim,dim,3,2,1,output_padding=1) #32x32xdim
        
        #condition on step number with different output layer modules...
        #32x32xout_chan
        self.convup1_list = nn.ModuleList([nn.ConvTranspose2d(2*dim,self.out_stacks*in_chan,3,1,1,output_padding=0) for i in range(out_layers)])        
        for m in self.convup1_list:
            m.weight.data.zero_()
            #m.bias.data.zero_()
        
    def _compute_up(self,net,x,y):
        x_in = torch.cat([x,y],dim=1)
        return net(x_in)
    
    def forward(self,in_list,o,i):
        #x, v are the 2 variables input, v can be gradE sometimes
        #i is the step number...
        x = torch.cat(in_list,dim=1)
        out1 = self.af(batch_forward(self.conv1_list,x,i))
        out2 = self.af(self.conv2(out1))
        out3 = self.af(self.conv3(out2))
        out4 = self.af(self.conv4(out3))
        
        out = self.af(self.convup4(out4))
        out = self.af(self._compute_up(self.convup3,out,out3))
        if self.mode == 'R':
            out = self.af(self._compute_up(self.convup2,out,out2))
        else:
            out = torch.tanh(self._compute_up(self.convup2,out,out2))
        
        out = batch_forward(self.convup1_list,torch.cat([out,out1],dim=1),o)
        
        if self.out_stacks>1:
            out_list = list(out.chunk(self.out_stacks,dim=1))
        else:
            out_list = out
        
        return out_list
    
class MLP_SQT(nn.Module):
    def __init__(self,in_dim,dim,out_layers,in_layers,mode='SQT',act='relu'):
        super().__init__()
        #small 4 layer MLP for low dimentional experiments...no U net architecture
        #in_dim: data should be [batchSize,in_dim], with no other dimensions
        #dim: width of intemediate layers
        #max_steps: maximum number of steps of leapfrog integration
        #mode choice: 'SQT', 'SQ', 'SQ_single'
    
        self.af = get_act(act)
        self.mode = mode 
        if self.mode == 'SQT':
            in_stacks, self.out_stacks = 2, 3
        elif self.mode == 'SQT_X':
            in_stacks, self.out_stacks = 1, 3
        elif self.mode == 'SQT_RW':
            in_stacks, self.out_stacks = 3, 3 
        elif self.mode == 'R':
            in_stacks, self.out_stacks = 2, 1
        #self.deep = deep
        
        self.L1_list= nn.ModuleList([nn.Linear(in_stacks*in_dim,dim) for i in range(in_layers)])
        self.L2 = nn.Linear(dim,dim)
        self.L3 = nn.Linear(dim,dim)
        self.L4 = nn.Linear(dim,dim)
        self.L5_list= nn.ModuleList([nn.Linear(dim,self.out_stacks*in_dim) for i in range(out_layers)])
        
        for m in self.L5_list:
            m.weight.data.zero_()
            m.bias.data.zero_()
    
    def forward(self,in_list,o,i):
        #i switch input layers while o switch output layer
        x = torch.cat(in_list,dim=1)
        x1 = self.af(batch_forward(self.L1_list,x,i))
        x = self.af(self.L2(x1))
        x = self.af(self.L3(x))
        #x = self.af(self.L4(x))
        if self.mode == 'R':
            x = self.af(self.L4(x))
        else:
            x = torch.tanh(self.L4(x))
            
        x = batch_forward(self.L5_list,x,o)
        
        if self.out_stacks>1:
            out_list = list(x.chunk(self.out_stacks,dim=1)) #list contain SQT or ST
            return out_list
        else:
            return x

class L2GradRW(nn.Module):
    def __init__(self,netE,in_chan,dim,N_steps,epsi=0.05,net_type='MLP',grad_mode='R',bias=True,act='relu',N_side=32):
        super().__init__()
        #this module performs random walk proposal utilizing gradient of data distribution.
        #Should give true tractable training of a MCMC sampler
        self.netE = netE
        self.N_steps = N_steps
        self.net_type = net_type #MLP for toy tasks and unstructrued tasks, conv for images
        self.grad_mode = grad_mode #if to use gradient of data distribution, 'no', 'x', 'R'
        self.bias = bias
        self.epsi = nn.Parameter(torch.tensor([epsi]).log())
        
        if net_type == 'MLP':
            self.v_size = in_chan
            self.netV = MLP_SQT(in_chan,dim,N_steps+1,N_steps+1,mode='SQT_RW',act=act)
            if grad_mode=='R':
                self.netR = MLP_SQT(in_chan,dim,N_steps,N_steps,mode='R',act=act)
            #series of random mask for MLP
            v_mask_list, v_mask_conj_list = [],[]
            for i in range(N_steps):
                v_mask, v_mask_conj = gen_mask(in_chan,'MLP')
                v_mask_list.append(v_mask)
                v_mask_conj_list.append(v_mask_conj)
            v_mask, v_mask_conj = torch.cat(v_mask_list,dim=0), torch.cat(v_mask_conj_list,dim=0)
            self.register_buffer('v_mask',v_mask)
            self.register_buffer('v_mask_conj',v_mask_conj)
        elif net_type == 'conv':
            self.v_size = in_chan*N_side*N_side
            self.netV = Conv_SQT(in_chan,dim,N_steps+1,N_steps+1,mode='SQT_RW',N_side=N_side)
            if grad_mode=='R':
                self.netR = Conv_SQT(in_chan,dim,N_steps,N_steps,mode='R',act=act,N_side=N_side)
            #fixed checkerboard mask for conv
            v_mask, v_mask_conj = gen_mask(N_side,'conv')
            self.register_buffer('v_mask',v_mask)
            self.register_buffer('v_mask_conj',v_mask_conj)
    
    def forward(self,x,v,reverse=False,TE=1,grad=True):
        #forward direction is sampling from noise to v vector, which will be added to x.
        sldj = 0
        epsi = self.epsi.exp()/(2*self.N_steps)
        v = v.clone()
        
        if not reverse:
            for i in range(self.N_steps):
                v_mask, v_mask_conj = self.get_v_mask(i)
                v_mask, v_mask_conj = v_mask.float(), v_mask_conj.float()
                
                gradE = self.get_grad(x,v_mask*v,i,TE,grad)
                
                S,Q,T = self.netV([x,v_mask*v,gradE],i,i)
                v = v_mask*v + v_mask_conj*(v*S.exp() - epsi*(gradE*Q.exp() + self.bias*T))
                sldj += (v_mask_conj*S).flatten(1).sum(1)
                
                gradE = self.get_grad(x,v_mask_conj*v,i,TE,grad)
                
                S,Q,T = self.netV([x,v_mask_conj*v,gradE],i,i)
                v = v_mask_conj*v + v_mask*(v*S.exp() - epsi*(gradE*Q.exp() + self.bias*T))
                sldj += (v_mask*S).flatten(1).sum(1)
                
            if self.grad_mode=='x':
                i = self.N_steps
                _,Q,T = self.netV([x,torch.zeros_like(v),torch.zeros_like(v)],i,i)
                v -= self.epsi.exp()*self.grad_E(x,T=TE,grad=False)*Q.exp()/2 + T
            
        elif reverse:
            if self.grad_mode=='x':
                i = self.N_steps
                _,Q,T = self.netV([x,torch.zeros_like(v),torch.zeros_like(v)],i,i)
                v += self.epsi.exp()*self.grad_E(x,T=TE,grad=False)*Q.exp()/2 + T
            
            for i in range(self.N_steps,0,-1):
                i -= 1
                v_mask, v_mask_conj = self.get_v_mask(i)
                v_mask, v_mask_conj = v_mask.float(), v_mask_conj.float()
                
                gradE = self.get_grad(x,v_mask_conj*v,i,TE,grad)
                
                S,Q,T = self.netV([x,v_mask_conj*v,gradE],i,i)
                v = v_mask_conj*v + v_mask*((v + epsi*(gradE*Q.exp() + self.bias*T))*(-S).exp())
                sldj += (v_mask*(-S)).flatten(1).sum(1)
                
                gradE = self.get_grad(x,v_mask*v,i,TE,grad)
                
                S,Q,T = self.netV([x,v_mask*v,gradE],i,i)
                v = v_mask*v + v_mask_conj*((v + epsi*(gradE*Q.exp() + self.bias*T))*(-S).exp())
                sldj += (v_mask_conj*(-S)).flatten(1).sum(1)
                
            
        
        if not grad:
            v, sldj = v.detach(), sldj.detach()
        return v, sldj
    
    def sample(self,x,T=1,grad=True):
        #run sampling for each x
        #grad indicate whether to keep activation for training sampler
        eps = torch.randn_like(x)
        
        z, sldjf = self.forward(x,eps,reverse=False,TE=T,grad=grad)
        
        epsi = self.epsi.exp()
        xp = x + epsi*z
        
        epsp, sldjb = self.forward(xp,-z,reverse=True,TE=T,grad=grad)
        
        Ee, Eep = self.Ev(eps), self.Ev(epsp)
        Ex, Exp = self.netE(x)/T, self.netE(xp)/T
        log_a = Ee - Eep + Ex - Exp + sldjf + sldjb #here we can directly calculate log_a the accept probability 
        #dE = Exp - Ex #dE can be useful for optimizing KL for burn in samples
        sldjxx = sldjf + self.v_size*self.epsi 
        #density of accept distribution, kept Ee and Ex to keep number reasonable
        
        return xp, log_a, sldjxx, Ex, Exp
    
    def get_grad(self,x,v,i,T,grad):
        if self.grad_mode=='no' or self.grad_mode=='x':
            gradE = torch.zeros_like(v)
            
        elif self.grad_mode=='R':
            R = self.netR([x,v],i,i)
            gradE = self.grad_E(x+R,T=T,grad=grad)
            
        return gradE
    
    def grad_E(self,x,T=1,grad=True):
        x.requires_grad_()
        E = self.netE(x).sum()/T
        grad = torch.autograd.grad(E,x,create_graph=grad)[0]
        return grad
    
    def Ev(self,v):
        return v.pow(2).flatten(1).sum(1)/2
    
    def get_v_mask(self,i):
        #get mask for v updates, only take step number as input
        if self.net_type=='conv':
            v_mask,v_mask_conj = self.v_mask, self.v_mask_conj
        elif self.net_type=='MLP':
            v_mask,v_mask_conj = self.v_mask[i:i+1], self.v_mask_conj[i:i+1]
            
        return v_mask, v_mask_conj
    
    def get_para(self):
        para_list = []
        for p in self.netV.parameters():
            para_list.append(p)
        if self.grad_mode=='R':
            for p in self.netR.parameters():
                para_list.append(p)   
        para_list.append(self.epsi)
        
        return para_list
    
    
class CovMALA(nn.Module):
    def __init__(self,netE,in_chan,epsi=0.05,net_type='MLP',full_cov=True,N_side=32):
        super().__init__()
        #full covariance MALA, full_cov will use full trangular covariance matrix, False will use element-wise variance.
        self.netE = netE
        self.net_type = net_type
        self.full_cov = full_cov
        self.register_buffer('epsi',torch.tensor(epsi))
        self.in_chan = in_chan
        self.N_side = N_side
        
        if self.net_type == 'MLP':
            dim = in_chan
        else:
            dim = in_chan*N_side*N_side #matrix size for conv
        self.dim = dim
        if full_cov:
            L = torch.diag(torch.zeros(dim)).view(1,dim,dim) #I think it will be advantageous to parameterize diagonal element of L in log space?
            mask = torch.from_numpy(np.triu(np.ones((dim,dim)))==1).bitwise_not_() # mask for lower trangular part
            self.register_buffer('mask',mask) 
            self.L_diag = nn.Parameter(torch.zeros(dim))
        else:
            L = torch.zeros(dim).view(dim,1) #paramterize L in log space?
        
        self.L = nn.Parameter(L)
        
    def sample(self,x,T=1,grad=True):
        #Its more efficient to write this as just the sampler instead of the forward and backward form...
        x = self.reshape_x(x,flatten=True)
        z0 = torch.randn_like(x)
        
        gradE = self.reshape_x(self.grad_E(self.reshape_x(x,flatten=False),T=T,grad=False),flatten=True)
        
        if self.full_cov:
            z = - (self.epsi/2)*torch.matmul(self.get_L(transpose=True),gradE) + z0
    
            xp = x + self.epsi*torch.matmul(self.get_L(transpose=False),z)
        else:
            #print(z0.shape)
            z = - (self.epsi/2)*self.L.exp()*gradE + z0
            
            xp = x + self.epsi*self.L.exp()*z
            
        gradEp = self.reshape_x(self.grad_E(self.reshape_x(xp,flatten=False),T=T,grad=False),flatten=True)
        
        if self.full_cov:
            z0p = (self.epsi/2)*torch.matmul(self.get_L(transpose=True),gradE + gradEp) - z0
        else:
            z0p = (self.epsi/2)*self.L.exp()*(gradE + gradEp) - z0
            
        Ex = self.netE(self.reshape_x(x,flatten=False))/T
        Exp = self.netE(self.reshape_x(xp,flatten=False))/T
        
        Ez0 = self.Ev(z0)
        Ez0p = self.Ev(z0p)
        
        if self.full_cov:
            sldjxx = (self.L_diag + self.epsi.log()).sum().expand(x.shape[0])
        else:
            sldjxx = (self.L + self.epsi.log()).sum().expand(x.shape[0])
        log_a = Ez0 - Ez0p + Ex - Exp
        
        xp = self.reshape_x(xp,flatten=False)
        
        return xp, log_a, sldjxx, Ex, Exp
    
    def reshape_x(self,x,flatten=True):
        if self.net_type == 'MLP':
            if flatten:
                return x.view(-1,self.in_chan,1)
            else:
                return x.squeeze()
        else:
            if flatten:
                return x.flatten(1).view(-1,self.dim,1)
            else: #this restore the shape of x
                return x.view(-1,self.in_chan,self.N_side,self.N_side)
        
        return 
    
    def get_L(self,transpose=False):
        L = self.L*self.mask + torch.diag(self.L_diag.exp()).view(1,self.dim,self.dim)
        if transpose:
            L = L.transpose_(1,2)
        return L
    
    def grad_E(self,x,T=1,grad=True):
        x.requires_grad_()
        E = self.netE(x).sum()/T
        grad = torch.autograd.grad(E,x,create_graph=grad)[0]
        return grad
    
    def Ev(self,v):
        return v.pow(2).flatten(1).sum(1)/2
    
    def get_para(self):
        if self.full_cov:
            return [self.L,self.L_diag]
        else:
            return [self.L]
        