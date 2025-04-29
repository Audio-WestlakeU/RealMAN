import torch
import torch.nn as nn
import numpy as np

class FNblock(nn.Module):
    """ 
    """
    def __init__(self, input_size, hidden_size=128, dropout=0.2, is_online=False, is_first=False):
        """the block of full-band and narrow-band fusion
        """
        super(FNblock, self).__init__()
        self.input_size = input_size
        self.full_hidden_size =  hidden_size // 2
        self.is_first = is_first
        self.is_online = is_online
        if self.is_online:
            self.narr_hidden_size = hidden_size
        else:
            self.narr_hidden_size = hidden_size  // 2
        self.dropout = dropout
        self.dropout_full =  nn.Dropout(p=self.dropout)
        self.dropout_narr = nn.Dropout(p=self.dropout)
        if is_first:
            self.fullLstm = nn.LSTM(input_size=self.input_size, hidden_size=self.full_hidden_size, batch_first=True, bidirectional=True)
        else:
             self.fullLstm = nn.LSTM(input_size=self.input_size+10, hidden_size=self.full_hidden_size, batch_first=True, bidirectional=True)
        self.narrLstm = nn.LSTM(input_size=2*self.full_hidden_size+10, hidden_size=self.narr_hidden_size, batch_first=True, bidirectional=not self.is_online)       
    def forward(self, x, fb_skip, nb_skip):
            #shape of x: nb,nv,nf,nt
        nb,nt,nf,nc = x.shape
        x = x.reshape(nb*nt,nf,-1)
        # if not self.is_first:
        #     x = torch.cat((x,fb_skip),dim=-1)
        x, _ = self.fullLstm(x)
        x = self.dropout_full(x)
        x = torch.cat((x,fb_skip),dim=-1)
        x = x.view(nb,nt,nf,-1).permute(0,2,1,3).reshape(nb*nf,nt,-1) 
        x, _ = self.narrLstm(x)
        x = self.dropout_narr(x)
        x = torch.cat((x,nb_skip),dim=-1)
        x = x.view(nb,nf,nt,-1).permute(0,2,1,3)
        return x
    
class CnnBlock(nn.Module): 
	""" Function: Basic convolutional block
    """
	# expansion = 1
	def __init__(self, kernel=(3,3), stride=(1,1), padding=(1,1)):
     
		super(CnnBlock, self).__init__()
		self.conv1 = nn.Conv2d(138, 64, kernel_size=kernel, stride=stride, padding=padding, bias=False)
		self.conv2 = nn.Conv2d(64, 32, kernel_size=kernel, stride=stride, padding=padding, bias=False)
		self.conv3 = nn.Conv2d(32, 8, kernel_size=kernel, stride=stride, padding=padding, bias=False)
		self.pooling = nn.AvgPool2d(kernel_size=(1, 5))                
		self.pad = padding
		self.relu = nn.ReLU(inplace=True)
		self.tanh = nn.Tanh()
  
	def forward(self, x):
		out = self.conv1(x)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.relu(out)
		out = self.pooling(out)
		out = self.conv3(out)
		out = self.tanh(out)
		return out


class SingleTinyIPDnet(nn.Module):
    """ 
    """
    def __init__(self,input_size=10,hidden_size=128,is_online=True):
        """the block of full-band and narrow-band fusion
        """
        super(SingleTinyIPDnet, self).__init__()
        self.is_online = is_online
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.block_1 = FNblock(input_size=self.input_size,is_online=False, is_first=True)
        self.block_2 = FNblock(input_size=self.hidden_size,is_online=False, is_first=False)
        self.conv = CnnBlock()
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = x.permute(0,3,2,1)
        nb,nt,nf,nc = x.shape
        fb_skip = x.reshape(nb*nt,nf,nc)
        nb_skip = x.permute(0,2,1,3).reshape(nb*nf,nt,nc)
        x = self.block_1(x,fb_skip=fb_skip,nb_skip=nb_skip)
        x = self.block_2(x,fb_skip=fb_skip, nb_skip=nb_skip)
        nb,nt,nf,nc = x.shape
        x = x.permute(0,3,2,1)
        nt2 = nt//5
        x = self.conv(x).permute(0,3,2,1).reshape(nb,nt2,nf,1,-1).permute(0,1,3,2,4) 
        doa_final = x.reshape(nb,nt2,1,nf*2,-1).permute(0,1,3,4,2)
        return doa_final

if __name__ == "__main__":
    x = torch.randn((1,10,256,200))
    model = SingleTinyIPDnet()
    import time
    ts = time.time()
    y = model(x)
    te = time.time()
    print(model)
    print(y.shape)
    print(te - ts)
    model = model.to('meta')
    x = x.to('meta')
    from torch.utils.flop_counter import FlopCounterMode # requires torch>=2.1.0
    with FlopCounterMode(model, display=False) as fcm:
        y = model(x)
        flops_forward_eval = (fcm.get_total_flops()) / 4.5
        res = y.sum()
        res.backward()
        flops_backward_eval = (fcm.get_total_flops() - flops_forward_eval) / 4.5
    params_eval = sum(param.numel() for param in model.parameters())
    print(f"flops_forward={flops_forward_eval/1e9:.2f}G, flops_back={flops_backward_eval/1e9:.2f}G, params={params_eval/1e6:.2f} M")

