import torch
import torch.nn as nn
import numpy as np
import Module as at_module

class CRNN(nn.Module):
	""" Proposed model
	"""
	def __init__(self, ):
		super(CRNN, self).__init__()

		cnn_in_dim = 18
		cnn_dim = 64
		res_flag = False
		self.cnn = nn.Sequential(
                at_module.CausCnnBlock(cnn_in_dim, cnn_dim, kernel=(3,3), stride=(1,1), padding=(1,1), use_res=res_flag),
                nn.MaxPool2d(kernel_size=(4, 1)),
				at_module.CausCnnBlock(cnn_dim, cnn_dim, kernel=(3,3), stride=(1,1), padding=(1,1), use_res=res_flag),
				nn.MaxPool2d(kernel_size=(2, 1)),
				at_module.CausCnnBlock(cnn_dim, cnn_dim, kernel=(3,3), stride=(1,1), padding=(1,1), use_res=res_flag),
				nn.MaxPool2d(kernel_size=(2, 1)),
				at_module.CausCnnBlock(cnn_dim, cnn_dim, kernel=(3,3), stride=(1,1), padding=(1,1), use_res=res_flag),
				nn.MaxPool2d(kernel_size=(2, 1)),
				at_module.CausCnnBlock(cnn_dim, cnn_dim, kernel=(3,3), stride=(1,1), padding=(1,1), use_res=res_flag),
				nn.MaxPool2d(kernel_size=(2, 5)),
            )

		ratio = 2
		rnn_in_dim = 256
		rnn_hid_dim = 256
		rnn_out_dim = 128*2*ratio
		rnn_bdflag = False
		if rnn_bdflag:
			rnn_ndirection = 2
		else:
			rnn_ndirection = 1
		self.rnn_bdflag = rnn_bdflag
		self.rnn = torch.nn.GRU(input_size=rnn_in_dim, hidden_size=rnn_hid_dim, num_layers=1,
								batch_first=True, bias=True, dropout=0.4, bidirectional=rnn_bdflag)

		self.rnn_fc = nn.Sequential(
			torch.nn.Linear(in_features=rnn_ndirection * rnn_hid_dim, out_features=rnn_out_dim),  # ,bias=False
			nn.Tanh(),
		)
		self.ipd2xyz = nn.Linear(512,256)
		self.relu = nn.ReLU()
		self.ipd2xyz2 = nn.Linear(256,360)
		self.sigmoid= nn.Sigmoid()
	def forward(self, x):
		fea = x
		nb, _, nf, nt = fea.shape # (55,4,256,1249)
		fea_cnn = self.cnn(fea)  # (nb, nch, nf, nt) 
		fea_rnn_in = fea_cnn.view(nb, -1, fea_cnn.size(3))  # (nb, nch*nf,nt), nt = 1
		fea_rnn_in = fea_rnn_in.permute(0, 2, 1)  # (nb, nt, nfea)

		fea_rnn, _ = self.rnn(fea_rnn_in)
		fea_rnn_fc = self.rnn_fc(fea_rnn) # (nb, nt, 2nf) 66,104,256
		fea_rnn_fc = self.relu(self.ipd2xyz(fea_rnn_fc))
		fea_rnn_fc = self.sigmoid(self.ipd2xyz2(fea_rnn_fc))
		#print(fea_rnn_fc.shape)
		return fea_rnn_fc





if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=7, python -m models.arch.SpatialNet
    x = torch.randn((1,18,257,249)) #.cuda() # 251 = 4 second; 129 = 8 kHz; 257 = 16 kHz
    model = CRNN()
    y = model(x)
    print(y.shape)
   
