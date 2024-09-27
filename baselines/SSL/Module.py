import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

# %% Complex number operations

def complex_multiplication(x, y):
	return torch.stack([ x[...,0]*y[...,0] - x[...,1]*y[...,1],   x[...,0]*y[...,1] + x[...,1]*y[...,0]  ], dim=-1)


def complex_conjugate_multiplication(x, y):
	return torch.stack([ x[...,0]*y[...,0] + x[...,1]*y[...,1],   x[...,1]*y[...,0] - x[...,0]*y[...,1]  ], dim=-1)


def complex_cart2polar(x):
	mod = torch.sqrt( complex_conjugate_multiplication(x, x)[..., 0] )
	phase = torch.atan2(x[..., 1], x[..., 0])
	return torch.stack((mod, phase), dim=-1)


# %% Signal processing and DOA estimation layers

class STFT(nn.Module):
	""" Function: Get STFT coefficients of microphone signals (batch processing by pytorch)
        Args:       win_len         - the length of frame / window
                    win_shift_ratio - the ratio between frame shift and frame length
                    nfft            - the number of fft points
                    win             - window type 
                                    'boxcar': a rectangular window (equivalent to no window at all)
                                    'hann': a Hann window
					signal          - the microphone signals in time domain (nbatch, nsample, nch)
        Returns:    stft            - STFT coefficients (nbatch, nf, nt, nch)
    """

	def __init__(self, win_len, win_shift_ratio, nfft, win='hann'):
		super(STFT, self).__init__()

		self.win_len = win_len
		self.win_shift_ratio = win_shift_ratio
		self.nfft = nfft
		self.win = win

	def forward(self, signal):

		nsample = signal.shape[-2]
		nch = signal.shape[-1]
		win_shift = int(self.win_len * self.win_shift_ratio)
		nf = int(self.nfft / 2) + 1

		nb = signal.shape[0]
		# nt = int((nsample) / win_shift) + 1  # for iSTFT
		nt = np.floor((nsample) / win_shift + 1).astype(int)
		stft = torch.zeros((nb, nf, nt, nch), dtype=torch.complex64)

		if self.win == 'hann':
			window = torch.hann_window(window_length=self.win_len, device=signal.device)
		for ch_idx in range(0, nch, 1):
			stft[:, :, :, ch_idx] = torch.stft(signal[:, :, ch_idx], n_fft=self.nfft, hop_length=win_shift, win_length=self.win_len,
								   window=window, center=True, normalized=False, return_complex=True)

		return stft


class getMetric(nn.Module):
	"""  
	Call: 
	# single source 
	getmetric = at_module.getMetric(source_mode='single', metric_unfold=True)
	metric = self.getmetric(doa_gt, vad_gt, doa_est, vad_est, ae_mode=['azi,'ele'], ae_TH=30, useVAD=False, vad_TH=vad_TH)
	# multiple source
	self.getmetric = getMetric(source_mode='multiple', metric_unfold=True)
	metric = self.getmetric(doa_gt, vad_gt, doa_est, vad_est, ae_mode=['azi,'ele'], ae_TH=30, useVAD=False, vad_TH=[2/3, 0.2]])
	"""
	def __init__(self, source_mode='multiple', metric_unfold=True, large_number=10000, invalid_source_idx=10):
		"""
		Args:
			useVAD	 	-  False, True
			soruce_mode	- 'single', 'multiple'
		"""
		super(getMetric, self).__init__()

		# self.ae_mode = ae_mode
		# self.ae_TH = ae_TH
		# self.useVAD = useVAD
		self.source_mode = source_mode
		self.metric_unfold = metric_unfold
		self.inf = large_number
		self.invlid_sidx = invalid_source_idx

	def forward(self, doa_gt, vad_gt, doa_est, vad_est, ae_mode, ae_TH=30, useVAD=True, vad_TH=[0.5,0.5]):
		"""
		Args:
			doa_gt, doa_est - (nb, nt, 2, ns) in degrees
			vad_gt, vad_est - (nb, nt, ns) binary values
			ae_mode 		- angle error mode, [*, *, *], * - 'azi', 'ele', 'aziele' 
			ae_TH			- angle error threshold, namely azimuth error threshold in degrees
			vad_TH 			- VAD threshold, [gtVAD_TH, estVAD_TH] 
		Returns:
			ACC, MAE or ACC, MD, FA, MAE, RMSE - [*, *, *]
		"""
		device = doa_gt.device
		# doa_gt = doa_gt * 180 / np.pi
		# doa_est = doa_est * 180 / np.pi
		if self.source_mode == 'single':

			nbatch, nt, naziele, nsources = doa_est.shape
			if useVAD == False:
				vad_gt = torch.ones((nbatch, nt, nsources)).to(device)
				vad_est = torch.ones((nbatch,nt, nsources)).to(device)
			else:
				vad_gt = vad_gt > vad_TH[0]
				vad_est = vad_est > vad_TH[1]
			vad_est = vad_est * vad_gt

			azi_error = self.angular_error(doa_est[:,:,1,:], doa_gt[:,:,1,:], 'azi')            
			ele_error = self.angular_error(doa_est[:,:,0,:], doa_gt[:,:,0,:], 'ele')
			aziele_error = self.angular_error(doa_est.permute(2,0,1,3), doa_gt.permute(2,0,1,3), 'aziele')
			
			corr_flag = ((azi_error < ae_TH)+0.0) * vad_est # Accorrding to azimuth error
			act_flag = 1*vad_gt
			K_corr = torch.sum(corr_flag) 
			ACC = torch.sum(corr_flag) / torch.sum(act_flag)
			MAE = []
			if 'ele' in ae_mode:
				MAE += [torch.sum(vad_gt * ele_error) / torch.sum(act_flag)]
			elif 'azi' in ae_mode:
				MAE += [ torch.sum(vad_gt * azi_error) / torch.sum(act_flag)]
				# MAE += [torch.sum(corr_flag * azi_error) / torch.sum(act_flag)]
			elif 'aziele' in ae_mode:
				MAE += [torch.sum(vad_gt * aziele_error) / torch.sum(act_flag)]
			else:
				raise Exception('Angle error mode unrecognized')
			MAE = torch.tensor(MAE)

			metric = [ACC, MAE]
			if self.metric_unfold:
				metric = self.unfold_metric(metric)

			return metric

		elif self.source_mode == 'multiple':
			nbatch = doa_est.shape[0]
			nmode = len(ae_mode)
			acc = torch.zeros(nbatch, 1)
			md = torch.zeros(nbatch, 1)
			fa = torch.zeros(nbatch, 1)
			mae = torch.zeros(nbatch, nmode)
			rmse = torch.zeros(nbatch, nmode)
			for b_idx in range(nbatch):
				doa_gt_one = doa_gt[b_idx, ...]
				doa_est_one = doa_est[b_idx, ...]
				
				nt = doa_gt_one.shape[0]
				num_sources_gt = doa_gt_one.shape[2]
				num_sources_est = doa_est_one.shape[2]

				if useVAD == False:
					vad_gt_one = torch.ones((nt, num_sources_gt)).to(device)
					vad_est_one = torch.ones((nt, num_sources_est)).to(device)
				else:
					vad_gt_one = vad_gt[b_idx, ...]
					vad_est_one = vad_est[b_idx, ...]
					vad_gt_one = vad_gt_one > vad_TH[0]
					vad_est_one = vad_est_one > vad_TH[1]

				corr_flag = torch.zeros((nt, num_sources_gt)).to(device)
				azi_error = torch.zeros((nt, num_sources_gt)).to(device)
				ele_error = torch.zeros((nt, num_sources_gt)).to(device)
				aziele_error = torch.zeros((nt, num_sources_gt)).to(device)
				K_gt = vad_gt_one.sum(axis=1)
				vad_gt_sum = torch.reshape(vad_gt_one.sum(axis=1)>0, (nt, 1)).repeat((1, num_sources_est))
				vad_est_one = vad_est_one * vad_gt_sum
				K_est = vad_est_one.sum(axis=1)
				for t_idx in range(nt):
					num_gt = int(K_gt[t_idx].item())
					num_est = int(K_est[t_idx].item())
					if num_gt>0 and num_est>0:
						est = doa_est_one[t_idx, :, vad_est_one[t_idx,:]>0]
						gt = doa_gt_one[t_idx, :, vad_gt_one[t_idx,:]>0]
						dist_mat_az = torch.zeros((num_gt, num_est))
						dist_mat_el = torch.zeros((num_gt, num_est))
						dist_mat_azel = torch.zeros((num_gt, num_est))
						for gt_idx in range(num_gt):
							for est_idx in range(num_est):
								dist_mat_az[gt_idx, est_idx] = self.angular_error(est[1,est_idx], gt[1,gt_idx], 'azi')
								dist_mat_el[gt_idx, est_idx] = self.angular_error(est[0,est_idx], gt[0,gt_idx], 'ele')
								dist_mat_azel[gt_idx, est_idx] = self.angular_error(est[:,est_idx], gt[:,gt_idx], 'aziele')
						
						invalid_assigns = dist_mat_az > ae_TH  # Accorrding to azimuth error
						# 	invalid_assigns = dist_mat_el > ae_TH
						# 	invalid_assigns = dist_mat_azel > ae_TH
						
						dist_mat_az_bak = dist_mat_az.clone()
						dist_mat_az_bak[invalid_assigns] = self.inf
						assignment = list(linear_sum_assignment(dist_mat_az_bak))
						assignment = self.judge_assignment(dist_mat_az_bak, assignment)
						for src_idx in range(num_gt):
							if assignment[src_idx] != self.invlid_sidx:
								corr_flag[t_idx, src_idx] = 1
								azi_error[t_idx, src_idx] = dist_mat_az[src_idx, assignment[src_idx]]
								ele_error[t_idx, src_idx] = dist_mat_el[src_idx, assignment[src_idx]]
								aziele_error[t_idx, src_idx] = dist_mat_azel[src_idx, assignment[src_idx]]

				K_corr = corr_flag.sum(axis=1)
				acc[b_idx, :] = K_corr.sum(axis=0) / (K_gt.sum(axis=0)+0.000001)
				md[b_idx, :] = (K_gt.sum(axis=0) - K_corr.sum(axis=0)) / K_gt.sum(axis=0)
				fa[b_idx, :] = (K_est.sum(axis=0) - K_corr.sum(axis=0)) / K_gt.sum(axis=0)

				mae_temp = []
				rmse_temp = []
				if 'ele' in ae_mode:
					mae_temp += [((ele_error*corr_flag).sum(axis=0)).sum() / (K_corr.sum(axis=0)+1e-5)]
					rmse_temp += [torch.sqrt(((ele_error*ele_error*corr_flag).sum(axis=0)).sum() / (K_corr.sum(axis=0)+1e-5))]
				elif 'azi' in ae_mode:
					mae_temp += [((azi_error*corr_flag).sum(axis=0)).sum() / (K_corr.sum(axis=0)+1e-5)]
					rmse_temp += [torch.sqrt(((azi_error*azi_error*corr_flag).sum(axis=0)).sum() / (K_corr.sum(axis=0)+1e-5))]
				elif 'aziele' in ae_mode:
					mae_temp += [((aziele_error*corr_flag).sum(axis=0)).sum() / (K_corr.sum(axis=0)+1e-5)]
					rmse_temp += [torch.sqrt(((aziele_error*aziele_error*corr_flag).sum(axis=0)).sum() / (K_corr.sum(axis=0)+1e-5))]
				else:
					raise Exception('Angle error mode unrecognized')
				mae[b_idx, :] = torch.tensor(mae_temp)
				rmse[b_idx, :] = torch.tensor(rmse_temp)

			ACC = torch.mean(acc, dim=0)
			MD = torch.mean(md, dim=0)
			FA = torch.mean(fa, dim=0)
			MAE = torch.mean(mae, dim=0)
			RMSE = torch.mean(rmse, dim=0)

			metric = [ACC, MD, FA, MAE, RMSE]
			if self.metric_unfold:
				metric = self.unfold_metric(metric)

			return metric

	def judge_assignment(self, dist_mat, assignment):
		final_assignment = torch.tensor([self.invlid_sidx for i in range(dist_mat.shape[0])])
		for i in range(min(dist_mat.shape[0],dist_mat.shape[1])):
			if dist_mat[assignment[0][i], assignment[1][i]] != self.inf:
				final_assignment[assignment[0][i]] = assignment[1][i]
			else:
				final_assignment[i] = self.invlid_sidx
		return final_assignment

	def angular_error(self, est, gt, ae_mode):
		if ae_mode == 'azi':
			ae = torch.abs((est-gt+180)%360 - 180)
		elif ae_mode == 'ele':
			ae = torch.abs(est-gt)
		elif ae_mode == 'aziele':
			ele_gt = gt[0, ...].float() / 180 * np.pi
			azi_gt = gt[1, ...].float() / 180 * np.pi
			ele_est = est[0, ...].float() / 180 * np.pi
			azi_est = est[1, ...].float() / 180 * np.pi
			aux = torch.cos(ele_gt) * torch.cos(ele_est) + torch.sin(ele_gt) * torch.sin(ele_est) * torch.cos(azi_gt - azi_est)
			aux[aux.gt(0.99999)] = 0.99999
			aux[aux.lt(-0.99999)] = -0.99999
			ae = torch.abs(torch.acos(aux)) * 180 / np.pi
		else:
			raise Exception('Angle error mode unrecognized')
		
		return ae

	def unfold_metric(self, metric):
		metric_unfold = []
		for m in metric:
			if m.numel() !=1:
				for n in range(m.numel()):
					metric_unfold += [m[n]]
			else:
				metric_unfold += [m]
		return metric_unfold
 

class AddChToBatch(nn.Module):
	""" Change dimension from  (nb, nch, ...) to (nb*(nch-1), ...) 
	"""
	def __init__(self, ch_mode):
		super(AddChToBatch, self).__init__()
		self.ch_mode = ch_mode

	def forward(self, data):
		nb = data.shape[0]
		nch = data.shape[1]

		if self.ch_mode == 'M':
			data_adjust = torch.zeros((nb*(nch-1),2)+data.shape[2:], dtype=torch.complex64) # (nb*(nch-1),2,nf,nt)
			for b_idx in range(nb):
				st = b_idx*(nch-1)
				ed = (b_idx+1)*(nch-1)
				data_adjust[st:ed, 0, ...] = data[b_idx, 0 : 1, ...].expand((nch-1,)+data.shape[2:])
				data_adjust[st:ed, 1, ...] = data[b_idx, 1 : nch, ...]

		elif self.ch_mode == 'MM':
			data_adjust = torch.zeros((nb*int((nch-1)*nch/2),2)+data.shape[2:], dtype=torch.complex64) # (nb*(nch-1)*nch/2,2,nf,nt)
			for b_idx in range(nb):
				for ch_idx in range(nch-1):
					st = b_idx*int((nch-1)*nch/2) + int((2*nch-2-ch_idx+1)*ch_idx/2)
					ed = b_idx*int((nch-1)*nch/2) + int((2*nch-2-ch_idx)*(ch_idx+1)/2)
					data_adjust[st:ed, 0, ...] = data[b_idx, ch_idx:ch_idx+1, ...].expand((nch-ch_idx-1,)+data.shape[2:])
					data_adjust[st:ed, 1, ...] = data[b_idx, ch_idx+1:, ...]
			
		return data_adjust.contiguous()

class RemoveChFromBatch(nn.Module):
	""" Change dimension from (nb*nmic, nt, nf) to (nb, nmic, nt, nf)
	"""
	def __init__(self, ch_mode):
		super(RemoveChFromBatch, self).__init__()
		self.ch_mode = ch_mode

	def forward(self, data, nb):
		nmic = int(data.shape[0]/nb)
		data_adjust = torch.zeros((nb, nmic)+data.shape[1:], dtype=torch.float32).to(data.device)
		for b_idx in range(nb):
			st = b_idx * nmic
			ed = (b_idx + 1) * nmic
			data_adjust[b_idx, ...] = data[st:ed, ...]
			
		return data_adjust.contiguous()
class CausConv2d(nn.Module):
	""" Causal 2D Convolution for spectrograms and GCCs sequences
	"""
	def __init__(self, in_channels, out_channels, kernel_size):
		super(CausConv2d, self).__init__()
		self.pad = kernel_size[0] - 1
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(self.pad, 0))

	def forward(self, x):
		return self.conv(x)[:, :, :-self.pad, :]


class CausConv1d(nn.Module):
	""" Causal 1D Convolution
	"""
	def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
		super(CausConv1d, self).__init__()
		self.pad = (kernel_size - 1) * dilation
		self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.pad, dilation=dilation)
		
	def forward(self, x):
		return self.conv(x)[:, :, :-self.pad]

class CausCnnBlock1x1(nn.Module):
	# expansion = 1
	def __init__(self, inplanes, planes, kernel=(1,1), stride=(1,1), padding=(0,0)):
		super(CausCnnBlock1x1, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel, stride=stride, padding=padding, bias=False)

	def forward(self, x):
		out = self.conv1(x)

		return out

class CausCnnBlock(nn.Module): 
	""" Function: Basic convolutional block
    """
	# expansion = 1
	def __init__(self, inplanes, planes, kernel=(3,3), stride=(1,1), padding=(1,1), use_res=True, downsample=None):
		super(CausCnnBlock, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel, stride=stride, padding=padding, bias=False)
		self.In1 = nn.InstanceNorm2d(planes)
		self.relu = nn.ReLU()

		self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel, stride=stride, padding=padding, bias=False)
		self.In2 = nn.InstanceNorm2d(planes)

		self.downsample = downsample
		self.stride = stride
		self.pad = padding
		self.use_res = use_res

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.In1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.In2(out)

		if self.use_res == True:
			if self.downsample is not None:
				residual = self.downsample(x)
			out += residual

		out = self.relu(out)

		return out

class DPIPD(nn.Module):
	""" Complex-valued Direct-path inter-channel phase difference	
	"""

	def __init__(self, ndoa_candidate, mic_location, nf=257, fre_max=8000, ch_mode='M', speed=343.0):
		super(DPIPD, self).__init__()

		self.ndoa_candidate = ndoa_candidate
		self.mic_location = mic_location
		self.nf = nf
		self.fre_max = fre_max
		self.speed = speed
		self.ch_mode = ch_mode

		nmic = mic_location.shape[-2]
		nele = ndoa_candidate[0]
		nazi = ndoa_candidate[1]
		ele_candidate = np.linspace(np.pi/2, np.pi/2, nele)
		azi_candidate = np.linspace(-np.pi, np.pi, nazi)
		ITD = np.empty((nele, nazi, nmic, nmic))  # Time differences, floats
		IPD = np.empty((nele, nazi, nf, nmic, nmic))  # Phase differences
		fre_range = np.linspace(0.0, fre_max, nf)
		for m1 in range(nmic):
			for m2 in range(nmic):
				r = np.stack([np.outer(np.sin(ele_candidate), np.cos(azi_candidate)),
							  np.outer(np.sin(ele_candidate), np.sin(azi_candidate)),
							  np.tile(np.cos(ele_candidate), [nazi, 1]).transpose()], axis=2)
				ITD[:, :, m1, m2] = np.dot(r, mic_location[m2, :] - mic_location[m1, :]) / speed
				IPD[:, :, :, m1, m2] = -2 * np.pi * np.tile(fre_range[np.newaxis, np.newaxis, :], [nele, nazi, 1]) * \
									   np.tile(ITD[:, :, np.newaxis, m1, m2], [1, 1, nf])
		dpipd_template_ori = np.exp(1j * IPD)
		self.dpipd_template = self.data_adjust(dpipd_template_ori) # (nele, nazi, nf, nmic-1) / (nele, nazi, nf, nmic*(nmic-1)/2)
		del ITD, IPD

	def forward(self, source_doa=None):
		# source_doa: (nb, ntimestep, 2, nsource)
		mic_location = self.mic_location
		nf = self.nf
		fre_max = self.fre_max
		speed = self.speed

		if source_doa is not None:
			source_doa = source_doa.transpose(0, 1, 3, 2) # (nb, ntimestep, nsource, 2)
			nmic = mic_location.shape[-2]
			nb = source_doa.shape[0]
			nsource = source_doa.shape[-2]
			ntime = source_doa.shape[-3]
			ITD = np.empty((nb, ntime, nsource, nmic, nmic))  # Time differences, floats
			IPD = np.empty((nb, ntime, nsource, nf, nmic, nmic))  # Phase differences
			fre_range = np.linspace(0.0, fre_max, nf)

			for m1 in range(nmic):
				for m2 in range(nmic):
					r = np.stack([np.sin(source_doa[:, :, :, 0]) * np.cos(source_doa[:, :, :, 1]),
								  np.sin(source_doa[:, :, :, 0]) * np.sin(source_doa[:, :, :, 1]),
								  np.cos(source_doa[:, :, :, 0])], axis=3)
					ITD[:, :, :, m1, m2] = np.dot(r, mic_location[m1, :] - mic_location[m2, :]) / speed # t2- t1
					IPD[:, :, :, :, m1, m2] = -2 * np.pi * np.tile(fre_range[np.newaxis, np.newaxis, np.newaxis, :],
										[nb, ntime, nsource, 1]) * np.tile(ITD[:, :, :, np.newaxis, m1, m2], [1, 1, 1, nf])*(-1)  # !!!! delete -1

			dpipd_ori = np.exp(1j * IPD)
			dpipd = self.data_adjust(dpipd_ori) # (nb, ntime, nsource, nf, nmic-1) / (nb, ntime, nsource, nf, nmic*(nmic-1)/2)
			#print(dpipd.shape)
			dpipd = dpipd.transpose(0, 1, 3, 4, 2) # (nb, ntime, nf, nmic-1, nsource)

		else:
			dpipd = None

		return self.dpipd_template, dpipd
	
	def data_adjust(self, data):
		# change dimension from (..., nmic-1) to (..., nmic*(nmic-1)/2)
		if self.ch_mode == 'M':
			data_adjust = data[..., 0, 1:] # (..., nmic-1)
		elif self.ch_mode == 'MM':
			nmic = data.shape[-1]
			data_adjust = np.empty(data.shape[:-2] + (int(nmic*(nmic-1)/2),), dtype=np.complex64)
			for mic_idx in range(nmic - 1):
				st = int((2 * nmic - 2 - mic_idx + 1) * mic_idx / 2)
				ed = int((2 * nmic - 2 - mic_idx) * (mic_idx + 1) / 2)
				data_adjust[..., st:ed] = data[..., mic_idx, (mic_idx+1):] # (..., nmic*(nmic-1)/2)
		else:
			raise Exception('Microphone channel mode unrecognised')

		return data_adjust



class PredDOA(nn.Module):
	def __init__(self,
	      method_mode = 'IDL',
		  source_num_mode = 'KNum',
		  max_num_sources = 1,
		  res_the = 1,
		  res_phi = 180,
		  fs = 16000,
		  nfft = 512,
		  ch_mode = 'M',
		  mic_location = np.array((((-0.08, 0.0, 0.0),(-0.04, 0.0, 0.0), (0.04, 0.0, 0.0),( 0.08, 0.0, 0.0),))),
		  device = 'cuda',
		  ):
		super(PredDOA, self).__init__()
		self.nfft = nfft
		self.fre_max = fs / 2
		self.ch_mode = ch_mode
		self.method_mode = method_mode
		self.source_num_mode = source_num_mode
		self.max_num_sources = max_num_sources
		self.fre_range_used = range(1, int(self.nfft/2)+1, 1)
		self.removebatch = RemoveChFromBatch(ch_mode=self.ch_mode)
		self.gerdpipd = DPIPD(ndoa_candidate=[res_the, res_phi],
                                        mic_location=mic_location,
                                        nf=int(self.nfft/2) + 1,
                                        fre_max=self.fre_max,
                                        ch_mode=self.ch_mode,
                                        speed=340)
		self.getmetric = getMetric(
            source_mode='single', metric_unfold=True)
		self.dev = device
  		
	def forward(self, pred_batch, gt_batch, tar_type, idx):
		if tar_type == 'spect':
			pred_batch,_= self.predgt2DOA_spect(pred_batch = pred_batch, gt_batch = gt_batch)
			metric = self.evaluate(pred_batch=pred_batch, gt_batch=gt_batch,idx=idx, tar_type='spect')
		elif tar_type == 'ipd':
			pred_batch,_= self.predgt2DOA_ipd(pred_batch = pred_batch, gt_batch = gt_batch)
			metric = self.evaluate(pred_batch=pred_batch, gt_batch=gt_batch,idx=idx, tar_type='ipd')
		return metric

	def predgt2DOA_spect(self, pred_batch, gt_batch):
		nb,nt,nn = pred_batch.shape
		vad_batch_pred,doa_batch_pred = pred_batch.topk(1,dim=-1)
		pred_batch = []
		pred_batch += [doa_batch_pred[:,:,:]]
		pred_batch += [vad_batch_pred]
		if gt_batch is not None: 
			if type(gt_batch) is list:
				for idx in range(len(gt_batch)):
					gt_batch[idx] = gt_batch[idx].detach()
			else:
				gt_batch = gt_batch.detach()
		return pred_batch,gt_batch

	def predgt2DOA_ipd(self, pred_batch, gt_batch):
		nb,nt,nf,nmic,nsrc = pred_batch.shape
		pred_batch = pred_batch.permute(0,3,1,2,4).reshape(nb*nmic,nt,nf,nsrc)
		pred_ipd = pred_batch 
		doa_batch_pred = torch.zeros((nb,nt,2,nsrc))
		vad_batch_pred = torch.zeros(((nb,nt,nsrc)))
		for i in range(nsrc):
			pred_batch_temp, gt_batch_temp = self.predgt2DOA(pred_ipd[:,:,:,i], gt_batch)
			doa_batch_pred[:,:,:,i:i+1] = pred_batch_temp[0]
			vad_batch_pred[:,:,i:i+1] = pred_batch_temp[1]
		pred_batch = [ doa_batch_pred ]
		pred_batch += [ vad_batch_pred ]
		pred_batch += [ pred_ipd ]
		if gt_batch is not None: 
			if type(gt_batch) is list:
				for idx in range(len(gt_batch)-2):
					gt_batch[idx] = gt_batch[idx].detach()
			else:
				gt_batch = gt_batch.detach()
		return pred_batch,gt_batch

	def predgt2DOA(self, pred_batch=None, gt_batch=None, time_pool_size=None):
		if pred_batch is not None:
			pred_batch = pred_batch.detach()
			dpipd_template_sbatch, _ = self.gerdpipd( ) # (nele, nazi, nf, nmic-1)
			nele, nazi, _, nmic = dpipd_template_sbatch.shape
			nbnmic, nt, nf = pred_batch.shape
			nb = int(nbnmic/nmic)

			dpipd_template_sbatch = np.concatenate((dpipd_template_sbatch.real[:,:,self.fre_range_used,:], dpipd_template_sbatch.imag[:,:,self.fre_range_used,:]), axis=2).astype(np.float32) # (nele, nazi, 2nf, nmic-1)
			dpipd_template = np.tile(dpipd_template_sbatch[np.newaxis, :, :, :, :],
									 [nb, 1, 1, 1, 1]) # (nb, nele, nazi, 2nf, nmic-1)
			dpipd_template = torch.from_numpy(dpipd_template) # (nb, nele, nazi, 2nf, nmic-1)
			pred_rebatch = self.removebatch(pred_batch, nb).permute(0, 2, 3, 1) # (nb, nt, 2nf, nmic-1
			
			pred_rebatch = pred_rebatch.to(self.dev)
			dpipd_template = dpipd_template.to(self.dev)
			if time_pool_size is not None:
				nt_pool = int(nt / time_pool_size)
				pred_phases = torch.zeros((nb, nt_pool, nf, nmic), dtype=torch.float32, requires_grad=False)  # (nb, nt_pool, 2nf, nmic-1)
				
				pred_phases = pred_phases.to(self.dev)
				for t_idx in range(nt_pool):
					pred_phases[:, t_idx, :, :]  = torch.mean(
						# pred_rebatch[:, t_idx * time_pool_size: (t_idx ) * time_pool_size+1, :, :], dim=1)
						pred_rebatch[:, t_idx*time_pool_size: (t_idx+1)*time_pool_size, :, :], dim=1)
				pred_rebatch = pred_phases * 1
				nt = nt_pool * 1
			pred_spatial_spectrum = torch.bmm(pred_rebatch.reshape(nb, nt, -1),
											  dpipd_template.reshape(nb, nele, nazi, -1).permute(0, 3, 1, 2).view(nb, nmic * nf, -1))/(nmic*nf/2)  # (nb, nt, nele*nazi)
			pred_spatial_spectrum = pred_spatial_spectrum.view(nb, nt, nele, nazi)

			pred_DOAs = torch.zeros((nb, nt, 2, self.max_num_sources), dtype=torch.float32, requires_grad=False)
			pred_VADs = torch.zeros((nb, nt, self.max_num_sources), dtype=torch.float32, requires_grad=False)
			
			pred_DOAs = pred_DOAs.to(self.dev)
			pred_VADs = pred_VADs.to(self.dev)
			if self.method_mode == 'IDL': # iterative source detection and localization
				for source_idx in range(self.max_num_sources):
					map = torch.bmm(pred_rebatch.reshape(nb, nt, -1),
									dpipd_template.reshape(nb, nele, nazi, -1).permute(0, 3, 1, 2).view(nb, nmic * nf, -1)) / (
									  nmic * nf / 2)  # (nb, nt, nele*nazi)
					map = map.view(nb, nt, nele, nazi)

					max_flat_idx = map.reshape((nb, nt, -1)).argmax(2)
					ele_max_idx, azi_max_idx = np.unravel_index(max_flat_idx.cpu().numpy(), map.shape[2:])  # (nb, nt)

					ele_candidate = np.linspace(np.pi/2, np.pi/2, nele)
					azi_candidate = np.linspace(-np.pi, np.pi, nazi)
					pred_DOA = np.stack((ele_candidate[ele_max_idx], azi_candidate[azi_max_idx]),
										axis=-1)  # (nb, nt, 2)
					pred_DOA = torch.from_numpy(pred_DOA)
					
					pred_DOA = pred_DOA.to(self.dev)
					pred_DOAs[:, :, :, source_idx] = pred_DOA

					max_dpipd_template = torch.zeros((nb, nt, nf, nmic), dtype=torch.float32, requires_grad=False)
					
					max_dpipd_template = max_dpipd_template.to(self.dev)
					for b_idx in range(nb):
						for t_idx in range(nt):
							max_dpipd_template[b_idx, t_idx, :, :] = \
								dpipd_template[b_idx, ele_max_idx[b_idx, t_idx], azi_max_idx[b_idx, t_idx], :,
								:] * 1.0  # (nb, nt, 2nf, nmic-1)
							ratio = torch.sum(
								max_dpipd_template[b_idx, t_idx, :, :] * pred_rebatch[b_idx, t_idx, :, :]) / \
									torch.sum(
										max_dpipd_template[b_idx, t_idx, :, :] * max_dpipd_template[b_idx, t_idx, :, :])
							# ratio2 = map[b_idx,t_idx,ele_max_idx[b_idx, t_idx], azi_max_idx[b_idx, t_idx] ]
							max_dpipd_template[b_idx, t_idx, :, :] = ratio * max_dpipd_template[b_idx, t_idx, :, :]
							if self.source_num_mode == 'KNum':
								pred_VADs[b_idx, t_idx, source_idx] = 1
							elif self.source_num_mode == 'UnkNum':
								pred_VADs[b_idx, t_idx, source_idx] = ratio * 1
					pred_rebatch = pred_rebatch - max_dpipd_template
     
			pred_batch = [pred_DOAs]
			pred_batch += [pred_VADs]
			pred_batch += [pred_spatial_spectrum]

		if gt_batch is not None: 
			if type(gt_batch) is list:
				for idx in range(len(gt_batch)-2):
					gt_batch[idx] = gt_batch[idx].detach()
			else:
				gt_batch = gt_batch.detach()

		return pred_batch, gt_batch

	def evaluate(self, pred_batch=None, gt_batch=None, tar_type=None, vad_TH=[0.001, 0.05],idx=None):
		ae_mode = ['azi']
		doa_gt =torch.cat((gt_batch[0],gt_batch[0]),dim=-1)# * 180 / np.pi 
		if tar_type == 'spect':
			doa_est = torch.cat((pred_batch[0],pred_batch[0]),dim=-1)
			doa_gt = doa_gt[:,:,:,np.newaxis].to(self.dev)
			doa_est = doa_est[:,:,:,np.newaxis].to(self.dev)		
			vad_est = pred_batch[-1].to(self.dev)
			vad_gt = gt_batch[1].to(self.dev)
		elif tar_type == 'ipd':
			doa_est = pred_batch[0] * 180 / np.pi
			doa_est = doa_est.to(self.dev)
			doa_gt = doa_gt[:,:,:,np.newaxis].to(self.dev)
			vad_est = torch.ones(vad_gt.shape).to(vad_gt)
			vad_gt = gt_batch[-1].to(self.dev)
		metric_5 = {}
		#metric_10 = {}
		print(doa_gt.shape)
		if idx != None:
			np.save('./results/'+str(idx)+'_doagt',doa_gt.cpu().numpy())
			np.save('./results/'+str(idx)+'_doaest',doa_est.cpu().numpy())
			np.save('./results/'+str(idx)+'_vadgt',vad_gt.cpu().numpy())
			np.save('./results/'+str(idx)+'_vadest',vad_est.cpu().numpy())
   	
		#metric_10['ACC'], metric_10['MAE'], = \
		# 	self.getmetric(doa_gt, vad_gt, doa_est, vad_est, ae_mode = ae_mode, ae_TH=10, useVAD=False, vad_TH=vad_TH)
		metric_5['ACC'], metric_5['MAE'], = \
		  	self.getmetric(doa_gt, vad_gt, doa_est, vad_est, ae_mode = ae_mode, ae_TH=5, useVAD=True, vad_TH=vad_TH)
		return metric_5#, metric_10

