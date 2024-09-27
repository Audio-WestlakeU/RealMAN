import torch
import os
from torch.utils.data import Dataset
import soundfile as sf
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy import signal
from utils_ import search_files, audiowu_high_array_geometry
from pathlib import Path


class RealData(Dataset):
	def __init__(self, data_dir, target_dir, noise_dir, input_fs=16000, use_mic_id =[1,2,3,4,5,6,7,8,0], target_fs=16000, snr = [-10,15], wav_use_len=4, on_the_fly = True, is_variable_array = False):
		self.ends='CH1.flac'
		self.data_paths = []
		self.all_targets = pd.DataFrame() 
		for dir in target_dir:
			target = pd.read_csv(dir)
			self.data_paths += [data_dir+i for i in target['filename'].to_list()]
			self.all_targets = pd.concat([self.all_targets, target], ignore_index=True)
		# set filename as the search index
		self.all_targets.set_index('filename', inplace=True)
		self.target_fs = target_fs
		self.input_fs = input_fs
		self.SNR = snr
		self.wav_use_len = 4
		self.target_len = self.wav_use_len * 10
		self.pos_mics =  audiowu_high_array_geometry()
		# get all noise file path (ends with CH1.flac)
		if on_the_fly:
			self.noise_paths = search_files(noise_dir,flag=self.ends)
		self.is_varibale_array = is_variable_array
		self.on_the_fly = on_the_fly
		self.use_mic_id = use_mic_id
	def __len__(self):
		return len(self.data_paths)
	
	def cal_vad(self,sig,fs=16000,th=-2.5):
		window_size = int(0.1 * fs)
		num_windows = len(sig) // window_size
		energies = []
		times = []
		for i in range(num_windows):
			window = sig[i * window_size:(i + 1) * window_size]
			fft_result = np.fft.fft(window)
			fft_result = fft_result[:window_size // 2]
			freqs = np.fft.fftfreq(window_size, 1 / fs)[:window_size // 2]
			energy = np.sum(np.abs(fft_result[(freqs >= 0) & (freqs <= 8000)]) ** 2)
			energies.append(np.log10(energy+1e-10))
		energies = np.array(energies)
		energies = np.where(energies < th, 0, 1)
		return torch.from_numpy(energies[:,np.newaxis])
    
    # variable-array model of the paper
	# def select_mic_array_no_linear(self, pos_mics, rng):
	# 	mic_id_list = np.arange(28)
	# 	not_use_five_linear_mics = True
	# 	while not_use_five_linear_mics:
	# 		num_values_to_select = rng.integers(low=2, high=9)
	# 		CH_list = list(rng.choice(mic_id_list, num_values_to_select, replace=False))
	# 		mic_gemo = pos_mics[CH_list, :]
	# 		if num_values_to_select == 5:
	# 			mic_gemo_sort = mic_gemo[mic_gemo[:, 0].argsort()]
	# 			distances = np.linalg.norm(mic_gemo_sort[1:] - mic_gemo_sort[:-1], axis=1)
	# 			#distances = distances.astype('float64')
	# 			process_distance = np.array([round(distance, 2) for distance in distances])
	# 			if (process_distance==0.03).all():
	# 				not_use_five_linear_mics = True
	# 			else:
	# 				not_use_five_linear_mics = False
	# 		else:
	# 			not_use_five_linear_mics = False
	# 	return CH_list, mic_gemo

	def select_mic_array_no_circle(self, pos_mics, rng):
		mic_id_list = np.arange(28)
		specific_group_1 = {0, 2, 4, 6, 24}
		specific_group_2 = {1, 3, 5, 7, 24}
		not_use_five_linear_mics = True
		while not_use_five_linear_mics: 
			num_values_to_select = rng.integers(low=2, high=9)
			CH_list = list(rng.choice(mic_id_list, num_values_to_select, replace=False))
			mic_gemo = pos_mics[CH_list, :]
			# 2 types 5-mic circle array
			if set(CH_list) == specific_group_1 or set(CH_list) == specific_group_2:
				not_use_five_linear_mics = True
			else:
				not_use_five_linear_mics = False
		return CH_list, mic_gemo

	def seg_signal(self,signal,fs,rng,dp_signal,len_signal_s=4):
		signal_start = rng.integers(low=0, high=signal.shape[0]-(len_signal_s*fs))
		#print(signal_start,signal_start*fs//frame_size,(signal_start+len_signal_s*frame_size)*fs//frame_size)
		seg_signal = signal[signal_start:signal_start+(len_signal_s*fs),:]
	
		seg_dp_signal = dp_signal[signal_start:signal_start+(len_signal_s*fs)]
		return seg_signal,signal_start,seg_dp_signal


	def load_signals(self, sig_path, use_mic_id):

		channels = []
		for i in use_mic_id:
			temp_path = sig_path.replace('.flac',f'_CH{i}.flac')
			single_ch_signal,fs = sf.read(temp_path)
			channels.append(single_ch_signal)
		mul_ch_signals = np.stack(channels, axis=-1)

		return mul_ch_signals,fs

	def load_noise(self,noise_path,begin_index,end_index,use_mic_id):
		channels = []

		for i in use_mic_id:
			temp_path = noise_path.replace('_CH1.flac', f'_CH{i}.flac')
			try:
				single_ch_signal,fs = sf.read(temp_path,start=begin_index, stop=end_index)
			except:
				print(temp_path,begin_index,end_index)
			channels.append(single_ch_signal)
		mul_ch_signals = np.stack(channels, axis=-1)
		return mul_ch_signals,fs

	def resample(self,mic_signal,fs,new_fs):
		signal_resampled = signal.resample(mic_signal, int(mic_signal.shape[0] * new_fs / fs))
		return signal_resampled

	def get_snr_coff(self, wav1, wav2, target_dB):
		ae1 = np.sum(wav1**2) / np.prod(wav1.shape)
		ae2 = np.sum(wav2**2) / np.prod(wav2.shape)
		if ae1 == 0 or ae2 == 0 or not np.isfinite(ae1) or not np.isfinite(ae2):
			return None
		coeff = np.sqrt(ae1 / ae2 * np.power(10, -target_dB / 10))
		return coeff

	def __getitem__(self, idx_seed):
		idx,seed = idx_seed
		rng = np.random.default_rng(np.random.PCG64(seed))
		sig_path = self.data_paths[idx]
		#print(self.data_paths,len(self.data_paths))
		if self.on_the_fly:	
			if self.is_varibale_array:
				use_mic_id_item,_ = self.select_mic_array_no_circle(self.pos_mics,rng=rng)
			else:
				use_mic_id_item = self.use_mic_id
			# cal vad
			dp_sig_path = sig_path.replace('/ma_speech/','/dp_speech/')
			dp_signal,dp_fs = sf.read(dp_sig_path)
			if dp_fs != self.target_fs:
				dp_signal = self.resample(mic_signal=dp_signal,fs=dp_fs,new_fs=self.target_fs)
			# print(dp_signal.shape)
			# sf.write('./dp_sig/' + str(idx)+'.wav',dp_signal,samplerate=self.target_fs)


			snr_item = rng.uniform(self.SNR[0], self.SNR[1])
			mic_signal, fs = self.load_signals(sig_path,use_mic_id=use_mic_id_item)
			if fs != self.target_fs:
				mic_signal = self.resample(mic_signal=mic_signal,fs=fs,new_fs=self.target_fs)
			len_signal = mic_signal.shape[0] / self.target_fs
			# pading or cut the source signal
			if len_signal < 5:
				input_length = int(self.wav_use_len * self.target_fs)
				input_mic_signal = np.zeros((input_length, mic_signal.shape[1]))
				min_length = min(input_length, mic_signal.shape[0])
				input_mic_signal[:min_length, :] = mic_signal[:min_length, :]
				dp_vad_temp = self.cal_vad(dp_signal)
				if dp_vad_temp.shape[0] > 40:
					dp_vad_temp = dp_vad_temp[:40,:]
				target = self.all_targets.at[sig_path.split('RealMAN/')[-1], 'angle(°)']
				if isinstance(target, float):
					targets = torch.ones((self.target_len,1)) * int(target)
					vad_source = torch.zeros((self.target_len,1))  
					dp_vad = torch.zeros((self.target_len,1))
					end_index = min(int(len_signal * 10), self.target_len)  
					vad_source[:end_index] = 1  
					dp_vad[:dp_vad_temp.shape[0],:] = dp_vad_temp
				elif isinstance(target, str):
					temp_targets = np.array([int(float(i)) for i in target.split(',')])
					targets = torch.zeros((self.target_len,1))

					length_to_copy = min(len(temp_targets), self.target_len)
					targets[:length_to_copy,:] = torch.from_numpy(temp_targets[:length_to_copy,np.newaxis])
					vad_source = torch.zeros((self.target_len,1))
					dp_vad = torch.zeros((self.target_len,1))
					vad_source[:length_to_copy] = 1
					dp_vad[:dp_vad_temp.shape[0],:] = dp_vad_temp
				else:
					print(type(target))
					print(sig_path,target) 				
			else:
				input_mic_signal,signal_start,input_dp_signal = self.seg_signal(signal=mic_signal,fs = self.target_fs,dp_signal=dp_signal,rng=rng)
				dp_vad = self.cal_vad(input_dp_signal)
				target = self.all_targets.at[sig_path.split('RealMAN/')[-1], 'angle(°)']
				if isinstance(target, float):
					targets = torch.ones((self.target_len,1)) * int(target)
					vad_source = torch.ones((self.target_len,1))  
				elif isinstance(target, str):
					targets = np.array([int(float(i)) for i in target.split(',')])
					targets_idx_begin = int(signal_start / (self.target_fs / 10))
					targets = torch.from_numpy(targets[targets_idx_begin:targets_idx_begin+self.target_len,np.newaxis])
					vad_source = torch.ones((self.target_len,1))
				else:
					print(sig_path,target)
			# add noise following the SNR

			noise_path = self.noise_paths[rng.integers(low=0, high=len(self.noise_paths))]
			wav_info = sf.info(noise_path)
			wav_frames = wav_info.frames
			noise_begin_index =  rng.integers(low=0, high=wav_frames-(self.wav_use_len*self.input_fs))
			noise_end_index =  noise_begin_index + (self.wav_use_len*self.input_fs)
			noise_signal,noise_fs = self.load_noise(noise_path,begin_index=noise_begin_index,end_index=noise_end_index,use_mic_id=use_mic_id_item)
			if noise_fs != self.target_fs:
				noise_signal = self.resample(noise_signal,noise_fs,self.target_fs)
			coeff =  self.get_snr_coff(input_mic_signal,noise_signal,snr_item)
			try:
				assert coeff is not None
			except:
				coeff = 1.0
			noise_signal = coeff * noise_signal
			input_mic_signal += noise_signal
			# sf.write('./sample/' + str(idx)+'.wav',new_mic_signal,self.new_fs)
			#print(new_mic_signal.shape,labels.to(torch.float32).shape,vad_source.to(torch.float32).shape)
			array_topo = self.pos_mics[use_mic_id_item]
			# print(dp_vad)
			return input_mic_signal,targets.to(torch.float32),dp_vad.to(torch.float32),array_topo

		else:
			#sig_path = self.data_paths[idx]
			dp_sig_path = sig_path.replace('/ma_noisy_speech/','/dp_speech/')
			dp_signal,dp_fs = sf.read(dp_sig_path)
			if dp_fs != self.target_fs:
				dp_signal = self.resample(mic_signal=dp_signal,fs=dp_fs,new_fs=self.target_fs)
			dp_vad = self.cal_vad(dp_signal)

			load_path = sig_path.replace('ma_noisy_speech','ma_speech')
			input_mic_signal, fs = self.load_signals(load_path,use_mic_id=self.use_mic_id)
			if fs != self.target_fs:
				input_mic_signal = self.resample(mic_signal=input_mic_signal,fs=fs,new_fs=self.target_fs)
			len_signal = input_mic_signal.shape[0] / self.target_fs
			num_points = int(len_signal * 10)
			target = self.all_targets.at[sig_path.split('RealMAN/')[-1], 'angle(°)']
			if isinstance(target, float):
				targets = torch.ones((num_points,1)) * int(target)
			elif isinstance(target, str):
				targets = np.array([int(float(i)) for i in target.split(',')])
				targets = torch.from_numpy(targets[:,np.newaxis])
			vad_source =  torch.ones((targets.shape[0],1))
			array_topo = self.pos_mics[self.use_mic_id]
			if vad_source.shape[0] > dp_vad.shape[0]:
				vad_source[:dp_vad.shape[0],:] = dp_vad[:,:]
			else:
				vad_source = dp_vad[:vad_source.shape[0],:]
			return input_mic_signal, targets.to(torch.float32), vad_source.to(torch.float32),array_topo

