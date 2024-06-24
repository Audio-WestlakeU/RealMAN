import os
from os.path import *
import random
from pathlib import Path
from typing import *

import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.rank_zero import (rank_zero_info, rank_zero_warn)

import numpy as np
import soundfile as sf
import torchaudio
import json
from data_loaders.utils.collate_func import default_collate_func
from data_loaders.utils.mix import *
from data_loaders.utils.my_distributed_sampler import MyDistributedSampler
from scipy.signal import resample_poly, correlate, correlation_lags
from data_loaders.utils.diffuse_noise import (gen_desired_spatial_coherence, gen_diffuse_noise)
from numpy.linalg import norm
import pdb


def load_wav(
    wav_path: str,
    sample_rate: int,
    channels: List[int],
    rng: np.random.Generator,
    audio_time_len: Optional[float] = None,
    frames_needed: Optional[int] = None,
):
    wav_info = sf.info(wav_path)

    if audio_time_len is not None or frames_needed is not None:
        assert wav_info.samplerate >= sample_rate, f"sample_rate {sample_rate} is higher than the original sample rate {wav_info.samplerate}"

        if audio_time_len is None:
            audio_time_len = frames_needed / sample_rate

        frames_needed = int(audio_time_len * wav_info.samplerate)
        wav_frames = wav_info.frames
        if frames_needed <= wav_frames:
            start = rng.integers(low=0, high=wav_frames - frames_needed + 1)
            end = start + frames_needed
        else:
            start = 0
            end = wav_frames
    else:
        start = 0
        end = -1

    wavs = []
    for chn_idx in channels:
        wav_chn_path = str(wav_path).replace('CH1', f'CH{chn_idx+1}')
        # wav_chn, sr = sf.read(wav_chn_path, start=start, stop=end, dtype='float32')  # [T]
        wav_chn, sr = torchaudio.load(wav_chn_path, frame_offset=start, num_frames=end-start) #[C,T]
        wav_chn = wav_chn[0,:]
        assert wav_chn.ndim == 1, wav_chn.shape
        if sr != sample_rate:
            wav_chn = resample_poly(wav_chn, up=sample_rate, down=sr, axis=0)
        wavs.append(wav_chn)  # [T]

    wav = np.stack(wavs, axis=0)  # [C,T]
    return wav, start, end

def find_corresponding_scene_noise(scene, speech_RT60_dict, noise_RT60_dict, num_cloest_scenes, noises_dir):
    speech_scene_RT60 = float(speech_RT60_dict[scene])
    closest = []
    for noise_scene, noise_RT60 in noise_RT60_dict.items():
        noise_RT60 = float(noise_RT60)
        diff = abs(noise_RT60 - speech_scene_RT60)

        closest.append((diff, noise_scene))

    closest.sort()

    noise_list = []
    noise_list.extend(list(noises_dir.rglob(f'{closest[0][1]}/*CH1.*')))
    diff_curr = closest[0][0]
    num_selected, idx = 1, 1

    while num_selected < num_cloest_scenes and idx < len(closest):
        noise_list.extend(list(noises_dir.rglob(f'{closest[idx][1]}/*CH1.*')))
        if closest[idx][0] != diff_curr:
            num_selected += 1
            diff_curr = closest[idx][0]
        idx += 1
    return noise_list

def find_corresponding_speech_SNR(uttr_dir, est_speech_snr_file):
    spk = uttr_dir.parent.name
    scene = uttr_dir.parent.parent.parent.name
    uttr_name = uttr_dir.name
    for i in range(len(est_speech_snr_file)):
        if est_speech_snr_file[i]['scene'] == scene and est_speech_snr_file[i]['speaker'] == spk:
            record_list = est_speech_snr_file[i]['source_basename_list']
            if uttr_name in record_list:
                uttr_idx = record_list.index(uttr_name)
                return float(est_speech_snr_file[i]['SNR_in_dB'][uttr_idx])


def normalize(vec: np.ndarray) -> np.ndarray:
    # get unit vector
    vec = vec / norm(vec)
    vec = vec / norm(vec)
    assert np.isclose(norm(vec), 1), 'norm of vec is not close to 1'
    return vec


def circular_array_geometry(radius: float, mic_num: int) -> np.ndarray:
    # 生成圆阵的拓扑（原点为中心），后期可以通过旋转、改变中心的位置来实现阵列位置的改变
    pos_rcv = np.empty((mic_num, 3))
    v1 = np.array([1, 0, 0])  # 第一个麦克风的位置（要求单位向量）
    v1 = normalize(v1)  # 单位向量
    # 将v1绕原点水平旋转angle角度，来生成其他mic的位置
    angles = np.arange(0, 2 * np.pi, 2 * np.pi / mic_num)
    for idx, angle in enumerate(angles):
        x = v1[0] * np.cos(angle) - v1[1] * np.sin(angle)
        y = v1[0] * np.sin(angle) + v1[1] * np.cos(angle)
        pos_rcv[idx, :] = normalize(np.array([x, y, 0]))
    # 设置radius
    pos_rcv *= radius
    return pos_rcv


def circular_cm_array_geometry(radius: float, mic_num: int) -> np.ndarray:
    # 圆形阵列+中心麦克风
    # circular array with central microphone
    pos_rcv = np.zeros((mic_num, 3))
    pos_rcv_c = circular_array_geometry(radius=radius, mic_num=mic_num - 1)
    pos_rcv[:-1, :] = pos_rcv_c
    return pos_rcv


class RealisticAudioDataModule(LightningDataModule):

    def __init__(
        self,
        dataset_dir: str = '/data/home/RealisticAudio/dataset',
        noise_dir: Optional[str] = None,
        record_dir: List[Optional[str]] = [None, None, None, None],
        target_dir: List[Optional[str]] = [None, None, None, None],
        datasets: Tuple[str, str, str, str] = ['train', 'val', 'test', 'test'],  # datasets for train/val/test/predict
        audio_time_len: Tuple[Optional[float], Optional[float], Optional[float], Optional[float]] = [4.0, 4.0, None, None],  # audio_time_len (seconds) for train/val/test/predictS
        channels: List[int] = [0],
        percision_mode: str = 'high',
        spk_pattern: str = 'static',  # static or moving
        noise_type: Literal['real', 'white+babble','None'] = 'real',  # real or white+babble
        use_rotate_noise: bool = False,  # use rotate noise
        save_for_ASR: bool = False,  # save for ASR
        noise_proportion_to_test: float = 1.0,  # proportion of noise to add to test set
        use_matching_RT60_for_train: bool = False,  # use matching RT60 for train set
        snr: Tuple[float, float] = [5, 10],  # SNR dB
        batch_size: List[int] = [1, 1],  # batch size for [train, val, {test, predict}]
        num_workers: int = 10,
        save_dir: Optional[str] = None,
        collate_func_train: Callable = default_collate_func,
        collate_func_val: Callable = default_collate_func,
        collate_func_test: Callable = default_collate_func,
        seeds: Tuple[Optional[int], int, int, int] = [None, 2, 3, 3],  # random seeds for train/val/test/predict sets
        # if pin_memory=True, will occupy a lot of memory & speed up
        pin_memory: bool = False,
        # prefetch how many samples, will increase the memory occupied when pin_memory=True
        prefetch_factor: int = 5,
        persistent_workers: bool = True,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.datasets = datasets
        self.audio_time_len = audio_time_len
        self.snr = snr
        self.persistent_workers = persistent_workers
        self.channels = channels
        self.percision_mode = percision_mode
        self.spk_pattern = spk_pattern
        self.save_dir = save_dir
        self.noise_proportion_to_test = noise_proportion_to_test
        self.use_matching_RT60_for_train = use_matching_RT60_for_train
        self.noise_type = noise_type
        self.noise_dir = noise_dir
        self.record_dir = record_dir
        self.target_dir = target_dir
        self.use_rotate_noise = use_rotate_noise
        self.save_for_ASR = save_for_ASR

        self.batch_size = batch_size
        while len(self.batch_size) < 4:
            self.batch_size.append(1)

        rank_zero_info("dataset: RealisticAudio")
        rank_zero_info(f'train/val/test/predict: {self.datasets}')
        rank_zero_info(f'batch size: train/val/test/predict = {self.batch_size}')
        rank_zero_info(f'audio_time_length: train/val/test/predict = {self.audio_time_len}')
        self.num_workers = num_workers

        self.collate_func = [collate_func_train, collate_func_val, collate_func_test, default_collate_func]

        self.seeds = []
        for seed in seeds:
            self.seeds.append(seed if seed is not None else random.randint(0, 1000000))

        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

    def setup(self, stage=None):
        self.current_stage = stage

    def construct_dataloader(self, dataset, audio_time_len, seed, shuffle, batch_size, collate_fn, record_dir, target_dir):
        ds = RealisticAudioDataset(
            dataset=dataset,
            dataset_dir=self.dataset_dir,
            channels=self.channels,
            snr=self.snr,
            audio_time_len=audio_time_len,
            sample_rate=16000,
            percision_mode=self.percision_mode,
            spk_pattern=self.spk_pattern,
            save_dir=self.save_dir,
            noise_proportion_to_test=self.noise_proportion_to_test,
            use_matching_RT60_for_train=self.use_matching_RT60_for_train,
            noise_type=self.noise_type,
            noise_dir=self.noise_dir,
            record_dir=record_dir,
            target_dir=target_dir,
            use_rotate_noise=self.use_rotate_noise,
            save_for_ASR=self.save_for_ASR,
        )

        return DataLoader(
            ds,
            sampler=MyDistributedSampler(ds, seed=seed, shuffle=shuffle),  #
            batch_size=batch_size,  #
            collate_fn=collate_fn,  #
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[0],
            audio_time_len=self.audio_time_len[0],
            seed=self.seeds[0],
            shuffle=True,
            batch_size=self.batch_size[0],
            collate_fn=self.collate_func[0],
            record_dir=self.record_dir[0],
            target_dir=self.target_dir[0],
        )

    def val_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[1],
            audio_time_len=self.audio_time_len[1],
            seed=self.seeds[1],
            shuffle=False,
            batch_size=self.batch_size[1],
            collate_fn=self.collate_func[1],
            record_dir=self.record_dir[1],
            target_dir=self.target_dir[1],
        )

    def test_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[2],
            audio_time_len=self.audio_time_len[2],
            seed=self.seeds[2],
            shuffle=False,
            batch_size=self.batch_size[2],
            collate_fn=self.collate_func[2],
            record_dir=self.record_dir[2],
            target_dir=self.target_dir[2],
        )

    def predict_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[3],
            audio_time_len=self.audio_time_len[3],
            seed=self.seeds[3],
            shuffle=False,
            batch_size=self.batch_size[3],
            collate_fn=self.collate_func[3],
            record_dir=self.record_dir[3],
            target_dir=self.target_dir[3],
        )


class RealisticAudioDataset(Dataset):

    def __init__(
            self,
            dataset: str,
            dataset_dir: str = '/data/home/RealisticAudio/dataset',
            raw_dir: str = '/data/home/RealisticAudio/raw_data',
            noise_dir: Optional[str] = None,
            record_dir: Optional[str] = None,
            target_dir: Optional[str] = None,
            percision_mode: str = 'high',
            channels: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 24],
            snr: Tuple[float, float] = [-5, 10],
            audio_time_len: Optional[float] = None,
            sample_rate: int = 16000,
            spk_pattern: str = 'static',
            noise_type: Literal['real', 'white+babble','None'] = 'real',  # real or white+babble
            save_dir: Optional[str] = None,
            noise_proportion_to_test: float = 1.0,  # proportion of noise to add to test set
            use_matching_RT60_for_train: bool = False,  # use matching RT60 for train set
            use_rotate_noise: bool = False,  # use rotate noise for train set
            save_for_ASR: bool = False,  
    ) -> None:
        """The RealisticAudio dataset

        Args:
            dataset_dir: a dir contains train/val/test/predict dirs
            dataset: train, val, test
            audio_time_len: cut the audio to `audio_time_len` seconds if given audio_time_len
        """
        super().__init__()
        assert dataset in ['train', 'val', 'test', 'predict'], dataset  # predict用于对齐

        self.dataset_dir = Path(dataset_dir).expanduser()
        self.dataset = dataset
        self.audio_time_len = audio_time_len
        self.sample_rate = sample_rate
        self.channels = channels
        self.save_test = True if (dataset == 'test' or 'val') and save_dir is not None else False
        self.save_dir = Path(save_dir).expanduser() if save_dir is not None else self.dataset_dir / 'test_mixed'
        self.spk_pattern = spk_pattern if spk_pattern in ['static', 'moving'] else '*'
        self.percision_mode = percision_mode
        self.noise_proportion_to_test = noise_proportion_to_test
        self.use_matching_RT60_for_train = use_matching_RT60_for_train
        self.RT60_dict = json.load(open('/home/quancs/projects/NBSS_pmt/dataset/datasets/RealisticDataset/formal_scene_info.json', 'r')) if self.use_matching_RT60_for_train else None
        self.noise_type = noise_type
        self.use_rotate_noise = use_rotate_noise
        self.save_for_ASR = save_for_ASR

        # find recorded speech signals
        self.record_dir = Path(os.path.join(self.dataset_dir, dataset, 'rec')).expanduser() if record_dir is None else Path(record_dir).expanduser()
        self.target_dir = Path(os.path.join(self.dataset_dir, dataset, 'tar')).expanduser() if target_dir is None else Path(target_dir).expanduser()
        self.noise_dir = Path(os.path.join(self.dataset_dir, dataset, 'noise', percision_mode)).expanduser() if noise_dir is None else Path(noise_dir).expanduser()

        # find the RT60 for each scene for noise or speech (train)
        if self.dataset == 'train' and self.use_matching_RT60_for_train:
            noise_scene_list = os.listdir(self.noise_dir)
            self.speech_RT60_dict = {}
            self.noise_RT60_dict = {}
            for scene in speech_scene_list:
                self.speech_RT60_dict.update({scene: self.RT60_dict[scene]['RT20']})
            for scene in noise_scene_list:
                self.noise_RT60_dict.update({scene: self.RT60_dict[scene]['RT20']})

        self.wav_pattern = self.spk_pattern + '/*/*CH1.*'
        self.uttrs = list(self.record_dir.rglob(self.wav_pattern))
        self.noises = list(self.noise_dir.rglob('*CH1.*'))
        self.target_uttrs = list(self.target_dir.rglob(self.wav_pattern))
        self.uttrs.sort()
        self.noises.sort()
        self.target_uttrs.sort()
        assert len(self.uttrs)>0, self.record_dir
        self.length = 20000 if self.dataset.startswith('train') else len(self.uttrs)
        self.snr = snr

        # load & save diffuse parameters
        pos_mics = circular_cm_array_geometry(0.03, 9)
        _, self.Cs = gen_desired_spatial_coherence(pos_mics=pos_mics, fs=self.sample_rate, noise_field='spherical', c=343, nfft=512)

        self.noises_babble = list(Path('/home/quancs/projects/NBSS_pmt/dataset/datasets/AISHELL').rglob('*.wav'))

        #
        self.factory_noises = [sf.read(Path(noise_path).expanduser(), dtype='float32', always_2d=False)[0] for noise_path in ['/home/quancs/projects/NBSS_pmt/dataset/datasets/RealisticDataset/Noise92/factory1.wav', '/home/quancs/projects/NBSS_pmt/dataset/datasets/RealisticDataset/Noise92/factory2.wav']]  # [T]
        self.factory_noise_sr = sf.info(Path('/home/quancs/projects/NBSS_pmt/dataset/datasets/RealisticDataset/Noise92/factory1.wav').expanduser()).samplerate
        self.factory_noises = [resample_poly(noise, up=self.sample_rate, down=self.factory_noise_sr, axis=0) for noise in self.factory_noises]

    def __getitem__(self, index_seed: tuple[int, int]):
        # for each item, an index and seed are given. The seed is used to reproduce this dataset on any machines
        index, seed = index_seed
        rng = np.random.default_rng(np.random.PCG64(seed))

        idx = index % len(self.uttrs)
        # step 1: load signals
        record, start, end = load_wav(self.uttrs[idx], self.sample_rate, self.channels, rng, self.audio_time_len)  # [C,T]
        # target, tsr = sf.read(self.target_dir / str(self.uttrs[idx]).removeprefix(str(self.record_dir) + '/').replace('_CH1', ''), start=start, stop=end, dtype='float32')  # [T]
        target, tsr = torchaudio.load(str(self.target_dir / str(self.uttrs[idx]).removeprefix(str(self.record_dir) + '/').replace('_CH1', '')), frame_offset=start, num_frames=end - start,
                                      )  # [T]
        target = target.squeeze().numpy()  # [T]
        if tsr != self.sample_rate:
            target = resample_poly(target, up=self.sample_rate, down=tsr, axis=0)
        assert target.shape[-1] == record.shape[-1], (target.shape, record.shape)

        ## pad or cut the signal
        start = 0
        if self.audio_time_len is not None:
            frames = int(self.audio_time_len * self.sample_rate)
            T = record.shape[-1]
            if frames > T:
                record = np.concatenate([record, np.zeros((record.shape[0], frames - T), dtype=np.float32)], axis=-1)
                target = np.concatenate([target, np.zeros(frames - T, dtype=np.float32)])
            else:
                start = rng.integers(low=0, high=T - frames + 1)
                record = record[:, start:start + frames]
                target = target[start:start + frames]

        # step 2: load noise
        if self.noise_type == 'white+babble':
            # babble, white
            noise_type = ['white', 'babble', 'factory'][rng.integers(low=0, high=3)]
            mix_frames = record.shape[-1]
            # generate diffuse noise
            if noise_type == 'babble':
                noises = []
                for i in range(9):
                    noise_i = np.zeros(shape=(mix_frames,), dtype=record.dtype)
                    for j in range(10):
                        noise_path = self.noises_babble[rng.integers(low=0, high=len(self.noises_babble))]
                        noise_ij, sr_noise = sf.read(noise_path, dtype='float32', always_2d=False)  # [T]
                        noise_ij = resample_poly(noise_ij, up=self.sample_rate, down=sr_noise, axis=0)
                        assert noise_ij.ndim == 1
                        noise_i += pad_or_cut([noise_ij], lens=[mix_frames], rng=rng)[0]
                    noises.append(noise_i)
                noise = np.stack(noises, axis=0).reshape(-1)
                noise = gen_diffuse_noise(noise=noise, L=mix_frames, Cs=self.Cs, nfft=512, rng=rng)  # shape [num_mic, mix_frames]
            elif noise_type == 'white':
                noise = rng.normal(size=record.shape[0] * record.shape[1])
                noise = gen_diffuse_noise(noise=noise, L=mix_frames, Cs=self.Cs, nfft=512, rng=rng)  # shape [num_mic, mix_frames]
            elif noise_type == 'factory':
                noise = self.factory_noises[rng.integers(low=0, high=len(self.factory_noises))]
                noise = pad_or_cut([noise], lens=[mix_frames*9], rng=rng)[0]  # [T*9]
                noise = gen_diffuse_noise(noise=noise, L=mix_frames, Cs=self.Cs, nfft=512, rng=rng)  # shape [num_mic, mix_frames]
            else:
                assert noise_type == 'point', ('unknown noise type', noise_type)
                noise = None

        elif self.noise_type == 'real':  # real noise
            # assert self.noise_type == 'real', self.noise_type
            if self.dataset != 'predict':
                if self.dataset == 'train':
                    if not self.use_matching_RT60_for_train:
                        noise_idx = rng.integers(low=0, high=len(self.noises))
                        noise_cands = self.noises
                    else:
                        scene = str(self.uttrs[idx].parent.parent.parent.name)
                        noise_cands = find_corresponding_scene_noise(scene, self.speech_RT60_dict, self.noise_RT60_dict, num_cloest_scenes=1, noises_dir=self.noise_dir)
                        noise_idx = rng.integers(low=0, high=len(noise_cands))
                else:  # val or test
                    # # temporal set
                    # noise_idx = rng.integers(low=0, high=len(self.noises))
                    # noise_cands = self.noises
                    noises_in_the_same_scene = list(Path(str(self.uttrs[idx].parent.parent.parent).replace('rec', 'noise')).rglob('*CH1.*')) if self.noise_dir is None else list(Path(os.path.join(self.noise_dir,self.uttrs[idx].parent.parent.parent.name)).rglob('*CH1.*'))
                    noises_in_the_same_scene.sort()
                    if len(noises_in_the_same_scene) != 0:
                        noise_idx = rng.integers(low=0, high=len(noises_in_the_same_scene))
                        noise_cands = noises_in_the_same_scene
                    else:  # if there is no noise in the same scene, did not add noise
                        noise_idx = None
                        noise = None

                if noise_idx is not None:
                    noise = np.zeros_like(record, dtype=np.float32)  # [C,T]

                    def load_noise(noise_cands, noise_idx, Tstart):
                        assert len(noise_cands) != 0, noise_cands
                        noise_path = noise_cands[noise_idx]
                        noise_tmp, start, end = load_wav(noise_path, self.sample_rate, self.channels, rng, None, record.shape[-1] - Tstart)  # [C,T]
                        noise[:, Tstart:Tstart + noise_tmp.shape[-1]] = noise_tmp[:, :]
                        if Tstart + noise_tmp.shape[-1] < noise.shape[-1]:
                            load_noise(noise_cands, (noise_idx + 1) % len(noise_cands), Tstart + noise_tmp.shape[-1])

                    load_noise(noise_cands, noise_idx, 0)

                    if self.use_rotate_noise:
                        assert noise.shape[0] == 9, noise.shape[0]
                        # 随机旋转圆阵
                        cir_chns = [1, 2, 3, 4, 5, 6, 7, 8]
                        cir_chns_new = []
                        i = rng.integers(0, 8)
                        while len(cir_chns_new) != len(cir_chns):
                            cir_chns_new.append(cir_chns[i])
                            i = (i + 1) % 8

                        noise = noise[[0] + cir_chns_new,:]
            else:
                noise = None

        else:
            assert self.noise_type == 'None', ('unknown noise type', self.noise_type)
            noise = None

        # mix signal with noise
        if self.noise_type == 'white+babble':
            if self.dataset == 'train':
                snr = rng.uniform(low=self.snr[0], high=self.snr[1])
                coeff = cal_coeff_for_adjusting_relative_energy(wav1=record, wav2=noise, target_dB=snr)
                if coeff is None:
                    coeff = 1.0
                noise[:, :] *= coeff
                noisy = record + noise
            else:  # val or test
                noisy = record
        elif self.noise_type == 'real':
            if self.dataset != 'predict':
                if self.dataset == 'train':
                    snr = rng.uniform(low=self.snr[0], high=self.snr[1])
                    coeff = cal_coeff_for_adjusting_relative_energy(wav1=record, wav2=noise, target_dB=snr)
                    try:
                        assert coeff is not None
                    except:
                        coeff = 1.0
                    noise[:, :] *= coeff
                    # snr_real = 10 * np.log10(np.sum(record**2) / np.sum(noise**2))
                    # assert np.isclose(snr, snr_real, atol=0.5), (snr, snr_real)
                    noisy = record + noise
                else:  # val or test
                    if noise is not None:
                        noisy = record + self.noise_proportion_to_test * noise
                    else:
                        noisy = record

                    if self.save_test:
                        # noise_energy = np.sum(noise**2) / noise.shape[-1]
                        # record_energy = np.sum(record**2) / record.shape[-1]
                        # record_snr = np.power(10, record_snr / 10)
                        # record_speech_energy = record_snr / (1 + record_snr) * record_energy
                        # record_noise_energy = record_energy - record_speech_energy
                        # final_snr = 10 * np.log10(record_speech_energy / noise_energy + record_noise_energy)
                        save_src_dir = Path(self.save_dir / str(self.uttrs[idx]).removeprefix(str(self.dataset_dir) + '/')).parent
                        save_tar_dir = Path(self.save_dir / str(self.uttrs[idx]).removeprefix(str(self.dataset_dir) + '/').replace('rec', 'tar')).parent
                        os.makedirs(save_src_dir, exist_ok=True)
                        os.makedirs(save_tar_dir, exist_ok=True)
                        for chn_idx in range(len(self.channels)):
                            sf.write(save_src_dir / str(self.uttrs[idx].stem + '.flac').replace('_CH1', f'_CH{self.channels[chn_idx]+1}'), noisy[chn_idx], self.sample_rate)
                        sf.write(save_tar_dir / str(self.uttrs[idx].stem + '.flac').replace('_CH1', '').replace('rec', 'tar'), target, self.sample_rate)
            else:
                noisy = record
        else:
            noisy = record

        # step 3: scale noisy and targets to [-0.9, 0.9]
        if self.dataset != 'predict':  # for predict, the original level should given
            scale_value = 0.9 / max(np.max(np.abs(noisy)), np.max(np.abs(target)))
            noisy[:, :] *= scale_value
            target[:] *= scale_value

        paras = {
            'index': index,
            'seed': seed,
            'saveto': [str(self.uttrs[idx]).removeprefix(str(self.record_dir) + '/')],
            'sample_rate': self.sample_rate,
            'dataset': f'RealisticAudio/{self.dataset}',
            'audio_time_len': self.audio_time_len,
            'num_spk': 1,
            'data': None,
            'save_for_ASR': self.save_for_ASR,
        }

        return (
            torch.as_tensor(noisy, dtype=torch.float32),  # shape [chn, time]
            torch.as_tensor(target.reshape(1, 1, target.shape[-1]), dtype=torch.float32),  # shape [spk, chn, time]
            paras,
        )

    def __len__(self):
        return self.length


if __name__ == '__main__':
    """python -m data_loaders.realistic_audio"""
    import numpy as np

    mic_list = list(np.arange(32))
    dset = RealisticAudioDataset(
        dataset='test',  #, cv_dev93, test_eval92
        audio_time_len=None,
        dataset_dir='/data/home/RealisticAudio/dataset_flac',
        percision_mode='high',
        spk_pattern='all',
        noise_type='real',
        channels = mic_list,
        # use_matching_RT60_for_train=True,
        # noise_proportion_to_test=0.0,
        save_dir='/data/home/RealisticAudio/dataset_val_test_0602',
    )
    for i in range(dset.length):
        dset.__getitem__((i, i))
