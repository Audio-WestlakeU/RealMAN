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


    
def select_microphone_array_for_enh(pos_mics, rng):
    center_mic = pos_mics[0, :]
    mic_id_list = set(np.arange(28))
    mic_id_list -= set([0])
    mic_id_list = list(mic_id_list)
    
    assert len(mic_id_list) == 27
    not_use_five_linear_mics = True
    while not_use_five_linear_mics:
        num_values_to_select = rng.integers(low=1, high=8)
        CH_list = list(rng.choice(mic_id_list, num_values_to_select, replace=False))
        mic_gemo = np.concatenate([[center_mic], pos_mics[CH_list, :]], axis=0)
        if num_values_to_select == 4:

            mic_gemo_sort = mic_gemo[mic_gemo[:, 0].argsort()]  
            distances = np.linalg.norm(mic_gemo_sort[1:] - mic_gemo_sort[:-1], axis=1)
            
            process_distance = np.array([round(distance, 2) for distance in distances])
            if (process_distance==0.03).all():
                not_use_five_linear_mics = True
            else:
                not_use_five_linear_mics = False
        else:
            not_use_five_linear_mics = False
    CH_list = [0] + CH_list
    assert len(CH_list) == mic_gemo.shape[0]
    return CH_list, mic_gemo


def audiowu_high_array_geometry() -> np.array:
    # the high-resolution mic array of the audio lab of westlake university
    R = 0.03
    pos_rcv = np.zeros((32, 3))
    pos_rcv[1:9, :] = circular_array_geometry(radius=R, mic_num=8)
    pos_rcv[9:17, :] = circular_array_geometry(radius=R * 2, mic_num=8)
    pos_rcv[17:25, :] = circular_array_geometry(radius=R * 3, mic_num=8)
    pos_rcv[25, :] = np.array([-R * 4, 0, 0])
    pos_rcv[26, :] = np.array([R * 4, 0, 0])
    pos_rcv[27, :] = np.array([R * 5, 0, 0])

    L = 0.045
    pos_rcv[28, :] = np.array([0, 0, L * 2])
    pos_rcv[29, :] = np.array([0, 0, L ])
    pos_rcv[30, :] = np.array([0, 0, -L])
    pos_rcv[31, :] = np.array([0, 0, -L * 2])
    return pos_rcv

  

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
        wav_chn_path = str(wav_path).replace('CH0', f'CH{chn_idx}')
        # wav_chn, sr = sf.read(wav_chn_path, start=start, stop=end, dtype='float32')  # [T]
        wav_chn, sr = torchaudio.load(wav_chn_path, frame_offset=start, num_frames=end-start, backend='sox') #[C,T]
        wav_chn = wav_chn[0,:]
        assert wav_chn.ndim == 1, wav_chn.shape
        if sr != sample_rate:
            wav_chn = resample_poly(wav_chn, up=sample_rate, down=sr, axis=0)
        wavs.append(wav_chn)  # [T]

    wav = np.stack(wavs, axis=0)  # [C,T]
    return wav, start, end


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



class RealisticAudioDataModule(LightningDataModule):

    def __init__(
        self,
        dataset_dir: str = '/data/home/RealisticAudio/RealMAN',
        noise_dir: List[Optional[str]] = [None, None, None, None],
        record_dir: List[Optional[str]] = [None, None, None, None],
        target_dir: List[Optional[str]] = [None, None, None, None],
        datasets: Tuple[str, str, str, str] = ['train', 'val', 'test', 'test'],  # datasets for train/val/test/predict
        audio_time_len: Tuple[Optional[float], Optional[float], Optional[float], Optional[float]] = [4.0, 4.0, None, None],  # audio_time_len (seconds) for train/val/test/predictS
        channels: List[int] = [0],
        spk_pattern: str = 'static',  # static or moving or all(*)
        sample_rate: int = 16000,
        noise_type: Literal['real', 'None'] = 'real',  # real or white+babble
        save_for_ASR: bool = False,  # save for ASR
        noise_proportion_to_test: float = 1.0,  # proportion of noise to add to test set
        use_microphone_array_generalization: bool = False,  # use microphone array generalization
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
        self.spk_pattern = spk_pattern
        self.save_dir = save_dir
        self.sample_rate = sample_rate
        self.noise_proportion_to_test = noise_proportion_to_test
        self.noise_type = noise_type
        self.noise_dir = noise_dir
        self.record_dir = record_dir
        self.target_dir = target_dir
        self.use_microphone_array_generalization = use_microphone_array_generalization
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

    def construct_dataloader(self, dataset, audio_time_len, seed, shuffle, batch_size, collate_fn, record_dir, target_dir, noise_dir):
        ds = RealisticAudioDataset(
            dataset=dataset,
            dataset_dir=self.dataset_dir,
            channels=self.channels,
            snr=self.snr,
            audio_time_len=audio_time_len,
            sample_rate=self.sample_rate,
            spk_pattern=self.spk_pattern,
            save_dir=self.save_dir,
            noise_proportion_to_test=self.noise_proportion_to_test,
            noise_type=self.noise_type,
            noise_dir=noise_dir,
            record_dir=record_dir,
            target_dir=target_dir,
            use_microphone_array_generalization=self.use_microphone_array_generalization,
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
            noise_dir=self.noise_dir[0],
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
            noise_dir=self.noise_dir[1],
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
            noise_dir=self.noise_dir[2],
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
            noise_dir=self.noise_dir[3],
        )


class RealisticAudioDataset(Dataset):

    def __init__(
            self,
            dataset: str,
            dataset_dir: str = '/data/home/RealisticAudio/RealMAN',
            noise_dir: Optional[str] = None,
            record_dir: Optional[str] = None,
            target_dir: Optional[str] = None,
            channels: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8],
            snr: Tuple[float, float] = [0, 15],
            audio_time_len: Optional[float] = None,
            sample_rate: int = 16000,
            spk_pattern: str = 'static',
            noise_type: Literal['real', 'None'] = 'real', 
            save_dir: Optional[str] = None,
            noise_proportion_to_test: float = 1.0,  # proportion of noise to add to test set
            use_microphone_array_generalization: bool = False,  # use microphone array generalization for train set
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
        self.noise_proportion_to_test = noise_proportion_to_test
        self.noise_type = noise_type
        self.use_microphone_array_generalization = use_microphone_array_generalization
        self.pos_mics = audiowu_high_array_geometry()
        self.save_for_ASR = save_for_ASR
        
        self.record_dir =  Path(record_dir).expanduser()
        self.target_dir = Path(target_dir).expanduser()
        self.noise_dir = Path(noise_dir).expanduser() if noise_dir is not None else None


        self.wav_pattern = self.spk_pattern + '/*/*CH0.flac' 
        self.uttrs = list(self.record_dir.rglob(self.wav_pattern))
        self.noises = list(self.noise_dir.rglob('*CH0.flac')) if self.noise_dir is not None else None
        self.target_uttrs = list(self.target_dir.rglob('*.flac'))
        self.uttrs.sort()
        if self.noises is not None:
            self.noises.sort()
        self.target_uttrs.sort()

        self.length = 20000 if self.dataset.startswith('train') else len(self.uttrs)
        self.snr = snr


    def __getitem__(self, index_seed: tuple[int, int]):
        # for each item, an index and seed are given. The seed is used to reproduce this dataset on any machines
        index, seed = index_seed
        rng = np.random.default_rng(np.random.PCG64(seed))

        idx = index % len(self.uttrs)
        # step 0: choose the microphone array
        if self.dataset == 'train' and self.use_microphone_array_generalization:
            self.channels, _ = select_microphone_array_for_enh(self.pos_mics, rng=rng)
        # step 1: load signals
        record, start, end = load_wav(self.uttrs[idx], self.sample_rate, self.channels, rng, self.audio_time_len)  # [C,T]
        # target, tsr = sf.read(self.target_dir / str(self.uttrs[idx]).removeprefix(str(self.record_dir) + '/').replace('_CH1', ''), start=start, stop=end, dtype='float32')  # [T]
        target, tsr = torchaudio.load(str(self.target_dir / str(self.uttrs[idx]).removeprefix(str(self.record_dir) + '/').replace('_CH0', '')), frame_offset=start, num_frames=end - start,
                                      backend='sox')  # [T]
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
        if self.noise_type == 'real':  # real noise
            # assert self.noise_type == 'real', self.noise_type
            if self.dataset != 'predict':
                if self.dataset == 'train':
                    noise_idx = rng.integers(low=0, high=len(self.noises))
                    noise_cands = self.noises
                else:  # val or test                   
                    if self.noise_dir is not None:
                        noise_idx = rng.integers(low=0, high=len(self.noises))
                        noise_cands = self.noises
                    else:
                        noise_idx = None
                        noise_cands = None

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
                
                else:
                    noise = None

            else:
                noise = None

        else:
            assert self.noise_type == 'None', ('unknown noise type', self.noise_type)
            noise = None

        # mix signal with noise
        if self.noise_type == 'real':
            if self.dataset != 'predict':
                if self.dataset == 'train':
                    snr = rng.uniform(low=self.snr[0], high=self.snr[1])
                    coeff = cal_coeff_for_adjusting_relative_energy(wav1=record, wav2=noise, target_dB=snr)
                    try:
                        assert coeff is not None
                    except:
                        coeff = 1.0
                    noise[:, :] *= coeff

                    noisy = record + noise
                else:  # val or test
                    if noise is not None:
                        noisy = record + self.noise_proportion_to_test * noise
                    else:
                        noisy = record

                    if self.save_test:
                        save_src_dir = Path(self.save_dir / str(self.uttrs[idx]).removeprefix(str(self.dataset_dir) + '/')).parent
                        save_tar_dir = Path(self.save_dir / str(self.uttrs[idx]).removeprefix(str(self.dataset_dir) + '/').replace('rec', 'tar')).parent
                        os.makedirs(save_src_dir, exist_ok=True)
                        os.makedirs(save_tar_dir, exist_ok=True)
                        for chn_idx in range(len(self.channels)):
                            sf.write(save_src_dir / str(self.uttrs[idx].stem + '.flac').replace('_CH0', f'_CH{self.channels[chn_idx]}'), noisy[chn_idx], self.sample_rate)
                        sf.write(save_tar_dir / str(self.uttrs[idx].stem + '.flac').replace('_CH0', '').replace('ma_speech', 'dp_speech'), target, self.sample_rate)
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
    dset = RealisticAudioDataset(
        dataset='val',  #, cv_dev93, test_eval92
        audio_time_len=None,
        dataset_dir='/data/home/RealisticAudio/RealMAN',
        noise_dir=None,
        record_dir='/data/home/RealisticAudio/RealMAN/val/ma_noisy_speech',
        target_dir='/data/home/RealisticAudio/RealMAN/val/dp_speech',
        spk_pattern='all',
        noise_type='real',
        use_microphone_array_generalization=True,
    )
    for i in range(dset.length):
        dset.__getitem__((i, i))
