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
from data_loaders.utils.collate_func import default_collate_func
from data_loaders.utils.mix import *
from data_loaders.utils.diffuse_rir import generate_late_rir_by_diffusion_model
from data_loaders.utils.my_distributed_sampler import MyDistributedSampler
from scipy.signal import resample_poly


class AlignWavDataset(Dataset):

    def __init__(
        self,
        dataset: str,
        target: str,
        aishell_dir: str = '~/datasets/AISHELL',
        rir_dir: Union[str, List[str]] = '~/datasets/CHiME3_moving_rirs',
        noise_dir: str = '/nvmework3/raw/noise',
        num_chn: int = 1,
        chn_shuffle: bool = False,
        snr: Tuple[float, float] = [5, 10],
        max_shift: float = 0.1,  # seconds
        audio_time_len: Optional[float] = None,
        sample_rate: int = 8000,
        return_noise: bool = False,
        return_rvbt: bool = False,
        return_clean: bool = True,
    ) -> None:
        """The ShellAlign dataset

        Args:
            aishell_dir: a dir contains [FreeTalk, TTS0017]
            noise_dir: a dir contains the recorded noise signals
            rir_dir: a dir contains [train, validation, test, rir_cfg.npz]
            target: revb_image, direct_path
            dataset: train, val, test
            audio_time_len: cut the audio to `audio_time_len` seconds if given audio_time_len
        """
        super().__init__()
        assert target in ['revb_image', 'direct_path'] or target.startswith('RTS'), target
        assert dataset.startswith('train') or dataset.startswith('val') or dataset.startswith('test'), dataset

        if isinstance(rir_dir, str):
            self.rir_dirs = [Path(rir_dir).expanduser()]
        else:
            self.rir_dirs = [Path(rd).expanduser() for rd in rir_dir]

        self.speed = None  # e.g. 'moving(0.12,0.4)' means moving with a speed in 0.12 ~ 0.4 m/s
        if 'moving' in dataset:
            speed = dataset.split('_')[-1].replace('moving(', '').replace(')', '').split(',')
            assert len(speed) == 2 or len(speed) == 3, speed
            self.speed = [float(spd) for spd in speed[:2]]
            # e.g. moving(0.12,0.4,0.5) means: with a probability of 0.5, moving with a speed in 0.12 ~ 0.4 m/s; with p=0.5, not moving
            self.prob_moving = float(speed[2]) if len(speed) == 3 else 1
            rir_cfg = dict(np.load(self.rir_dirs[0] / 'rir_cfg.npz', allow_pickle=True))
            self.adjacent_points_distance = rir_cfg['args'].item()['trajectory'][1]
        else:
            self.prob_moving = 0

        self.dataset0 = dataset
        dataset = dataset.split('_')[0]
        self.num_chn = num_chn
        self.chn_shuffle = chn_shuffle
        self.target = target
        self.dataset = dataset
        self.audio_time_len = audio_time_len
        self.max_shift = max_shift
        self.sample_rate = sample_rate
        self.return_rvbt = return_rvbt
        self.return_noise = return_noise
        self.return_clean = return_clean

        # find clean speech signals
        self.aishell_dir = Path(aishell_dir).expanduser()
        spk_dir_train, spk_dir_val, spk_dir_test = [], [], []
        for d in [self.aishell_dir / 'FreeTalk', self.aishell_dir / 'TTS0017' / 'SPEECHDATA']:
            subds = list(d.glob('*'))
            subds.sort()
            if 'FreeTalk' in str(d):
                assert len(subds) == 39, subds
                subds = subds[:38]  # remove file "录音人信息.xls"
                spk_dir_train += subds  #[:30]
                spk_dir_val += subds  #[30:34]
                spk_dir_test += subds  #[34:38]
            else:
                assert len(subds) == 17, subds
                spk_dir_train += subds  #[:13]
                spk_dir_val += subds  #[13:15]
                spk_dir_test += subds  #[15:17]

        spk_dirs = {'train': spk_dir_train, 'val': spk_dir_val, 'test': spk_dir_test}[dataset]
        self.uttrs = []
        self.spk2uttrs = dict()
        for dir in spk_dirs:
            uttrs = list(dir.rglob("*.wav"))
            self.uttrs += uttrs

            spkid = dir.name
            if spkid not in self.spk2uttrs:
                self.spk2uttrs[spkid] = []
            self.spk2uttrs[spkid] += uttrs

        self.uttrs.sort()
        self.length = {'train': 20000, 'val': 2000, 'test': 2000}[dataset]

        # find noises
        self.noise_dir = Path(noise_dir).expanduser()
        self.noises_high = list(self.noise_dir.rglob('noise/*.high.wav'))  # for each noise, the first 80% is used for training, while the last two 10% are for validation and test
        self.noises_high.sort()
        self.noises_low = list(self.noise_dir.rglob('noise/*.low.wav'))  # for each noise, the first 80% is used for training, while the last two 10% are for validation and test
        self.noises_low.sort()
        for noise_path in self.noises_high:
            noise, _ = sf.read(noise_path, frames=48000 * 1, start=48000 * 10, dtype='float32', always_2d=True)
            res = [(noise[:, i] == 0).all() for i in range(noise.shape[1])]
            if np.all(res):
                self.noises_high.remove(noise_path)
        for noise_path in self.noises_low:
            noise, _ = sf.read(noise_path, frames=48000 * 1, start=48000 * 10, dtype='float32', always_2d=True)
            res = [(noise[:, i] == 0).all() for i in range(noise.shape[1])]
            if np.all(res):
                self.noises_low.remove(noise_path)
        self.noises = self.noises_high + self.noises_low
        self.noise_time_range = {'train': [0.0, 0.8], 'val': [0.8, 0.9], 'test': [0.9, 1.0]}[dataset]

        # find rirs
        self.shuffle_rir = True if dataset == "train" else False
        self.snr = snr
        self.rirs = []
        for rir_dir in self.rir_dirs:
            rir_dir = rir_dir / {"train": "train", "val": 'validation', 'test': 'test'}[dataset]
            self.rirs += [r for r in list(set(rir_dir.rglob('*.npz')) - set(rir_dir.rglob('*rir*.npz')))]
        self.rirs.sort()

        # system direct-path rir for low mics and high mics
        self.sirs, self.sir_sr = [], 48000
        for dppath in list(Path("~/datasets/RealisticDataset/dprirs").expanduser().rglob("*.wav")):
            s, r = sf.read(dppath, dtype='float32')
            assert r == 48000, r
            self.sirs.append(s)

    def __getitem__(self, index_seed: tuple[int, int]):
        # for each item, an index and seed are given. The seed is used to reproduce this dataset on any machines
        index, seed = index_seed

        rng = np.random.default_rng(np.random.PCG64(seed))

        num_spk = 1
        num_chn = self.num_chn
        # step 1: load single channel clean speech signals
        cleans, uttr_paths, cands = [], [], []
        for i in range(num_spk):
            uttr_paths.append(self.uttrs[rng.choice(range(len(self.uttrs)))])
            cands.append(self.spk2uttrs[Path(uttr_paths[i]).parent.name])
            wav, sr_src = sf.read(uttr_paths[i], dtype='float32')
            assert sr_src == 48000, sr_src
            wav = convolve1(wav, self.sirs[rng.integers(0, len(self.sirs))])  # convolve a dprir
            if sr_src != self.sample_rate:
                wav = resample_poly(wav, up=self.sample_rate, down=sr_src, axis=0)
            cleans.append(wav)

        # step 2: append signals if they are shorter than the length needed, then cut them to needed
        if self.audio_time_len is None:
            lens = [clean.shape[0] for clean in cleans]  # clean speech length of each speaker
            mix_frames = max(lens)
        else:
            mix_frames = int(self.audio_time_len * self.sample_rate)
            lens = [mix_frames] * len(cleans)

        for i, wav in enumerate(cleans):
            # repeat
            while len(wav) < lens[i]:
                wav2, fs = sf.read(rng.choice(cands[i], size=1)[0])
                if fs != self.sample_rate:
                    wav2 = resample_poly(wav2, up=self.sample_rate, down=fs, axis=0)
                wav = np.concatenate([wav, wav2])
            # cut to needed length
            if len(wav) > lens[i]:
                start = rng.integers(low=0, high=len(wav) - lens[i] + 1)
                wav = wav[start:start + lens[i]]
            cleans[i] = wav

        # step 3: load rirs
        if self.shuffle_rir:
            rir_this = self.rirs[rng.integers(low=0, high=len(self.rirs))]
        else:
            rir_this = self.rirs[index % len(self.rirs)]
        rir_dict = np.load(rir_this, allow_pickle=True)
        sr_rir = rir_dict['fs']
        assert sr_rir == self.sample_rate, (sr_rir, self.sample_rate)

        rir = rir_dict['rir']  # shape [nsrc,nmic,time]
        num_mic = rir_dict['pos_rcv'].shape[0]
        spk_rir_idxs = rng.choice(rir.shape[0], size=num_spk, replace=False).tolist()
        rir = rir[spk_rir_idxs]  # might be a path
        if isinstance(rir[0], str):  # mmap_mode='r' for fast partial loading
            rir = [np.load(rir_this.parent / rir_path, mmap_mode='r') for rir_path in rir]

        assert len(spk_rir_idxs) == num_spk, spk_rir_idxs
        if self.target == 'direct_path':  # read simulated direct-path rir
            rir_target = rir_dict['rir_dp']  # shape [nsrc,nmic,time] or [[nloc,nmic,time],...]
            rir_target = rir_target[spk_rir_idxs]
            if isinstance(rir_target[0], str):
                rir_target = [np.load(rir_this.parent / rir_path, mmap_mode='r') if rir_path.endswith('.npy') else np.load(rir_this.parent / rir_path)['arr'] for rir_path in rir_target]
        elif self.target == 'revb_image':  # rvbt_image
            rir_target = rir  # shape [nsrc,nmic,time] or [[nloc,nmic,time],...]
        else:
            raise NotImplementedError('Unknown target: ' + self.target)
        # sample channels
        if self.chn_shuffle:
            chns = rng.choice(list(range(num_mic)), size=num_chn, replace=False)
            rir = [riri[:, chns, :] for riri in rir]
            rir_target = [riri[:, chns, :] for riri in rir_target]
        else:
            assert num_chn == num_mic, (num_chn, num_mic)
            chns = list(range(num_mic))
        sel_chns = rir_dict['selected_channels'][chns]
        arr_geometry = rir_dict['arr_geometry']

        # step 4: convolve rir and clean speech
        # moving or not
        if self.prob_moving > 0 and self.prob_moving < 1:
            moving = True if rng.uniform() > self.prob_moving else False
        else:
            moving = False if self.speed is None else True
        # if moving speed is too slow, then set moving=False
        if moving:
            speed_this = rng.uniform(low=self.speed[0], high=self.speed[1], size=1) + 1e-5  # +1e-5 to fix the error: 0 m/s is sampled
            samples_per_rir = np.round(self.adjacent_points_distance / speed_this * sr_rir).astype(np.int32)
            if samples_per_rir >= mix_frames:
                moving = False
        # simulate
        if moving == False:
            if rir[0].ndim == 3:  # a trajectory, sample a point in the trajectory
                which_point = [rng.integers(low=0, high=rir_spk.shape[0]) for rir_spk in rir]
                rir = [rir_spk[[which_point[i]]] for i, rir_spk in enumerate(rir)]
                rir_target = [rir_spk[[which_point[i]]] for i, rir_spk in enumerate(rir_target)]
                pos_src = np.array([pos_src[[which_point[i]]] for i, pos_src in enumerate(rir_dict['pos_src'])])

            dp_len = np.max(np.argmax(rir_target, axis=-1))
            rirs = []
            for spk_idx, _ in enumerate(rir):
                rir = generate_late_rir_by_diffusion_model(
                    Fs=rir_dict['fs'],
                    T60=rir_dict['RT60'],
                    RIRs_early=rir[spk_idx],
                    pos_src=pos_src[spk_idx].astype(np.float32),
                    pos_rcv=rir_dict['pos_rcv'],
                    rng=rng,
                    length=int(rir_dict['RT60'] * rir_dict['fs']) + dp_len,
                )
                rirs.append(rir)
            # 只有一个位置，选择第0个位置的rir
            rvbts, targets = zip(*[convolve(wav=wav, rir=rir_spk[0], rir_target=rir_spk_t[0], ref_channel=0, align=True) for (wav, rir_spk, rir_spk_t) in zip(cleans, rirs, rir_target)])
        else:
            rvbts, targets = [], []
            for spk_idx, (wav, rir_spk, rir_spk_t, nsamp_spk) in enumerate(zip(cleans, rir, rir_target, samples_per_rir)):
                num_rirs = int(np.ceil(mix_frames / nsamp_spk)) + 1
                # sample indexes for rirs used for convolve_traj
                rir_idx_spk_cands = list(range(rir_spk.shape[0]))
                if rng.integers(low=0, high=2) == 0:
                    rir_idx_spk_cands.reverse()
                start = rng.integers(low=0, high=len(rir_idx_spk_cands))
                rir_idx_spk_sel = rir_idx_spk_cands[start:]
                while len(rir_idx_spk_sel) < num_rirs:
                    rir_idx_spk_sel += rir_idx_spk_cands
                rir_idx_spk_sel = rir_idx_spk_sel[:num_rirs]

                # sample rir
                rir_spk_t = rir_spk_t[rir_idx_spk_sel]
                # load samples before RT15 only, then generate the samples after RT15 using diffusion model
                dp_len = np.max(np.argmax(rir_spk_t, axis=-1))
                # n_sample_RT15 = int(rir_dict['RT60'] / 4 * rir_dict['fs']) + dp_len
                rir_spk = rir_spk[rir_idx_spk_sel]
                rir_spk = generate_late_rir_by_diffusion_model(
                    Fs=rir_dict['fs'],
                    T60=rir_dict['RT60'],
                    RIRs_early=rir_spk,
                    pos_src=rir_dict['pos_src'].astype(rir_dict['pos_rcv'].dtype)[spk_idx][rir_idx_spk_sel],
                    pos_rcv=rir_dict['pos_rcv'],
                    rng=rng,
                    length=int(rir_dict['RT60'] * rir_dict['fs']) + dp_len,
                )

                # convolve_traj
                rvbts_i = convolve_traj_with_win(wav=wav, traj_rirs=rir_spk, samples_per_rir=nsamp_spk, wintype='trapezium20')
                targets_i = convolve_traj_with_win(wav=wav, traj_rirs=rir_spk_t, samples_per_rir=nsamp_spk, wintype='trapezium20')
                rvbts_i, targets_i = align(rir=rir_spk[0, 0], rvbt=rvbts_i, target=targets_i, src=wav)
                rvbts.append(rvbts_i), targets.append(targets_i)
        rvbts, targets = np.stack(rvbts, axis=0), np.stack(targets, axis=0)

        # step 5: add a random time shift between clean & rvbts & targets
        shift_len = rng.integers(low=-int(self.max_shift * self.sample_rate), high=int(self.max_shift * self.sample_rate))

        def shift(arr: np.ndarray, target_len: int, shift_len: int):
            assert shift_len >= 0, shift_len
            arr_new = np.zeros(list(arr.shape[:-1]) + [target_len])
            arr_new[..., shift_len:shift_len + arr.shape[-1]] = arr
            return arr_new

        mix_frames = mix_frames + int(self.max_shift * self.sample_rate)

        clean = np.stack(cleans, axis=0)
        if shift_len >= 0:
            clean = shift(clean, mix_frames, shift_len)
            rvbts = shift(rvbts, mix_frames, 0)
            targets = shift(targets, mix_frames, 0)
        elif shift_len < 0:
            clean = shift(clean, mix_frames, 0)
            rvbts = shift(rvbts, mix_frames, -shift_len)
            targets = shift(targets, mix_frames, -shift_len)

        # step 6: load noise and mix with a sampled SNR
        mix = np.sum(rvbts, axis=0)  # sum speakers
        if 'high' in arr_geometry:
            noise_path = self.noises_high[rng.integers(low=0, high=len(self.noises_high))]
        elif 'low' in arr_geometry:
            noise_path = self.noises_low[rng.integers(low=0, high=len(self.noises_low))]
        else:
            noise_path = self.noises[rng.integers(low=0, high=len(self.noises))]
        noise_info = sf.info(noise_path)
        noise_frames = noise_info.duration * noise_info.samplerate
        noise_start, noise_end = int(self.noise_time_range[0] * noise_frames), int(self.noise_time_range[1] * noise_frames)

        assert int(noise_info.samplerate / self.sample_rate) * self.sample_rate == noise_info.samplerate, (self.sample_rate, noise_info.samplerate)
        noise_frames_needed = mix_frames * int(noise_info.samplerate / self.sample_rate)
        start = rng.integers(low=noise_start, high=noise_end - noise_frames_needed) if (noise_end - noise_start) > noise_frames_needed else noise_start

        noise, _ = sf.read(noise_path, frames=noise_frames_needed, start=start, dtype='float32', always_2d=True)
        if num_chn == 1 or self.chn_shuffle:
            noise = noise[:, sel_chns].T
        else:
            assert num_chn == 9, num_chn
            # 随机旋转圆阵
            cir_chns = [0, 1, 2, 3, 4, 5, 6, 7]
            cir_chns_new = []
            i = rng.integers(0, 8)
            while len(cir_chns_new) != len(cir_chns):
                cir_chns_new.append(cir_chns[i])
                i = (i + 1) % 8
            # 中心+圆阵
            if noise_path.name.endswith('.high.wav'):
                noise = noise[:, [24] + cir_chns_new].T
            else:
                noise = noise[:, [8] + cir_chns_new].T

        if noise_info.samplerate != self.sample_rate:
            noise = resample_poly(noise, up=self.sample_rate, down=noise_info.samplerate, axis=-1)

        snr_this = rng.uniform(low=self.snr[0], high=self.snr[1])
        coeff = cal_coeff_for_adjusting_relative_energy(wav1=mix, wav2=noise, target_dB=snr_this)
        if coeff is None and (noise == 0).all():
            return self.__getitem__((index, seed + 1))
        assert coeff is not None
        noise *= coeff
        snr_real = 10 * np.log10(np.sum(mix**2) / np.sum(noise**2))
        assert np.isclose(snr_this, snr_real, atol=0.5), (snr_this, snr_real)
        mix[:, :] = mix + noise

        # scale mix and targets to [-0.9, 0.9]
        scale_value = 0.9 / max(np.max(np.abs(mix)), np.max(np.abs(targets)))
        mix[:, :] *= scale_value
        targets[:, :] *= scale_value
        noise = noise * scale_value if self.return_noise else None
        rvbts = rvbts * scale_value if self.return_rvbt else None

        paras = {
            'index': index,
            'seed': seed,
            'saveto': [f'{index}.wav'],
            'target': self.target,
            'sample_rate': self.sample_rate,
            'dataset': f'AlignWav/{self.dataset0}',
            'snr': float(snr_real),
            'audio_time_len': self.audio_time_len,
            'num_spk': num_spk,
            'rir': {
                'RT60': rir_dict['RT60'],
                'pos_src': rir_dict['pos_src'],
                'pos_rcv': rir_dict['pos_rcv'],
            },
            'data': {
                'rir': rir,
                'noise': noise,
                'rvbt': rvbts,
            }
        }

        if self.return_clean:
            return (
                torch.as_tensor(mix, dtype=torch.float32),  # shape [chn, time]
                torch.as_tensor(targets, dtype=torch.float32),  # shape [spk, chn, time]
                torch.as_tensor(clean, dtype=torch.float32),  # shape [spk, time]
                paras,
            )
        else:  # for training enhancement networks which don't accept clean as input
            return (
                torch.as_tensor(mix, dtype=torch.float32),  # shape [chn, time]
                torch.as_tensor(targets, dtype=torch.float32),  # shape [spk, chn, time]
                paras,
            )

    def __len__(self):
        return self.length


class AlignWavDataModule(LightningDataModule):

    def __init__(
        self,
        aishell_dir: str = '~/datasets/AISHELL',
        rir_dir: Union[str, List[str]] = '~/datasets/CHiME3_moving_rirs',
        noise_dir: str = '/nvmework4/raw/noise',
        num_chn: int = 1,
        chn_shuffle: bool = False,
        sample_rate: int = 48000,
        target: str = "direct_path",  # e.g. rvbt_image, direct_path
        datasets: Tuple[str, str, str, str] = ['train', 'val', 'test', 'test'],  # datasets for train/val/test/predict
        audio_time_len: Tuple[Optional[float], Optional[float], Optional[float], Optional[float]] = [4.0, 4.0, None, None],  # audio_time_len (seconds) for train/val/test/predictS
        snr: Tuple[float, float] = [5, 10],  # SNR dB
        max_shift: float = 0.1,  # seconds
        return_noise: bool = False,
        return_rvbt: bool = False,
        return_clean: bool = True,
        batch_size: List[int] = [1, 1],  # batch size for [train, val, {test, predict}]
        num_workers: int = 10,
        collate_func_train: Callable = default_collate_func,
        collate_func_val: Callable = default_collate_func,
        collate_func_test: Callable = default_collate_func,
        seeds: Tuple[Optional[int], int, int, int] = [None, 2, 3, 3],  # random seeds for train/val/test/predict sets
        # if pin_memory=True, will occupy a lot of memory & speed up
        pin_memory: bool = False,
        # prefetch how many samples, will increase the memory occupied when pin_memory=True
        prefetch_factor: int = 10,
        persistent_workers: bool = True,
    ):
        super().__init__()
        self.aishell_dir = aishell_dir
        self.rir_dir = rir_dir
        self.noise_dir = noise_dir
        self.num_chn = num_chn
        self.chn_shuffle = chn_shuffle
        self.sample_rate = sample_rate
        self.target = target
        self.datasets = datasets
        self.audio_time_len = audio_time_len
        self.snr = snr
        self.max_shift = max_shift
        self.return_noise = return_noise
        self.return_rvbt = return_rvbt
        self.return_clean = return_clean
        self.persistent_workers = persistent_workers

        self.batch_size = batch_size
        while len(self.batch_size) < 4:
            self.batch_size.append(1)

        rank_zero_info("dataset: AlignWav")
        rank_zero_info(f'train/val/test/predict: {self.datasets}')
        rank_zero_info(f'batch size: train/val/test/predict = {self.batch_size}')
        rank_zero_info(f'audio_time_length: train/val/test/predict = {self.audio_time_len}')
        rank_zero_info(f'target: {self.target}')

        self.num_workers = num_workers

        self.collate_func = [collate_func_train, collate_func_val, collate_func_test, default_collate_func]

        self.seeds = []
        for seed in seeds:
            self.seeds.append(seed if seed is not None else random.randint(0, 1000000))

        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

    def setup(self, stage=None):
        self.current_stage = stage

    def construct_dataloader(self, dataset, audio_time_len, seed, shuffle, batch_size, collate_fn):
        ds = AlignWavDataset(
            dataset=dataset,
            target=self.target,
            aishell_dir=self.aishell_dir,
            rir_dir=self.rir_dir,
            noise_dir=self.noise_dir,
            num_chn=self.num_chn,
            chn_shuffle=self.chn_shuffle,
            snr=self.snr,
            max_shift=self.max_shift,
            audio_time_len=audio_time_len,
            sample_rate=self.sample_rate,
            return_noise=self.return_noise,
            return_rvbt=self.return_rvbt,
            return_clean=self.return_clean,
        )

        return DataLoader(
            ds,
            sampler=MyDistributedSampler(ds, seed=seed, shuffle=shuffle),  #
            batch_size=batch_size,  #
            collate_fn=collate_fn,  #
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )

    def train_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[0],
            audio_time_len=self.audio_time_len[0],
            seed=self.seeds[0],
            shuffle=True,
            batch_size=self.batch_size[0],
            collate_fn=self.collate_func[0],
        )

    def val_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[1],
            audio_time_len=self.audio_time_len[1],
            seed=self.seeds[1],
            shuffle=False,
            batch_size=self.batch_size[1],
            collate_fn=self.collate_func[1],
        )

    def test_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[2],
            audio_time_len=self.audio_time_len[2],
            seed=self.seeds[2],
            shuffle=False,
            batch_size=self.batch_size[2],
            collate_fn=self.collate_func[2],
        )

    def predict_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[3],
            audio_time_len=self.audio_time_len[3],
            seed=self.seeds[3],
            shuffle=False,
            batch_size=self.batch_size[3],
            collate_fn=self.collate_func[3],
        )


if __name__ == '__main__':
    """python -m data_loaders.align_wav"""
    dset = AlignWavDataset(
        target='direct_path',
        dataset='train',  #, cv_dev93, test_eval92
        rir_dir='/nvmework1/quancs/datasets/realistic_audio_rirs_9chn_16k',
        noise_dir='/nvmework4/raw/noise',
        audio_time_len=4,
        num_chn=9,
        sample_rate=16000,
    )
    for i in range(1):
        dset.__getitem__((i, i))

    from jsonargparse import ArgumentParser
    parser = ArgumentParser("")
    parser.add_class_arguments(AlignWavDataModule, nested_key='data')
    parser.add_argument('--save_dir', type=str, default='dataset')
    parser.add_argument('--dataset', type=str, default='train')
    parser.add_argument('--gen_unprocessed', type=bool, default=True)
    parser.add_argument('--gen_target', type=bool, default=True)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if not args.gen_unprocessed and not args.gen_target:
        exit()

    args_dict = args.data
    args_dict['num_workers'] = 20  # 0 for debuging
    args_dict['datasets'] = ['train_moving(0.0,0.4)', 'val_moving(0.0,0.4)', 'test_moving(0.0,0.4)', 'test_moving(0.0,0.4)']
    args_dict['rir_dir'] = [
        '/nvmework1/quancs/datasets/realistic_audio_rirs_9chn',
        # '/nvmework1/quancs/datasets/realistic_audio_rirs_9chn_p1',
        # '/nvmework3/quancs/datasets/realistic_audio_rirs_9chn_p2',
        # '/home/quancs/datasets/realistic_audio_rirs_9chn_p3',
    ]
    args_dict['audio_time_len'] = [4.0, 4.0, 4.0, 4.0]
    args_dict['return_noise'] = True
    args_dict['sample_rate'] = 48000
    args_dict['num_chn'] = 9
    args_dict['noise_dir'] = '/nvmework4/raw/noise'
    datamodule = AlignWavDataModule(**args_dict)
    datamodule.setup()

    if args.dataset.startswith('train'):
        dataloader = datamodule.train_dataloader()
    elif args.dataset.startswith('val'):
        dataloader = datamodule.val_dataloader()
    elif args.dataset.startswith('test'):
        dataloader = datamodule.test_dataloader()
    else:
        assert args.dataset.startswith('predict'), args.dataset
        dataloader = datamodule.predict_dataloader()

    if type(dataloader) != dict:
        dataloaders = {args.dataset: dataloader}
    else:
        dataloaders = dataloader

    for ds, dataloader in dataloaders.items():

        for idx, (noisy, tar, clean, paras) in enumerate(dataloader):
            print(f'{idx}/{len(dataloader)}', end=' ')
            if idx > 20:
                continue
            # write target to dir
            if args.gen_target and not args.dataset.startswith('predict'):
                tar_path = Path(f"{args.save_dir}/{paras[0]['dataset']}/target").expanduser()
                tar_path.mkdir(parents=True, exist_ok=True)
                cln_path = Path(f"{args.save_dir}/{paras[0]['dataset']}/clean").expanduser()
                cln_path.mkdir(parents=True, exist_ok=True)
                assert np.max(np.abs(tar[0, :, 0, :].numpy())) <= 1
                for spk in range(tar.shape[1]):
                    sp = tar_path / basename(paras[0]['saveto'][spk])
                    if not sp.exists():
                        sf.write(sp, tar[0, spk, 0, :].numpy(), samplerate=paras[0]['sample_rate'])
                    sp = cln_path / basename(paras[0]['saveto'][spk])
                    if not sp.exists():
                        sf.write(sp, clean[0, spk, :].numpy(), samplerate=paras[0]['sample_rate'])

            # write unprocessed's 0-th channel
            if args.gen_unprocessed:
                tar_path = Path(f"{args.save_dir}/{paras[0]['dataset']}/noisy").expanduser()
                tar_path.mkdir(parents=True, exist_ok=True)
                assert np.max(np.abs(noisy[0, 0, :].numpy())) <= 1
                for spk in range(len(paras[0]['saveto'])):
                    sp = tar_path / basename(paras[0]['saveto'][spk])
                    if not sp.exists():
                        sf.write(sp, noisy[0, 0, :].numpy(), samplerate=paras[0]['sample_rate'])

            # # write noise
            # if paras[0]['data']['noise'] is not None:
            #     tar_path = Path(f"{args.save_dir}/{paras[0]['dataset']}/noise").expanduser()
            #     tar_path.mkdir(parents=True, exist_ok=True)
            #     sp = tar_path / basename(paras[0]['saveto'][0])
            #     sf.write(sp, paras[0]['data']['noise'], samplerate=paras[0]['sample_rate'])

            print(noisy.shape, None if args.dataset.startswith('predict') else tar.shape, paras)
