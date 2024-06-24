import numpy as np

from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d


def generate_late_rir_by_diffusion_model(
    Fs: int,
    T60: float,
    RIRs_early: np.ndarray,
    pos_src: np.ndarray,
    pos_rcv: np.ndarray,
    rng: np.random.Generator,
    length: int = None,  # T60 if None
    c: float = 343,
) -> np.ndarray:
    """use diffusion model to generate the late reverberation

    Args:
        Fs: the sample rate
        T60: the reverberation time T60
        RIRs_early: the early part of rir. [num_pos, num_mic, num_samples]
        pos_src: [num_pos, 3]
        pos_rcv: [num_mic, 3]
        length: the desired length of the RIRs
        c: the sound speed. Defaults to 343.

    Return:
        RIRs_full: [num_pos, num_mic, num_samples]
    """

    dist = np.sqrt(np.sum((pos_rcv[np.newaxis, :, :] - pos_src[:, np.newaxis, :])**2, axis=-1))  # [num_pos,num_mic]
    tau_dp = dist / c * Fs  # direct-path time, [num_pos, num_mic]
    # calculate the average power of the last samples
    nSamplesISM = RIRs_early.shape[-1]
    w_sz = 10e-3 * Fs  # Maximum window size (samples) to compute the final power of the early RIRs_early
    w_start = np.where(tau_dp > nSamplesISM - w_sz, tau_dp, nSamplesISM - w_sz)  # [num_pos, num_mic]
    w_start = np.ceil(w_start).astype(np.int32)
    w_center = (w_start + (nSamplesISM - w_start) / 2.0)
    finalPower = np.empty(dist.shape, dtype=np.float32)
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            finalPower[i, j] = np.sum(RIRs_early[i, j, w_start[i, j]:]**2) / (nSamplesISM - w_start[i, j])
    # estimate the amplitude parameter A (gpuRIR paper e.q. (8))
    alpha = -13.8155 / (T60 * Fs)  # -13.8155 == log(10^(-6))
    A = finalPower / np.exp(alpha * (w_center - tau_dp))  # [num_pos, num_mic]
    #
    length = int(np.ceil(T60 * Fs)) if length is None else length
    assert nSamplesISM < length, (nSamplesISM, T60, Fs)
    nSampleLate = length - nSamplesISM
    RIRs_full = np.empty(RIRs_early.shape[:-1] + (length,), dtype=RIRs_early.dtype)
    for i in range(RIRs_early.shape[0]):
        for j in range(RIRs_early.shape[1]):
            RIRs_full[i, j, :nSamplesISM] = RIRs_early[i, j, :]
            # Get logistic distribution from uniform distribution
            uniform = rng.uniform(size=(nSampleLate,))
            logistic = 0.551329 * np.log(uniform / (1.0 - uniform + 1e-6))  # ; // 0.551329 == sqrt(3)/pi
            # Apply power envelope
            pow_env = A[i, j] * np.exp(alpha * (nSamplesISM + np.arange(nSampleLate) - tau_dp[i, j]))
            RIRs_full[i, j, nSamplesISM:] = np.sqrt(pow_env) * logistic
    return RIRs_full


if __name__ == '__main__':
    path = Path('/home/quancs/projects/NBSS_pmt/dataset/datasets_nvme3/realistic_audio_rirs_9chn_p2')
    npzs = path.rglob('train/*.npz')
    for npz_file in npzs:
        rir_dict = dict(np.load(npz_file, allow_pickle=True))
        rirs = [np.load(npz_file.parent / rir_path) for rir_path in rir_dict['rir']]
        for i, rir in enumerate(rirs):
            T60, Fs = rir_dict['RT60'], rir_dict['fs']
            rir_full_gen = generate_late_rir_by_diffusion_model(
                Fs=Fs,
                T60=T60,
                RIRs_early=rir[:20, :, :int(np.ceil(T60 * Fs / 4))],
                pos_src=rir_dict['pos_src'].astype(rir_dict['pos_rcv'].dtype)[i][:20, ...],
                pos_rcv=rir_dict['pos_rcv'],
                rng=np.random.Generator(np.random.PCG64(100)),
            )
            fig = plt.figure(figsize=(10, 7))
            axes = fig.subplots(nrows=2, ncols=1)
            axes[0].clear()
            axes[0].plot(range(rir.shape[-1]), rir[0, 0], label='original')
            axes[0].plot(range(rir_full_gen.shape[-1]), rir_full_gen[0, 0], label='diffuse')
            axes[0].legend()
            # power envelope
            axes[1].clear()
            axes[1].plot(range(rir.shape[-1]), uniform_filter1d((rir[0, 0]**2).astype(np.float32), size=int(10e-3 * Fs)), label='original_env')
            axes[1].plot(range(rir_full_gen.shape[-1]), uniform_filter1d((rir_full_gen[0, 0]**2).astype(np.float32), size=int(10e-3 * Fs)), label='diffuse_env')
            axes[1].legend()
            axes[0].set_xlim(0, max(rir.shape[-1], rir_full_gen.shape[-1]))
            axes[1].set_xlim(0, max(rir.shape[-1], rir_full_gen.shape[-1]))
            plt.tight_layout()
            plt.show(block=True)
            plt.close()
        print()
