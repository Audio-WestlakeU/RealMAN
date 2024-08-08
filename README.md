<div align=center>
<img src=images/realman_logo_v1.png width="400"/>
 
**A Real-Recorded and Annotated Microphone Array Dataset for Dynamic Speech Enhancement and Localization** 

<span style="font-size:11px;">
<a href="https://arxiv.org/abs/2406.19959" target="_blank">**arXiv paper**</a> by
<a href="https://audio.westlake.edu.cn/" target="_blank">AudioLab at Westlake University</a> &
<a href="https://www.aishelltech.com/" target="_blank"> AIShell Technology Co. Ltd</a> 

</span>
<!-- The details of the RealMAN dataset are described in the following paper: -->
</div>

---
### Introduction
***Motivation:***
The training of deep learning-based multichannel speech enhancement and source localization systems relies heavily on the simulation of room impulse response and multichannel diffuse noise, due to the lack of large-scale real-recorded datasets. However, the acoustic mismatch between simulated and real-world data could degrade the model performance when applying in real-world scenarios. To bridge this simulation-to-real gap, we presents a new relatively large-scale real-recorded and annotated dataset. 

***Description:*** 
The Audio Signal and Information Processing Lab at Westlake University, in collaboration with AISHELL, has released the **Real**-recorded and annotated **M**icrophone **A**rray speech&**N**oise (**RealMAN**) dataset, which provides annotated multi-channel speech and noise recordings for dynamic speech enhancement and localization:
- A 32-channel array with high-fidelity microphones is used for recording
- A loudspeaker is used for playing source speech signals
- A total of 83-hour speech signals (48 hours for static speaker and 35 hours for moving speaker) are recorded in 32 different scenes, and 144 hours of background noise are recorded in 31 different scenes
- Both speech and noise recording scenes cover various common indoor, outdoor, semi-outdoor and transportation environments
- The azimuth angle of the loudspeaker is annotated with an omni-direction fisheye camera, and is used for the training of source localization networks
- The direct-path signal is obtained by filtering the played speech signal with an estimated direct-path propagation filter, and is used for the training of speech enhancement networks

***Baseline demonstration:***
- compared to using simulated data, the proposed dataset is indeed able to train better speech enhancement and source localization networks
- using various sub-arrays of the proposed 32-channel microphone array can successfully train variable-array networks that can be directly used to unseen arrays 

***Importance:***
- Benchmark speech enhancement and localization algorithms in real scenarios
- Offer a substantial amount of real-world training data for potentially improving the performance of real-world applications



<div align=center>
<img src=images/devices.png width="700"/>
</div>


### Download

To download the entire dataset, you can access: 
<a href="https://mab.to/uFs0WNo0hgrV6/us3" target="_blank">Origninal data page</a> or 
<a href="https://www.aishelltech.com/RealMAN" target="_blank">AISHELL page</a>. 
The dataset comprises the following components:

| File | Size | Description |
| -------- | -- | -- |
| `train.rar` | 521.76 GB | The training set consisting of 36.6 hours of static speaker speech and 26.6 hours of moving speaker speech  (`ma_speech`), 106.3 hours of noise recordings (`ma_noise`), 0-channel direct path speech (`dp_speech`) and sound source location (`train_*_source_location.csv`). |
| `val_raw.rar` | 65.57 GB | The raw validation set consisting of 4.5 hours of static speaker speech and 3.3 hours of moving speaker speech (`ma_speech`),  16.0 hours of noise recordings (`ma_noise`), 0-channel direct path speech (`dp_speech`) and sound source location (`val_*_source_location.csv`). |
| `val.rar` | 25.57 GB | The validation set consisting of mixed noisy speech recordings (`ma_noise`), 0-channel direct path speech (`dp_speech`), sound source location (`val_*_source_location.csv`). |
| `test_raw.rar` | 91.75 GB | The raw test set consisting of 6.9 hours of static speaker speech and 4.8 hours of moving speaker speech (`ma_speech`),  22.2 hours of noise recordings (`ma_noise`), 0-channel direct path speech (`dp_speech`) and sound source location (`test_*_source_location.csv`). |
| `test.rar` | 38.02 GB | The test set consisting of mixed noisy speech recordings (`ma_noise`), 0-channel direct path speech (`dp_speech`), sound source location (`test_*_source_location.csv`). |
| `dataset_info.rar` | 127.9 MB | The dataset information file including scene photos, scene information (T60, recording duration, etc), and speaker information |
| `transcriptions.trn` | 2.4 MB | The transcription file of speech for the dataset |



<!-- ```
### Download Scripts

The download scripts for the RealMAN dataset are available in the `download_scripts` directory. The scripts can be used to download the entire dataset or individual recordings. The scripts use the `wget` command to download the files from the Google Drive links provided in the dataset description.

To download the entire dataset, run the following command:

```
./download_all.sh
```

To download a specific recording, run the following command:

```
./download_recording.sh device_recording_date_location_microphone_array_version
```

For example, to download the recording from the SM1 device recorded on January 1st, 2020 in Berlin, run the following command:

```
./download_recording.sh RealMAN_SM1_2020-01-01_Berlin_SM1_v1
```
``` -->

The dataset is organized into the following directory structure:

```
RealMAN
├── transcriptions.trn
├── dataset_info
│   ├── scene_images
│   ├── scene_info.json
│   └── speaker_info.csv
└── train|val|test|val_raw|test_raw
    ├── train_moving_source_location.csv
    ├── train_static_source_location.csv
    ├── dp_speech
    │   ├── BadmintonCourt2
    │   │   ├── moving
    │   │   │   ├── 0010
    │   │   │   │   ├── TRAIN_M_BAD2_0010_0003.flac
    │   │   │   │   └── ...
    │   │   │   └── ...
    │   │   └── static
    │   └── ...
    ├── ma_speech|ma_noisy_speech
    │   ├── BadmintonCourt2
    │   │   ├── moving
    │   │   │   ├── 0010
    │   │   │   │   ├── TRAIN_M_BAD2_0010_0003_CH0.flac
    │   │   │   │   └── ...
    │   │   │   └── ...
    │   │   ├── static
    │   └── ...
    └── ma_noise
```

The naming convention is as follows:

```
# Recorded Signal
[TRAIN|VAL|TEST]_[M|S]_scene_speakerId_utteranceId_channelId.flac

# Direct-Path Signal
[TRAIN|VAL|TEST]_[M|S]_scene_speakerId_utteranceId.flac

# Source Location
[train|val|test]_[moving|static]_source_location.csv
```


### Baseline
- <a href="https://github.com/Audio-WestlakeU/RealMAN/tree/main/baselines/SE" target="_blank">Speech enhancement</a>
- <a href="https://github.com/Audio-WestlakeU/RealMAN/tree/main/baselines/SSL" target="_blank">Speech source localization</a>


### License

The dataset is licensed under the Creative Commons Attribution 4.0 International (**CC-BY-4.0**) license. 

<!-- 
### Citation
To attribute this work, please use the following citation format:
```
@Article{RealMAN2024,
  author = "Bing Yang and Changsheng Quan and Yabo Wang and Pengyu Wang and Yujie Yang and Ying Fang and Nian Shao and Hui Bu and Xin Xu and Xiaofei Li",
  title = "RealMAN: A Real-Recorded and Annotated Microphone Array Dataset for Dynamic Speech Enhancement and Localization",
  journal = "",
  year = "2024",
}
```
 -->
