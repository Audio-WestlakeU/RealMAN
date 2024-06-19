## RealMAN: A Real-Recorded and Annotated Microphone Array Dataset for Dynamic Speech Enhancement and Localization


### Description
The **Real**-recorded and annotated **M**icrophone **A**rray speech&**N**oise (**RealMAN**) dataset provides annotated multi-channel speech and noise recordings for dynamic speech enhancement and localization:
- A 32-channel array with high-fidelity microphones is used for recording
- A loudspeaker is used for playing source speech signals
- A total of 83-hour speech signals (48 hours for static speaker and 35 hours of moving speaker) are recorded in 32 different scenes, and 144 hours of background noise are recorded in 31 different scenes
- Both speech and noise recording scenes cover various common indoor, outdoor, semi-outdoor and transportation environments
- The azimuth angle of the loudspeaker is annotated with an omni-direction fisheye camera, and is used for the training of source localization networks
- The direct-path signal is obtained by filtering the played speech signal with an estimated direct-path propagation filter, and is used for the training of speech enhancement networks.

The RealMAN dataset is valuable in two aspects:
- Benchmark speech enhancement and localization algorithms in real scenarios
- Offer a substantial amount of real-world training data for potentially improving the performance of real-world applications

The details of the RealMAN dataset are described in the following paper: [<a href="" target="_blank">arXiv</a>]


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
| <a href="https://www.myairbridge.com/dl-service-3eaa53cc/dl.66727f46e788b5.94756052" target="_blank">train.rar</a> | 521.76 GB | The training set consisting of 48 hours of static speaker speech and 35 hours of moving speaker speech |
| <a href="https://www.myairbridge.com/dl-service-3eaa53cc/dl.66727f7eac7a64.56207586" target="_blank">val.rar</a> | 25.57 GB | The validation set consisting of 12 hours of static speaker speech and 9 hours of moving speaker speech |
| <a href="https://www.myairbridge.com/dl-service-3eaa53cc/dl.66727f9cd14af4.31130467" target="_blank">val_raw.rar</a> | 65.57 GB | The validation set consisting of 12 hours of static speaker speech and 9 hours of moving speaker speech without any noise added |
| <a href="https://www.myairbridge.com/dl-service-3eaa53cc/dl.66727f36b225e9.45148635" target="_blank">test.rar</a> | 38.02 GB | The test set consisting of 12 hours of static speaker speech and 9 hours of moving speaker speech |
| <a href="https://www.myairbridge.com/dl-service-3eaa53cc/dl.66727f23ff9304.13647498" target="_blank">test_raw.rar</a> | 91.75 GB | The test set consisting of 12 hours of static speaker speech and 9 hours of moving speaker speech without any noise added |
| <a href="https://www.myairbridge.com/dl-service-3eaa53cc/dl.66727bceaa1409.33560489" target="_blank">dataset_info.rar</a> | 127.9 MB | The dataset information file including scene images, scene information, and speaker information |
| <a href="https://www.myairbridge.com/dl-service-3eaa53cc/dl.66727f69e9c803.79477985" target="_blank">transcriptions.trn</a> | 2.4 MB | The transcription file for the dataset |



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
└── train/val/test/val_raw/test_raw
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
    ├── ma_speech/ma_noisy_speech
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
TRAIN/VAL/TEST_M/S_scene_speakerId_utteranceId_channelId.flac

# Direct-Path Signal
TRAIN/VAL/TEST_M/S_scene_speakerId_utteranceId.flac

# Source Location
train/val/test_moving/static_source_location.csv
```

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