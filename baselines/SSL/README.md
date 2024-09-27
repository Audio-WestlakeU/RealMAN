### Introduction

In the RealMAN paper, we use the CRNN and a tiny version of IPDnet [1] as our baseline models. The design of the [CRNN structure](https://github.com/BingYang-20/SRP-DNN) is based on [2]. For the IPDnet, please visit the [FN-SSL](https://github.com/Audio-WestlakeU/FN-SSL) for more details.

| Code | Description |
| --- | --- |
| `CRNN.py, SingleTinyIPDnet.py` | The network implementation. |
| `RecordData.py` | The reference dataloader implementation. |
| `run_CRNN.py, run_IPDnet.py` | The pytorch-lightning Trainer implementation.|

### Usage
We have implemented the baseline methods using the Pytorch-lightning framework.

* For train,

```
python run_CRNN.py/run_IPDnet.py fit --data.batch_size=[*,*] --trainer.devices=*,*
```

* For inference,

```
python run_CRNN.py/run_IPDnet.py test --ckpt_path logs/MyModel/version_x/checkpoints/**.ckpt --trainer.devices=*,*
```

Meanwhile, we have provided code for training the variable-array version of IPDnet by setting the  `is_variable_array` of dataset module to True.



[1] Yabo Wang, Bing Yang, Xiaofei Li. [IPDnet: A Universal Direct-Path IPD Estimation Network for Sound Source Localization](https://arxiv.org/abs/2405.07021).

[2] Bing Yang, Hong Liu, Xiaofei Li. [SRP-DNN: Learning direct-path phase difference for multiple moving sound source localization](https://ieeexplore.ieee.org/document/9746624). ICASSP, 2022.
