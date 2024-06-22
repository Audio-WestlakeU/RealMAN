import importlib
import os

import torch
import yaml
from jsonargparse import ArgumentParser
from pytorch_lightning import LightningModule
from argparse import ArgumentParser as Parser
import traceback
from typing import *


class FakeModule(torch.nn.Module):

    def __init__(self, module: LightningModule) -> None:
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module.predict_step(x, 0)


def _get_num_params(model: torch.nn.Module):
    num_params = sum(param.numel() for param in model.parameters())
    return num_params


def _test_FLOPs(model: LightningModule, save_dir: str, num_chns: int, fs: int, audio_time_len: int = 4, num_params: int = None):
    module = FakeModule(model).cuda()

    try:
        # torcheval
        from torcheval.tools.module_summary import get_module_summary
        ms = get_module_summary(module, module_args=(torch.randn(1, num_chns, int(fs * audio_time_len), dtype=torch.float32),))
        flops_forward_eval, flops_back_eval = ms.flops_forward, ms.flops_backward
        params_eval = num_params if num_params is not None else ms.num_parameters

        print(f"FLOPs: forward={flops_forward_eval/1e9:.2f} G, back={flops_back_eval/1e9:.2f} G")

        with open(os.path.join(save_dir, 'FLOPs.yaml'), 'w') as f:
            yaml.dump(
                {
                    "flops_forward": f"{flops_forward_eval/1e9:.2f} G",
                    "flops_backward": f"{flops_back_eval/1e9:.2f} G",
                    "params": f"{params_eval/1e6:.3f} M",
                    "fs": fs,
                    "audio_time_len": audio_time_len,
                    "num_chns": num_chns,
                }, f)
            f.close()

        with open(os.path.join(save_dir, 'FLOPs-detailed.txt'), 'w') as f:
            f.write(str(ms))
            f.close()
    except Exception as e:
        exp_file = os.path.join(save_dir, 'FLOPs-failed.txt')
        traceback.print_exc(file=open(exp_file, 'w'))
        print(f"FLOPs test failed '{repr(e)}', see {exp_file}")


def import_class(class_path: str):
    try:
        iclass = importlib.import_module(class_path)
        return iclass
    except:
        imodule = importlib.import_module('.'.join(class_path.split('.')[:-1]))
        iclass = getattr(imodule, class_path.split('.')[-1])
        return iclass


def replace_main_module(config: Dict[str, Any], model_class_path: str):
    for k, v in config.items():
        if type(v) == dict:
            config[k] = replace_main_module(v, model_class_path=model_class_path)
        else:
            if k == 'class_path' and v.startswith('__main__'):
                config[k] = '.'.join(model_class_path.split('.')[:-1] + [v.split('.')[-1]])
    return config


def _test_FLOPs_from_config(save_dir: str, model_class_path: str, num_chns: int, fs: int, audio_time_len: int = 4, config_file: str = None):
    if config_file is None:
        config_file = os.path.join(save_dir, 'config.yaml')

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.load(f, yaml.FullLoader)
        config['model'] = replace_main_module(config=config['model'], model_class_path=model_class_path)
        if 'compile' in config['model'] and config['model']['compile'] == True:
            config['model']['compile'] = False  # Note: the flops measured on compiled model is not accurate
    model_class = import_class(model_class_path)
    parser = ArgumentParser()
    parser.add_class_arguments(model_class)

    model_config = parser.instantiate_classes(config['model'])
    model = model_class(**model_config.as_dict())
    num_params = _get_num_params(model=model)
    try:
        # torcheval report error for shared modules, so config module to not share
        if "full_share" in config['model']['arch']['init_args']:
            if config['model']['arch']['init_args']['full_share'] == True:
                config['model']['arch']['init_args']['full_share'] = False
                model_config = parser.instantiate_classes(config['model'])
                model = model_class(**model_config.as_dict())
            elif type(config['model']['arch']['init_args']['full_share']) == int:
                config['model']['arch']['init_args']['full_share'] = 9999
                model_config = parser.instantiate_classes(config['model'])
                model = model_class(**model_config.as_dict())
    except:
        ...
    _test_FLOPs(model, save_dir=save_dir, num_chns=num_chns, fs=fs, audio_time_len=audio_time_len, num_params=num_params)


def write_FLOPs(model: LightningModule, save_dir: str, model_file: str, num_chns: int, fs: int = None, nfft: int = None, audio_time_len: int = 4):
    assert fs is not None or nfft is not None, (fs, nfft)

    if model.__class__.__module__.startswith('__main__'):
        model_class = f"{model_file.replace('.py', '')}.{type(model).__name__}"
    else:
        model_class = f"{str(model.__class__.__module__)}.{type(model).__name__}"

    if fs:
        cmd = f'CUDA_VISIBLE_DEVICES={model.device.index}, python -m utils.flops ' + f'--save_dir {save_dir} --model_class_path {model_class} ' + f'--num_chns {num_chns} --fs {fs} --audio_time_len {audio_time_len}'
    else:
        cmd = f'CUDA_VISIBLE_DEVICES={model.device.index}, python -m utils.flops ' + f'--save_dir {save_dir} --model_class_path {model_class} ' + f'--num_chns {num_chns} --nfft {nfft} --audio_time_len {audio_time_len}'
    print(cmd)
    os.system(cmd)


if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=5, python -m utils.flops --save_dir logs/SSFNetLM/version_90 --model_class_path models.SSFNetLM.SSFNetLM --num_chns 6 --fs 8000
    # CUDA_VISIBLE_DEVICES=5, python -m utils.flops --save_dir logs/SSFNetLM/version_90 --model_class_path models.SSFNetLM.SSFNetLM --num_chns 6 --nfft 256
    parser = Parser()
    parser.add_argument('--save_dir', type=str, required=True, help='save FLOPs to dir')
    parser.add_argument('--model_class_path', type=str, required=True, help='the import path of your Lightning Module')
    parser.add_argument('--num_chns', type=int, required=True, help='the number of microphone channels')
    parser.add_argument('--fs', type=int, default=None, help='sampling rate')
    parser.add_argument('--nfft', type=int, default=None, help='sampling rate')
    parser.add_argument('--audio_time_len', type=float, default=4., help='seconds of test mixture waveform')
    parser.add_argument('--config_file', type=str, default=None, help='config file path')
    args = parser.parse_args()

    fs = args.fs
    if fs is None:
        if args.nfft is None:
            print('MACs test error: you should specify the fs or nfft')
            exit(-1)
        fs = {256: 8000, 512: 16000}[args.nfft]

    _test_FLOPs_from_config(
        save_dir=args.save_dir,
        model_class_path=args.model_class_path,
        num_chns=args.num_chns,
        fs=fs,
        audio_time_len=args.audio_time_len,
        config_file=args.config_file,
    )
