import os

os.environ["OMP_NUM_THREADS"] = str(1)  # limit the threads to reduce cpu overloads, will speed up when there are lots of CPU cores on the running machine

try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

from typing import Any, List
import pytorch_lightning as pl
import torch
if torch.multiprocessing.get_start_method() != 'spawn':
    torch.multiprocessing.set_start_method('spawn', force=True)  # fix stoi stuck

import soundfile as sf
import models.utils.general_steps as GS

from models.fasnet.FaSNet import FaSNet_TAC as FaSNet_TAC_Arch, FaSNet_origin
from models.utils.metrics import (cal_metrics_functional, recover_scale)
from models.utils import MyJsonEncoder, tag_and_log_git_status
from torchmetrics.functional.audio import permutation_invariant_training as pit, pit_permutate, scale_invariant_signal_distortion_ratio as si_sdr, scale_invariant_signal_noise_ratio as si_snr
from torch import Tensor
import soundfile as sf
import json
from pandas import DataFrame

torch.backends.cuda.matmul.allow_tf32 = True  # The flag below controls whether to allow TF32 on matmul. This flag defaults to False in PyTorch 1.12 and later.
torch.backends.cudnn.allow_tf32 = True  # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.

COMMON_AUDIO_METRICS = ['SNR', 'SDR', 'SI_SDR', 'WB_PESQ', 'DNSMOS']

# from torchmetrics.functional.audio import deep_noise_suppression_mean_opinion_score as dnsmos
from models.utils.dnsmos import deep_noise_suppression_mean_opinion_score as dnsmos
from torchmetrics.functional.audio import signal_distortion_ratio as sdr


def neg_si_snr(ys_hat: Tensor, ys: Tensor) -> Tensor:
    # ys, ys_hat: [B, Spk, T]
    si_snr_val, perms = pit(preds=ys_hat, target=ys, metric_func=si_snr, eval_func='max')
    neg_si_snr_val = -si_snr_val.mean()
    return neg_si_snr_val


class FaSNet_TAC(pl.LightningModule):

    def __init__(
            self,
            exp_name: str = "exp",
            #enc_dim=64, feature_dim=64, hidden_dim=128, layer=4, segment_size=50, nspk=2, win_len=4, context_len=16, sr=16000
            model: str = 'FaSNet_TAC',
            enc_dim: int = 64,
            feature_dim: int = 64,
            hidden_dim: int = 128,
            layer: int = 4,
            segment_size: int = 50,
            nspk: int = 2,
            win_len: int = 4,
            context_len: int = 16,
            sr: int = 16000,
            learning_rate: float = 0.001,
            dataset: str = 'wsj0',  # choices=['tac', 'wsj0']
            channels: List[int] = [0, 1, 2, 3, 4, 5, 6, 7],  # the channel used
            use_microphone_array_generalization: bool = False,  # whether to use microphone array generalization
            metrics: List[str] = COMMON_AUDIO_METRICS,
            write_examples: int = 9999,
            **kwargs):
        super().__init__()
        assert len(channels) > 1, "at least two channels"
        # save all the hyperparameters to the checkpoint
        self.save_hyperparameters()

        if model == 'FaSNet_TAC':
            self.FaSNet = FaSNet_TAC_Arch(
                enc_dim=enc_dim,
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                layer=layer,
                segment_size=segment_size,
                nspk=nspk,
                win_len=win_len,
                context_len=context_len,
                sr=sr,
            )
        else:
            raise Exception(f'model {model} is not supported')

        self.none_mic = torch.zeros(1, device=self.device).long()  # fixed-array
        self.negsisdr = neg_si_snr
        self.ref_chn_idx = 0
        self.use_microphone_array_generalization = use_microphone_array_generalization
        self.metrics=metrics
        self.write_examples = write_examples


    def on_train_start(self):
        if self.current_epoch == 0:
            if self.trainer.is_global_zero and hasattr(self.logger, 'log_dir'):
                # add git tags for better change tracking
                # note: if change self.logger.log_dir to self.trainer.log_dir, the training will stuck on multi-gpu training
                tag_and_log_git_status(self.logger.log_dir + '/git.out', self.logger.version, self.hparams.exp_name, model_name=type(self).__name__)
        self.none_mic = self.none_mic.to(self.device)

    def forward(self, x):
        return self.FaSNet(x, self.none_mic)

    def training_step(self, batch, batch_idx):
        x, ys, _ = batch
        if not self.use_microphone_array_generalization:
            x = x[:, self.hparams.channels, ...]
        ys = ys[:, :, self.ref_chn_idx, :]
        ys_est = self(x)
        neg_si_sdr = self.negsisdr(ys_est, ys)
        self.log('train/neg_si_sdr', neg_si_sdr, batch_size=x.shape[0])
        return neg_si_sdr

    def validation_step(self, batch, batch_idx):
        x, ys, paras = batch
        x = x[:, self.hparams.channels, ...]
        ys = ys[:, :, self.ref_chn_idx, :]
        ys_est = self(x)
        neg_si_sdr = self.negsisdr(ys_est, ys)

        dnsmos_val = dnsmos(ys_est, paras[0]['sample_rate'], personalized=False)
        for idx, name in enumerate(['p808', 'sig', 'bak', 'ovr']):
            self.log(f'val/dnsmos_{name}', dnsmos_val[..., idx].mean(), sync_dist=True, batch_size=x.shape[0])
        self.log('val/sdr', sdr(ys_est, ys).mean(), sync_dist=True, batch_size=x.shape[0])
        self.log('val/neg_si_sdr', neg_si_sdr, sync_dist=True, batch_size=x.shape[0])
        return neg_si_sdr

    def on_test_epoch_start(self):
        self.exp_save_path = self.trainer.logger.log_dir
        os.makedirs(self.exp_save_path, exist_ok=True)
        self.results, self.cpu_metric_input = [], []

    def on_test_epoch_end(self):
        GS.on_test_epoch_end(self=self, results=self.results, cpu_metric_input=self.cpu_metric_input, exp_save_path=self.exp_save_path)

    def test_step(self, batch, batch_idx):
        x, ys, paras = batch
        x = x[:, self.hparams.channels, ...]
        yr = ys[:, :, self.ref_chn_idx, :]
        sample_rate = 16000 if 'sample_rate' not in paras[0] else paras[0]['sample_rate']

        # 预测
        ys_hat = self(x)
        neg_si_sdr = self.negsisdr(ys_hat, yr)
        # imp_metrics, metrics = get_metrics(mix=x[0, ref_channel], clean=ys[0], estimate=ys_est[0])
        self.log('test/neg_si_sdr', neg_si_sdr)
        
        # write results & infos
        if paras[0]['save_for_ASR']:
            wavname = paras[0]['saveto'][0].replace('_CH1', '_CH25')
        else:
            wavname = os.path.basename(f"{paras[0]['index']}.wav")
        result_dict = {'id': batch_idx, 'wavname': wavname, 'neg_si_sdr': neg_si_sdr.item()}

        # recover wav's original scale. solve min ||Y^T a - X||F to obtain the scales of the predictions of speakers, cuz sisdr will lose scale
        x_ref = x[0, self.ref_chn_idx, :]
        _, ps = pit(preds=ys_hat, target=yr, metric_func=si_sdr, eval_func='max')
        yr_hat = pit_permutate(ys_hat, ps)  # reorder first

        # calculate metrics, input_metrics, improve_metrics on GPU
        metrics, input_metrics, imp_metrics = cal_metrics_functional(self.metrics, yr_hat[0], yr[0], x_ref.expand_as(yr[0]), sample_rate, device_only='gpu')
        result_dict.update(input_metrics)
        result_dict.update(imp_metrics)
        result_dict.update(metrics)
        self.cpu_metric_input.append((self.metrics, yr_hat[0].detach().cpu(), yr[0].detach().cpu(), x_ref.expand_as(yr[0]).detach().cpu(), sample_rate, 'cpu', None))

        # write examples
        if paras[0]['index'] < self.write_examples:
            GS.test_setp_write_example(
                self=self,
                xr=x[:, self.ref_chn_idx],
                yr=yr,
                yr_hat=yr_hat,
                sample_rate=sample_rate,
                paras=paras,
                result_dict=result_dict,
                wavname=wavname,
                exp_save_path=self.exp_save_path,
            )

        # return metrics, which will be collected, saved in test_epoch_end
        if 'metrics' in paras[0]:
            del paras[0]['metrics']  # remove circular reference
        result_dict['paras'] = paras[0]
        self.results.append(result_dict)
        return result_dict

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        if isinstance(batch, Tensor):
            x = batch
        else:
            x, ys, paras = batch

        x = x[:, self.hparams.channels, ...]
        ys_hat = self.forward(x)
        return ys_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5),
                'monitor': 'val/dnsmos_ovr',
            }
        }


from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor, ModelCheckpoint, ModelSummary)
from pytorch_lightning.cli import (LightningArgumentParser, LightningCLI)
# from pytorch_lightning.loggers import TensorBoardLogger
from models.utils.my_logger import MyLogger as TensorBoardLogger
from models.utils.my_save_config_callback import MySaveConfigCallback as SaveConfigCallback
from models.utils.my_rich_progress_bar import MyRichProgressBar as RichProgressBar


class FaSNetCLI(LightningCLI):

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:

        parser.set_defaults({
            # "data.batch_size": [16, 32],
            "trainer.accumulate_grad_batches": 1,
            "trainer.gradient_clip_val": 5,
            "trainer.gradient_clip_algorithm": "norm",
            "trainer.max_epochs": 200,
            "trainer.strategy": "ddp_find_unused_parameters_false",
        })

        # RichProgressBar
        parser.add_lightning_class_args(RichProgressBar, nested_key='progress_bar')
        if pl.__version__.startswith('1.5.'):
            parser.set_defaults({
                "progress_bar.refresh_rate_per_second": 1,
            })
        else:
            parser.set_defaults({"progress_bar.console_kwargs": {
                "force_terminal": True,
                "no_color": True,
                "width": 200,
            }})

        # ModelCheckpoint
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        model_checkpoint_defaults = {
            "model_checkpoint.filename": "epoch{epoch}_neg_si_sdr{val/neg_si_sdr:.4f}",
            "model_checkpoint.monitor": "val/neg_si_sdr",
            "model_checkpoint.mode": "min",
            "model_checkpoint.every_n_epochs": 1,
            "model_checkpoint.save_top_k": 5,
            "model_checkpoint.auto_insert_metric_name": False,
            "model_checkpoint.save_last": True
        }
        parser.set_defaults(model_checkpoint_defaults)

        # EarlyStopping
        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        early_stopping_defaults = {
            "early_stopping.monitor": "val/neg_si_sdr",
            "early_stopping.patience": 30,
            "early_stopping.mode": "min",
            "early_stopping.min_delta": 0.01,
        }
        parser.set_defaults(early_stopping_defaults)

        # LearningRateMonitor
        parser.add_lightning_class_args(LearningRateMonitor, "learning_rate_monitor")
        learning_rate_monitor_defaults = {
            "learning_rate_monitor.logging_interval": "epoch",
        }
        parser.set_defaults(learning_rate_monitor_defaults)

        # ModelSummary
        parser.add_lightning_class_args(ModelSummary, 'model_summary')
        model_summary_defaults = {
            "model_summary.max_depth": -1,
        }
        parser.set_defaults(model_summary_defaults)

        return super().add_arguments_to_parser(parser)

    def before_fit(self):
        resume_from_checkpoint: str = self.config['fit']['ckpt_path']
        if resume_from_checkpoint is not None and resume_from_checkpoint.endswith('last.ckpt'):
            # log in same dir
            # resume_from_checkpoint example: /mnt/home/quancs/projects/NBSS_pmt/logs/NBSS_ifp/version_29/checkpoints/last.ckpt
            resume_from_checkpoint = os.path.normpath(resume_from_checkpoint)
            splits = resume_from_checkpoint.split(os.path.sep)
            version = int(splits[-3].replace('version_', ''))
            save_dir = os.path.sep.join(splits[:-3])
            self.trainer.logger = TensorBoardLogger(save_dir=save_dir, name="", version=version, default_hp_metric=False)
        else:
            model_name = self.model.name if hasattr(self.model, 'name') else type(self.model).__name__
            self.trainer.logger = TensorBoardLogger('logs/', name=model_name, default_hp_metric=False)

    def before_test(self):
        torch.set_num_interop_threads(5)
        torch.set_num_threads(5)
        if self.config['test']['ckpt_path'] != None:
            ckpt_path = self.config['test']['ckpt_path']
        else:
            raise Exception('You should give --ckpt_path if you want to test')
        epoch = os.path.basename(ckpt_path).split('_')[0]
        write_dir = os.path.dirname(os.path.dirname(ckpt_path))

        test_set = 'test'
        if 'test_set' in self.config['test']['data']:
            test_set = self.config['test']['data']["test_set"]
        elif 'init_args' in self.config['test']['data'] and 'test_set' in self.config['test']['data']['init_args']:
            test_set = self.config['test']['data']['init_args']["test_set"]
        exp_save_path = os.path.normpath(write_dir + '/' + epoch + '_' + test_set + '_set')

        self.trainer.logger = TensorBoardLogger(exp_save_path, name='', default_hp_metric=False)

    def after_test(self):
        if not self.trainer.is_global_zero:
            return
        import fnmatch
        files = fnmatch.filter(os.listdir(self.trainer.log_dir), 'events.out.tfevents.*')
        for f in files:
            os.remove(self.trainer.log_dir + '/' + f)
            print('tensorboard log file for test is removed: ' + self.trainer.log_dir + '/' + f)


if __name__ == '__main__':
    # Example: nohup python FaSNet_TAC.py fit --config configs/datasets/ss_4_nmix_headtail_mid_frontend_ovlp.yaml --data.batch_size=[16,32] --model.channels=[0,2] --model.exp_name=notag --trainer.gpus=5, >> logs/FaSNet_TAC/version_0.log 2>&1 &
    cli = FaSNetCLI(
        FaSNet_TAC,
        pl.LightningDataModule,
        save_config_callback=SaveConfigCallback,
        save_config_kwargs={'overwrite': True},
        subclass_mode_data=True,
    )
