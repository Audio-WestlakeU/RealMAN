#from OptSRPDNN import opt
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import numpy as np
#import Dataset2 as at_dataset
import Module as at_module
import baselines.SSL.CRNN as at_model
from utils.my_save_config_callback import MySaveConfigCallback as SaveConfigCallback
from utils import tag_and_log_git_status
from utils import MyLogger as TensorBoardLogger
from utils import MyRichProgressBar as RichProgressBar
from packaging.version import Version
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI
from pytorch_lightning import LightningDataModule, LightningModule
from torch import Tensor
import torch
from typing import Tuple
import os
from RecordData import RealData
from sampler import MyDistributedSampler
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
os.environ["OMP_NUM_THREADS"] = str(8)
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
# torch.set_float32_matmul_precision('medium')


#opts = opt()
#dirs = opts.dir()

dataset_train = RealData(data_dir='/data/home/RealisticAudio/RealMAN/',
                target_dir=['/data/home/RealisticAudio/RealMAN/train/train_static_source_location.csv',
                            '/data/home/RealisticAudio/RealMAN/train/train_moving_source_location.csv'],
                noise_dir='/data/home/RealisticAudio/RealMAN/train/ma_noise/')

dataset_val = RealData(data_dir='/data/home/RealisticAudio/RealMAN/',
                target_dir=['/data/home/RealisticAudio/RealMAN/val/val_static_source_location.csv',
                            '/data/home/RealisticAudio/RealMAN/val/val_moving_source_location.csv'],
                noise_dir='/data/home/RealisticAudio/RealMAN/val/ma_noise/',
                on_the_fly=False)

dataset_test = RealData(data_dir='/data/home/RealisticAudio/RealMAN/',
                target_dir=[#'/data/home/RealisticAudio/RealMAN/test/test_static_source_location.csv',
                            '/data/home/RealisticAudio/RealMAN/test/test_moving_source_location.csv'
                            ],
                noise_dir='/data/home/RealisticAudio/RealMAN/test/ma_noise/',
                on_the_fly=False)

class MyDataModule(LightningDataModule):

    def __init__(self, num_workers: int = 5, batch_size: Tuple[int, int] = (16, 16)):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        return super().prepare_data()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset_train,sampler=MyDistributedSampler(dataset=dataset_train,seed=2,shuffle=True), batch_size=self.batch_size[0], num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset_val, sampler=MyDistributedSampler(dataset=dataset_val,seed=2,shuffle=False),batch_size=self.batch_size[1], num_workers=self.num_workers)
        
    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset_test, sampler=MyDistributedSampler(dataset=dataset_test,seed=2,shuffle=False),batch_size=self.batch_size[1], num_workers=self.num_workers)
        #return DataLoader(dataset_test,batch_size=self.batch_size[1], num_workers=self.num_workers)

class MyModel(LightningModule):

    def __init__(
            self,
            tar_useVAD: bool = True,
            res_the: int = 1,
            res_phi: int = 360,
            fs: int = 16000,
            win_len: int = 512,
            nfft: int = 512,
            win_shift_ratio: float = 0.625,
            max_num_sources: int = 1,
            return_metric: bool = True,
            compile: bool = False,
            device: str = 'cuda',
            exp_name: str = 'exp',            
    ):
        super().__init__()
        self.arch = at_model.CRNN()
        if compile:
            assert Version(torch.__version__) >= Version(
                '2.0.0'), torch.__version__
            
            self.arch = torch.compile(self.arch)

        # save all the parameters to self.hparams
        self.save_hyperparameters(ignore=['arch'])
        self.tar_useVAD = tar_useVAD
        self.max_num_sources = max_num_sources
        self.nfft = nfft
        self.fre_max = fs / 2
        self.return_metric = return_metric
        self.dostft = at_module.STFT(
            win_len=win_len, win_shift_ratio=win_shift_ratio, nfft=nfft)
        self.fre_range_used = range(1, int(self.nfft/2)+1, 1)
        self.get_metric = at_module.PredDOA()
        self.dev = device
        self.res_phi = res_phi
    def forward(self, x):
        return self.arch(x)

    def on_train_start(self):
        if self.current_epoch == 0:
            if self.trainer.is_global_zero and hasattr(self.logger, 'log_dir') and 'notag' not in self.hparams.exp_name:
                tag_and_log_git_status(self.logger.log_dir + '/git.out', self.logger.version,
                                       self.hparams.exp_name, model_name=type(self).__name__)

            if self.trainer.is_global_zero and hasattr(self.logger, 'log_dir'):
                with open(self.logger.log_dir + '/model.txt', 'a') as f:
                    f.write(str(self))
                    f.write('\n\n\n')
                # measure the model FLOPs
                # write_FLOPs(model=self, save_dir=self.logger.log_dir,
                #             num_chns=2, fs=16000, audio_time_len=4, model_file=__file__)

    def training_step(self, batch, batch_idx: int):
        mic_sig_batch = batch[0]
        targets_batch = batch[1]
        vad_batch = batch[2]
        data_batch = self.data_preprocess(mic_sig_batch, targets_batch)
        in_batch = data_batch[0]
        gt_batch = [data_batch[1],vad_batch]
        pred_batch = self(in_batch)
        loss = self.cal_cls_loss(pred_batch=pred_batch, gt_batch=gt_batch)
        self.log("train/loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx: int):
        mic_sig_batch = batch[0]
        targets_batch = batch[1]
        vad_batch = batch[2]
        data_batch = self.data_preprocess(mic_sig_batch, targets_batch)
        in_batch = data_batch[0]
        gt_batch = [data_batch[1],vad_batch]
        pred_batch = self(in_batch)
        if pred_batch.shape[1] > gt_batch[0].shape[1]:
            pred_batch = pred_batch[:,:gt_batch[0].shape[1],:]
        else:
            gt_batch[0] = gt_batch[0][:,:pred_batch.shape[1],:]
            gt_batch[1] = gt_batch[1][:,:pred_batch.shape[1],:]   
        loss = self.cal_cls_loss(pred_batch=pred_batch, gt_batch=gt_batch)
        self.log("valid/loss", loss, sync_dist=True)
        metric = self.get_metric(pred_batch=pred_batch,gt_batch=gt_batch,idx=None,tar_type='spect')
        for m in metric:
            self.log('valid/'+m, metric[m].item(), sync_dist=True)       

    def test_step(self, batch: Tensor, batch_idx: int):
        mic_sig_batch = batch[0]
        targets_batch = batch[1]
        vad_batch = batch[2]
        data_batch = self.data_preprocess(mic_sig_batch, targets_batch = targets_batch)
        in_batch = data_batch[0]
        gt_batch = [data_batch[1],vad_batch]
        pred_batch = self(in_batch)
        if pred_batch.shape[1] > gt_batch[0].shape[1]:
            pred_batch = pred_batch[:,:gt_batch[0].shape[1],:]
        else:
            gt_batch[0] = gt_batch[0][:,:pred_batch.shape[1],:]
            gt_batch[1] = gt_batch[1][:,:pred_batch.shape[1],:] 
        #print(pred_batch.shape)           
        loss = self.cal_cls_loss(pred_batch=pred_batch, gt_batch=gt_batch)
        self.log("test/loss", loss, sync_dist=True)
        metric = self.get_metric(pred_batch=pred_batch,gt_batch=gt_batch,idx=batch_idx,tar_type='spect')
        for m in metric:
            self.log('test/'+m, metric[m].item(), sync_dist=True)
            #self.log('test2/'+m, metric2[m].item(), sync_dist=True)

    def predict_step(self, batch, batch_idx: int):
        data_batch = self.data_preprocess(mic_sig_batch=batch.permute(0,2,1))
        in_batch = data_batch[0]        
        preds = self.forward(in_batch)
        return preds[0]

    def MSE_loss(self, preds, targets):
        nbatch = preds.shape[0]
        sum_loss = torch.nn.functional.mse_loss(preds, targets, reduction='none').contiguous().view(nbatch,-1)
        item_num = sum_loss.shape[1]
        return sum_loss.sum(axis=1) / item_num

    def cal_cls_loss(self, pred_batch=None, gt_batch=None):
        doa_batch = gt_batch[0]
        vad_batch = gt_batch[1]
        doa_batch = doa_batch[:,:,:].type(torch.LongTensor).cuda()
        nb,nt,_ = pred_batch.shape
        new_target_batch = torch.zeros(nb,nt,self.res_phi)
        for b in range(nb):
            for t in range(nt):
                new_target_batch[b,t,:] = self.gaussian_encode_symmetric(angles=doa_batch[b,t,],res_phi=self.res_phi)
        vad_expanded = vad_batch.expand(-1, -1, self.res_phi)
        new_target_batch = new_target_batch * vad_expanded.to(new_target_batch)
        
        pred_batch_cart = pred_batch.to(self.dev)
        new_target_batch = new_target_batch.to(self.dev)
        loss = torch.nn.functional.mse_loss(
            pred_batch_cart.contiguous(), new_target_batch.contiguous())
        return loss

    def gaussian_encode_symmetric(self, angles,res_phi,sigma=16):
        def gaussian_func_symmetric(gt_angle, sigma):
            angles = torch.arange(res_phi)#.to(gt_angle)
            distance = torch.minimum(torch.abs(angles - gt_angle.item()), torch.abs(angles - gt_angle.item() + res_phi))
            out = torch.exp(-0.5 * (distance % res_phi) ** 2 / sigma ** 2)
            return out
        spectrum = torch.zeros(res_phi)#.to(angles)
        if angles.shape[0] == 0:
            return spectrum
        for angle in angles:
            spectrum = torch.maximum(spectrum, gaussian_func_symmetric(angle, sigma)).cpu()
        return spectrum
    
    def data_preprocess(self, mic_sig_batch=None, targets_batch=None, eps=1e-6):
        data = []
        if mic_sig_batch is not None:
            mic_sig_batch = mic_sig_batch              
            stft = self.dostft(signal=mic_sig_batch)
            nb,nf,nt,nc = stft.shape
            stft = stft.permute(0, 3, 1, 2)
            stft_rebatch = stft.to(self.dev)
            nb, nc, nf, nt = stft_rebatch.shape
            mag = torch.abs(stft_rebatch)
            mean_value = torch.mean(mag.reshape(mag.shape[0],-1), dim = 1)
            mean_value = mean_value[:,np.newaxis,np.newaxis,np.newaxis].expand(mag.shape)
            stft_rebatch_real = torch.real(stft_rebatch) / (mean_value + eps)
            stft_rebatch_image = torch.imag(stft_rebatch) / (mean_value + eps)
            real_image_batch = torch.cat(
                (stft_rebatch_real, stft_rebatch_image), dim=1)
            data += [real_image_batch[:, :, self.fre_range_used, :]]
            data += [targets_batch]
        return data 
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.arch.parameters(), lr=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.975, last_epoch=-1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'monitor': 'valid/loss',
            }
        }


class MyCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

        parser.set_defaults(
            {"trainer.strategy": "ddp"})
        parser.set_defaults({"trainer.accelerator": "gpu"})

        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        parser.set_defaults({
            "early_stopping.monitor": "valid/loss",
            "early_stopping.min_delta": 0.01,
            "early_stopping.patience": 100,
            "early_stopping.mode": "min",
        })

        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        model_checkpoint_defaults = {
            "model_checkpoint.filename": "epoch{epoch}_valid_loss{valid/loss:.4f}",
            "model_checkpoint.monitor": "valid/loss",
            "model_checkpoint.mode": "min",
            "model_checkpoint.every_n_epochs": 1,
            "model_checkpoint.save_top_k": 100,
            "model_checkpoint.auto_insert_metric_name": False,
            "model_checkpoint.save_last": True
        }
        parser.set_defaults(model_checkpoint_defaults)

        # RichProgressBar
        parser.add_lightning_class_args(
            RichProgressBar, nested_key='progress_bar')
        parser.set_defaults({
            "progress_bar.console_kwargs": {
                "force_terminal": True,
                "no_color": True,  
                "width": 200,  
            }
        })

        # LearningRateMonitor
        parser.add_lightning_class_args(
            LearningRateMonitor, "learning_rate_monitor")
        learning_rate_monitor_defaults = {
            "learning_rate_monitor.logging_interval": "epoch",
        }
        parser.set_defaults(learning_rate_monitor_defaults)


    def before_fit(self):
        resume_from_checkpoint: str = self.config['fit']['ckpt_path']
        if resume_from_checkpoint is not None and resume_from_checkpoint.endswith('last.ckpt'):
            resume_from_checkpoint = os.path.normpath(resume_from_checkpoint)
            splits = resume_from_checkpoint.split(os.path.sep)
            version = int(splits[-3].replace('version_', ''))
            save_dir = os.path.sep.join(splits[:-3])
            self.trainer.logger = TensorBoardLogger(
                save_dir=save_dir, name="", version=version, default_hp_metric=False)
        else:
            model_name = type(self.model).__name__
            self.trainer.logger = TensorBoardLogger(
                'logs/', name=model_name, default_hp_metric=False)

    def before_test(self):
        torch.set_num_interop_threads(5)
        torch.set_num_threads(5)
        if self.config['test']['ckpt_path'] != None:
            ckpt_path = self.config['test']['ckpt_path']
        else:
            raise Exception('You should give --ckpt_path if you want to test')
        epoch = os.path.basename(ckpt_path).split('_')[0]
        write_dir = os.path.dirname(os.path.dirname(ckpt_path))
        exp_save_path = os.path.normpath(write_dir + '/' + epoch)

        import time
        # add 10 seconds for threads to simultaneously detect the next version
        self.trainer.logger = TensorBoardLogger(
            exp_save_path, name='', default_hp_metric=False)
        time.sleep(10)

    def after_test(self):
        if not self.trainer.is_global_zero:
            return
        import fnmatch
        files = fnmatch.filter(os.listdir(
            self.trainer.log_dir), 'events.out.tfevents.*')
        for f in files:
            os.remove(self.trainer.log_dir + '/' + f)
            print('tensorboard log file for test is removed: ' +
                  self.trainer.log_dir + '/' + f)


if __name__ == '__main__':
    cli = MyCLI(
        MyModel,
        MyDataModule,
        seed_everything_default=2, 
        save_config_callback=SaveConfigCallback,
        save_config_kwargs={'overwrite': True},
        #parser_kwargs={"parser_mode": "omegaconf"},
    )
