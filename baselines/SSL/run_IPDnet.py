#from OptSRPDNN import opt
#from utils_ import forgetting_norm
from torch.utils.data import DataLoader
import numpy as np
# import Dataset as at_dataset
import Module as at_module
from SingleTinyIPDnet import SingleTinyIPDnet
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
from copy import deepcopy
from sampler import MyDistributedSampler
from scipy.special import jn
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
os.environ["OMP_NUM_THREADS"] = str(8)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

#opts = opt()
#dirs = opts.dir()

dataset_train = RealData(data_dir='/data/home/RealisticAudio/RealMAN/',
                target_dir=['/data/home/RealisticAudio/RealMAN/train/train_static_source_location.csv',
                            '/data/home/RealisticAudio/RealMAN/train/train_moving_source_location.csv'],
                noise_dir='/data/home/RealisticAudio/RealMAN/train/ma_noise/',
                use_mic_id=[1,3,5,7,0],
                )

dataset_val = RealData(data_dir='/data/home/RealisticAudio/RealMAN/',
                target_dir=['/data/home/RealisticAudio/RealMAN/val/val_static_source_location.csv',
                            '/data/home/RealisticAudio/RealMAN/val/val_moving_source_location.csv'],
                noise_dir='/data/home/RealisticAudio/RealMAN/val/ma_noise/',
                use_mic_id=[1,3,5,7,0],
                on_the_fly=False)

dataset_test = RealData(data_dir='/data/home/RealisticAudio/RealMAN/',
                target_dir=['/data/home/RealisticAudio/RealMAN/test/test_static_source_location.csv',
                            #'/data/home/RealisticAudio/RealMAN/test/test_moving_source_location.csv'
                            ],
                noise_dir='/data/home/RealisticAudio/RealMAN/test/ma_noise/',
                use_mic_id=[1,3,5,7,0],
                on_the_fly=False)

class MyDataModule(LightningDataModule):
    
    def __init__(self, num_workers: int = 5, batch_size: Tuple[int, int] = (16, 16)):
        super().__init__()
        self.num_workers = num_workers
        # train: batch_size[0]; test: batch_size[1]
        self.batch_size = batch_size
        #self.sampler = MyDistributedSampler(seed=1)

    def prepare_data(self) -> None:
        return super().prepare_data()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset_train,sampler=MyDistributedSampler(dataset=dataset_train,seed=2,shuffle=True), batch_size=self.batch_size[0], num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset_val, sampler=MyDistributedSampler(dataset=dataset_val,seed=2,shuffle=False),batch_size=self.batch_size[1], num_workers=self.num_workers)
        
    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset_test, sampler=MyDistributedSampler(dataset=dataset_test,seed=2,shuffle=False),batch_size=self.batch_size[1], num_workers=self.num_workers)

class MyModel(LightningModule):

    def __init__(
            self,
            tar_useVAD: bool = True,
            ch_mode: str = 'M',
            res_the: int = 1,
            res_phi: int = 180,
            fs: int = 16000,
            win_len: int = 512,
            nfft: int = 512,
            win_shift_ratio: float = 0.625,
            method_mode: str = 'IDL',
            source_num_mode: str = 'KNum',
            cuda_activated: bool = True,
            max_num_sources: int = 1,
            return_metric: bool = True,
            exp_name: str = 'exp',
            compile: bool = False,
            device: str = 'cuda',
    ):
        super().__init__()
        self.arch = SingleTinyIPDnet()
        if compile:
            assert Version(torch.__version__) >= Version(
                '2.0.0'), torch.__version__
            self.arch = torch.compile(self.arch)

        # save all the parameters to self.hparams
        self.save_hyperparameters(ignore=['arch'])
        self.dev = device
        self.tar_useVAD = tar_useVAD
        self.method_mode = method_mode
        self.cuda_activated = cuda_activated
        self.source_num_mode = source_num_mode
        self.max_num_sources = max_num_sources
        self.ch_mode = ch_mode
        self.nfft = nfft
        self.fre_max = fs / 2
        self.return_metric = return_metric
        self.dostft = at_module.STFT(
            win_len=win_len, win_shift_ratio=win_shift_ratio, nfft=nfft)
        self.addbatch = at_module.AddChToBatch(ch_mode=self.ch_mode)
        self.removebatch = at_module.RemoveChFromBatch(ch_mode=self.ch_mode)
        self.fre_range_used = range(1, int(self.nfft/2)+1, 1)
        self.get_metric = at_module.PredDOA()
        self.res_the = res_the
        self.res_phi = res_phi
    def forward(self, x):
        return self.arch(x)

    def on_train_start(self):
        if self.current_epoch == 0:
            if self.trainer.is_global_zero and hasattr(self.logger, 'log_dir') and 'notag' not in self.hparams.exp_name:
                # note: if change self.logger.log_dir to self.trainer.log_dir, the training will stuck on multi-gpu training
                tag_and_log_git_status(self.logger.log_dir + '/git.out', self.logger.version,
                                       self.hparams.exp_name, model_name=type(self).__name__)

            if self.trainer.is_global_zero and hasattr(self.logger, 'log_dir'):
                with open(self.logger.log_dir + '/model.txt', 'a') as f:
                    f.write(str(self))
                    f.write('\n\n\n')

    def training_step(self, batch, batch_idx: int):
        mic_sig_batch = batch[0]
        targets_batch = batch[1]
        vad_batch = batch[2]
        array_topo = batch[3]
        data_batch = self.data_preprocess(mic_sig_batch, targets_batch, array_topo, vad_batch)
        in_batch = data_batch[0]
        gt_batch = data_batch[1:]
        pred_batch = self(in_batch)
        loss = self.cal_loss(pred_batch=pred_batch, gt_batch=gt_batch)
        self.log("train/loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx: int):
        mic_sig_batch = batch[0]
        targets_batch = batch[1]
        vad_batch = batch[2]
        array_topo= batch[3]
        data_batch = self.data_preprocess(mic_sig_batch, targets_batch, array_topo, vad_batch)
        in_batch = data_batch[0]
        gt_batch = data_batch[1:]
        pred_batch = self(in_batch)
        if pred_batch.shape[1] > gt_batch[0].shape[1]:
            pred_batch = pred_batch[:,:gt_batch[0].shape[1],:,:,:]
        else:
            gt_batch[0] = gt_batch[0][:,:pred_batch.shape[1],:]
            gt_batch[1] = gt_batch[1][:pred_batch.shape[1],:,:,:]
            gt_batch[-1] = gt_batch[-1][:,:pred_batch.shape[1],:]   
        loss = self.cal_loss(pred_batch=pred_batch, gt_batch=gt_batch)

        self.log("valid/loss", loss, sync_dist=True)
        get_metric = at_module.PredDOA(mic_location=gt_batch[-2])
        metric = get_metric(pred_batch=pred_batch,gt_batch=gt_batch,idx=None,tar_type='ipd')
        for m in metric:
            self.log('valid/'+m, metric[m].item(), sync_dist=True)      

    def test_step(self, batch: Tensor, batch_idx: int):
        mic_sig_batch = batch[0]
        targets_batch = batch[1]
        vad_batch = batch[2]
        array_topo = batch[3]
        data_batch = self.data_preprocess(mic_sig_batch, targets_batch, array_topo, vad_batch)
        in_batch = data_batch[0]
        gt_batch = data_batch[1:]
        pred_batch = self(in_batch)
        if pred_batch.shape[1] > gt_batch[0].shape[1]:
            pred_batch = pred_batch[:,:gt_batch[0].shape[1],:,:,:]
        else:
            gt_batch[0] = gt_batch[0][:,:pred_batch.shape[1],:]
            gt_batch[1] = gt_batch[1][:pred_batch.shape[1],:,:,:]
            gt_batch[-1] = gt_batch[-1][:,:pred_batch.shape[1],:]   
        loss = self.cal_loss(pred_batch=pred_batch, gt_batch=gt_batch)
        self.log("test/loss", loss, sync_dist=True)
        get_metric = at_module.PredDOA(mic_location=gt_batch[-2])
        metric = get_metric(pred_batch=pred_batch,gt_batch=gt_batch,idx=batch_idx,tar_type='ipd')
        for m in metric:
            self.log('test/'+m, metric[m].item(), sync_dist=True)
            #self.log('test2/'+m, metric2[m].item(), sync_dist=True)

    def predict_step(self, batch, batch_idx: int):
        data_batch = self.data_preprocess(mic_sig_batch=batch.permute(0,2,1))
        in_batch = data_batch[0]
        #print(in_batch.device)        
        preds = self.forward(in_batch)
        return preds[0]

    def cal_loss(self, pred_batch=None, gt_batch=None):
        ipd_gt = gt_batch[1]
        nb, nt, _, nmic, nsource = pred_batch.shape
        pred_batch = pred_batch.reshape(nb*nt, -1, nsource).permute(0,2,1)
        ipd_gt = ipd_gt.reshape(nb*nt,-1,nsource).permute(0,2,1)
        # [sigle source do not need PIT for loss calculation]
        # pred_batch = pred_batch.to(self.dev)
        # best_metric, best_perm = permutation_invariant_training(pred_batch, ipd_gt, self.MSE_loss, 'min')
        # pred_batch= pit_permutate(pred_batch, best_perm)
        loss = torch.nn.functional.mse_loss(
            pred_batch.contiguous(), ipd_gt.contiguous())
        return loss
    
    def euclidean_distances_to_bessel(self, data,fre_use, order=0):
        reference = data[0, :]
        distances = np.sqrt(np.sum((data[1:] - reference) ** 2, axis=1))
        frequencies = 2 * np.pi * np.linspace(0, 8000, 257) / 340
        frequencies = frequencies[fre_use]
        bessel_values_extended = []
        for distance in distances:
            bessel_value = jn(order, frequencies * distance)
            zero_vector = np.zeros(256)
            extended_value = np.concatenate((bessel_value, zero_vector))
            bessel_values_extended.append(extended_value)
        bessel_values_final = np.array(bessel_values_extended)
        return bessel_values_final.T
    
    def data_preprocess(self, mic_sig_batch=None, targets_batch=None,array_gemo_data=None, vad_data_batch=None, eps=1e-6):
        ori_mic_location = array_gemo_data.cpu().numpy()[0]
        mic_sig_batch = mic_sig_batch
        data = []
        # [random shuffle the microphone array]
        # state = np.random.get_state()
        # mic_sig_batch = mic_sig_batch.permute(2,0,1).cpu().numpy()
        # np.random.shuffle(mic_sig_batch)
        # mic_sig_batch = torch.from_numpy(mic_sig_batch).permute(1,2,0)
        # np.random.set_state(state)
        # np.random.shuffle(ori_mic_location)     
        mic_loc = ori_mic_location       
        gerdpipd = at_module.DPIPD(ndoa_candidate=[self.res_the, self.res_phi],
                    mic_location=mic_loc,
                    nf=int(self.nfft/2) + 1,
                    fre_max=self.fre_max,
                    ch_mode=self.ch_mode,
                    speed=340)             
        stft = self.dostft(signal=mic_sig_batch)
        nb,nf,nt,nc = stft.shape
        stft_rebatch = stft.permute(0, 3, 1, 2)
        stft_rebatch = stft_rebatch.to(self.dev)
        nb, nc, nf, nt = stft_rebatch.shape
        mag = torch.abs(stft_rebatch)
        mean_value = torch.mean(mag.reshape(mag.shape[0],-1), dim = 1)
        mean_value = mean_value[:,np.newaxis,np.newaxis,np.newaxis].expand(mag.shape)
        stft_rebatch_real = torch.real(stft_rebatch) / (mean_value + eps)
        stft_rebatch_image = torch.imag(stft_rebatch) / (mean_value + eps)
        real_image_batch = torch.cat(
            (stft_rebatch_real, stft_rebatch_image), dim=1)
        data += [real_image_batch[:, :, self.fre_range_used, :]]
        ele_data = torch.ones(targets_batch.shape) * 90
        azi_ele = torch.cat((ele_data[:,:,np.newaxis,:].to(targets_batch),targets_batch[:,:,np.newaxis,:]),dim=-2)
        DOAw_batch = azi_ele / 180 * np.pi
        source_doa = DOAw_batch.cpu().numpy()
        if self.ch_mode == 'M':
            _, ipd_batch = gerdpipd(source_doa=source_doa)
        elif self.ch_mode == 'MM':
            _, ipd_batch = gerdpipd(source_doa=source_doa)
        # set non-source taget (2d diffuse coherence)
        non_source_target = self.euclidean_distances_to_bessel(mic_loc,fre_use=self.fre_range_used)                
        non_source_target = torch.from_numpy(non_source_target)  
        
        ipd_batch = np.concatenate((ipd_batch.real[:, :, self.fre_range_used, :, :], ipd_batch.imag[:, :, self.fre_range_used, :, :]), axis=2).astype(np.float32)  # (nb, ntime, 2nf, nmic-1, nsource)
        ipd_batch = torch.from_numpy(ipd_batch)
        ipd_batch = ipd_batch.to(self.dev)
        if self.tar_useVAD:
            nb, nt, nf, nmic, nsrc = ipd_batch.shape
            vad_batch = vad_data_batch.clone()
            vad_batch = vad_batch.to(self.dev)
            vad_batch_copy = deepcopy(vad_batch).to(vad_batch)
            th = 0
            vad_batch_copy[vad_batch_copy <= th] = 0
            vad_batch_copy[vad_batch_copy > th] = 1
            vad_batch_expand_ipd = vad_batch_copy[:, :, np.newaxis, np.newaxis, :].expand(nb, nt, nf, nmic, nsrc)#.reshape(nb,nt,512,-1,nsrc)                
            ipd_batch = ipd_batch * vad_batch_expand_ipd
            for i in range(nb):
                for j in range(nt):
                    for k in range(nsrc):
                        if (ipd_batch[i,j,:,:,k] == 0).all():
                            ipd_batch[i,j,:,:,k] = non_source_target.to(ipd_batch)
        ipd_batch = ipd_batch.view(nb*nt, nf, nmic, nsrc)
        data += [targets_batch]
        data += [ipd_batch]
        data += [mic_loc]
        data += [vad_batch]
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
        # parser.set_defaults({"trainer.gradient_clip_val": 5, "trainer.gradient_clip_algorithm":"norm"})

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
            # resume_from_checkpoint example: /home/zhangsan/logs/MyModel/version_29/checkpoints/last.ckpt
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
        seed_everything_default=2,  # can be any seed value
        save_config_callback=SaveConfigCallback,
        save_config_kwargs={'overwrite': True},
        #parser_kwargs={"parser_mode": "omegaconf"},
    )
