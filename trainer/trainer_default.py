import os
import sys
import pathlib as pb
import argparse as ap
import typing as T
import datetime
import random

import tqdm
from loguru import logger
import numpy as np
import skimage.io
import piq

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision

ROOT_PATH = pb.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))
import custom_nn.utils
import custom_nn.custom_layers as custom_layers
from custom_nn.utils import EntityKwargs
import custom_nn.base_model as base_model_module
import custom_nn.model as model_module
import custom_nn.discriminator as discriminator_module
import custom_nn.loss as loss_module
from fastmri.data import subsample
from fastmri.data import transforms, mri_data



class FastMRIDefaultTrainer:
    
    @classmethod
    def to_entity_kwargs(cls, obj):
        if isinstance(obj, EntityKwargs):
            out = obj
        elif isinstance(obj, T.Dict):
            out = EntityKwargs(**obj)
        elif isinstance(obj, T.List):
            out = []
            for o in obj:
                out.append(cls.to_entity_kwargs(o))
        elif obj is None:
            out = None
        else:
            logger.debug(str(obj))
            raise NotImplementedError()
        return out
    
    def __init__(
        self
        
        , dataset_path
        , batch_size
        
        , model__entity_kwargs: T.Union[EntityKwargs, T.Dict]
        , model__optimizer_entity_kwargs: T.Union[EntityKwargs, T.Dict]=None
        , model__scheduler_entity_kwargs: T.Union[EntityKwargs, T.Dict]=None
        , model__load: str=None

        , discriminator__entity_kwargs: T.Union[EntityKwargs, T.Dict]=None
        , discriminator__optimizer_entity_kwargs: T.Union[EntityKwargs, T.Dict]=None
        , discriminator__scheduler_entity_kwargs: T.Union[EntityKwargs, T.Dict]=None
        , discriminator__load: str=None
        
        , texture_proxy__entity_kwargs: T.Union[EntityKwargs, T.Dict]=None
        
        , criterion__entity2_kwargs_list: T.List[T.Union[EntityKwargs, T.Dict]]=None
        
        , logs_dir: str='./logs'
        
        , train_dataset_cache_file="train_dataset_cache.pkl"
        , val_dataset_cache_file="val_dataset_cache.pkl"
        , test_dataset_cache_file="test_dataset_cache.pkl"
    
        , device='cuda:0'
        , **kwargs  # Dummy arguments
    ):
    
        self.device = torch.device(device)
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        
        # Base model
        self.model = self.get_model(self.to_entity_kwargs(model__entity_kwargs), model__load).to(self.device)
        self.model__optimizer_entity_kwargs = self.to_entity_kwargs(model__optimizer_entity_kwargs)
        self.model__scheduler_entity_kwargs = self.to_entity_kwargs(model__scheduler_entity_kwargs)
        
        if discriminator__entity_kwargs is not None:
            self.discriminator = self.get_model(self.to_entity_kwargs(discriminator__entity_kwargs), discriminator__load).to(self.device)
            self.discriminator__optimizer_entity_kwargs = self.to_entity_kwargs(discriminator__optimizer_entity_kwargs)
            self.discriminator__scheduler_entity_kwargs = self.to_entity_kwargs(discriminator__scheduler_entity_kwargs)
            
        if texture_proxy__entity_kwargs is not None:
            self.texture_proxy = self.get_model(self.to_entity_kwargs(texture_proxy__entity_kwargs)).to(self.device)
            
        self.criterion__entity2_kwargs_list = self.to_entity_kwargs(criterion__entity2_kwargs_list)
        self.logs_dir = logs_dir
        
        self.train_dataset_cache_file = train_dataset_cache_file
        self.val_dataset_cache_file = val_dataset_cache_file
        self.test_dataset_cache_file = test_dataset_cache_file
        
    def train(self, epochs):
        train_dataloader = self.get_train_dataloader()
        val_dataloader = self.get_val_dataloader()
        criterion = self.get_criterion(self.criterion__entity2_kwargs_list)
        
        # Base model with {z, w} encoders
        steps = [self._generator_train_step]
        optimizers = [self.get_optimizer(self.model, self.model__optimizer_entity_kwargs)]
        schedulers = [None]
        if self.model__scheduler_entity_kwargs is not None:
            schedulers[0] = self.get_scheduler(optimizers[0], self.model__scheduler_entity_kwargs)
            
        # Discriminator
        if hasattr(self, 'discriminator'):
            steps.append(self._discriminator_train_step)
            optimizers.append(self.get_optimizer(self.discriminator, self.discriminator__optimizer_entity_kwargs))
            schedulers.append(None)
            if self.discriminator__scheduler_entity_kwargs is not None:
                schedulers[1] = self.get_scheduler(optimizers[1], self.discriminator__scheduler_entity_kwargs)
        
        writer = self.get_writer()
            
        logger.info('Init has complete. Training starts...\n')
        for e in range(epochs):
            self.run(e, train_dataloader, criterion,
                     steps, 
                     optimizers,
                     schedulers,
                     'train_', checkpoint=False, writer=writer)
            
            self.run(e, val_dataloader, None,
                     [self._generator_val_step],
                     [None],
                     [None],
                     'val_', checkpoint=True, writer=writer)
            
        logger.success('Training is complete!\n')
        
    def predict(
        self
        , center_fractions
        , accelerations
        , num_examples
        , out_dir
    ):
        
        self.model.eval()
        dataloader = self.get_val_dataloader(center_fractions, accelerations, batch_size=1)
        dataloader_length = len(dataloader)
        os.makedirs(out_dir, exist_ok=True)
        
        for i, batch in enumerate(dataloader, 1):
            image, mask, known_freq, known_image, mean, std, fname = batch
            
            image = image.to(self.device)
            known_freq = known_freq.to(self.device)
            known_image = known_image.to(self.device)
            mask = mask.to(self.device)
            mean = mean.to(self.device)
            std = std.to(self.device)
            
            try:
                rec_image, _, _, _ = self.model(image, known_freq, mask, is_deterministic=True)
            except ValueError:
                rec_image = self.model(image, known_freq, mask, is_deterministic=True)
                
            rec_image = rec_image.detach()
            image = image.detach()
            known_image = known_image.detach()
            
            metric_ssim = piq.ssim(rec_image, known_image, data_range=1.).item()
            metric_psnr = piq.psnr(rec_image, known_image, data_range=1.).item()
            
            save_to_name = pb.Path(out_dir) / pb.Path(fname[0]).stem
            skimage.io.imsave(save_to_name + '_input.png', image.squeeze().numpy())
            skimage.io.imsave(save_to_name + '.png', rec_image.squeeze().numpy())
            skimage.io.imsave(save_to_name + '_gt.png', known_image.squeeze().numpy())
            logger.info(f'{i}/{dataloader_length}: {pb.Path(fname).stem}: PSNR = {metric_psnr:.4f}, SSIM = {metric_ssim:.4f}\n')
            
            if i == num_examples:
                break
            
        logger.success('Prediction is complete!\n')
        
    def get_model(self, model_entity_kwargs: EntityKwargs, load: str=None) -> nn.Module:
        model = model_entity_kwargs.entity.lower()
        kwargs = model_entity_kwargs.kwargs
        
        if model == 'stylishfastmri':
            model = model_module.StylishFastMRI(**kwargs)
        elif model == 'basestylishfastmri':
            model = base_model_module.BaseStylishFastMRI(**kwargs)
        elif model == 'discriminator':
            model = discriminator_module.Discriminator(**kwargs)
        elif model == 'mobilenet_v2_encoder':
            model = custom_layers.MobileNetV2Encoder(**kwargs)
        elif model == 'mobilenet_v2_vaencoder':
            model = custom_layers.MobileNetV2VAEncoder(**kwargs)
        else: 
            raise NotImplementedError()
        
        if load is not None:
            model.load_state_dict(torch.load(load))
            logger.info(f'Loaded model from: {load}')
            
        return model
    
    def get_fastmri_data_transform(
        self
        , center_fractions=[0.08, 0.04, 0.02, 0.01]
        , accelerations=[4, 8, 16, 32]
    ):
        
        mask_func = subsample.RandomMaskFunc(
            center_fractions=center_fractions,
            accelerations=accelerations
        )
        unet_data_transform = transforms.UnetDataTransform(which_challenge="singlecoil", mask_func=mask_func)
        
        def data_transform(kspace, mask, target, data_attributes, filename, slice_num):
            image, mask, masked_kspace, target, mean, std, fname, slice_num, max_value = unet_data_transform(
                kspace=kspace,
                target=target,
                mask=mask,
                attrs=data_attributes,
                fname=filename,
                slice_num=slice_num
            )
            
            return image, mask, masked_kspace, target, mean, std, fname
        
        return data_transform
        
    def get_train_dataloader(self):
        dataset = mri_data.SliceDataset(
            root=pb.Path(self.dataset_path) / 'singlecoil_train',
            use_dataset_cache=True,
            dataset_cache_file=self.train_dataset_cache_file,
            transform=self.get_fastmri_data_transform(),
            sample_rate=0.2,
            challenge='singlecoil'
        )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=True,
            worker_init_fn=self.seed_worker
        )
        
        return dataloader
    
    def get_val_dataloader(self, *args, batch_size=None, **kwargs):
        dataset = mri_data.SliceDataset(
            root=pb.Path(self.dataset_path) / 'singlecoil_val',
            use_dataset_cache=True,
            dataset_cache_file=self.val_dataset_cache_file,
            transform=self.get_fastmri_data_transform(*args, **kwargs),
            sample_rate=0.2,
            challenge='singlecoil'
        )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size if batch_size is not None else self.batch_size,
            num_workers=2,
            shuffle=False,
            worker_init_fn=self.seed_worker
        )
        
        return dataloader
            
    def get_test_dataloader(self):
        dataset = mri_data.SliceDataset(
            root=pb.Path(self.dataset_path) / 'singlecoil_test',
            use_dataset_cache=True,
            dataset_cache_file=self.test_dataset_cache_file,
            transform=self.get_fastmri_data_transform(),
            sample_rate=0.2,
            challenge='singlecoil'
        )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=False,
            worker_init_fn=self.seed_worker
        )
        
        return dataloader
    
    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    def get_optimizer(self, model, optimizer_entity_kwargs):
        entity = optimizer_entity_kwargs.entity.lower()
        kwargs = optimizer_entity_kwargs.kwargs
        
        if entity == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), **kwargs)
        else:
            raise NotImplementedError()
        
        return optimizer
    
    def get_scheduler(self, optimizer, schdeduler_entity_kwargs):
        entity = schdeduler_entity_kwargs.entity.lower()
        kwargs = schdeduler_entity_kwargs.kwargs
        
        if entity == 'lambdalr':
            lambd = lambda epoch: kwargs['base'] ** epoch
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambd)
        else:
            raise NotImplementedError()
        
        return scheduler
    
    def get_criterion(self, entity2_kwargs_list: T.List[EntityKwargs]):
        criterion = ap.Namespace()
        
        for entity2_kwargs in entity2_kwargs_list:
            entity_label, entity = [e.lower() for e in entity2_kwargs.entity]
            kwargs = entity2_kwargs.kwargs
            coef_ = kwargs.pop('coef', 1.)
            
            if entity == 'l1':
                criterion_atom = nn.L1Loss(**kwargs)
            elif entity == 'kl_normal':
                criterion_atom = loss_module.KLNormalDivergence()
            elif entity == 'kl':
                criterion_atom = nn.KLDivLoss(log_target=True)
            elif entity == 'non_saturating_gan':
                criterion_atom = loss_module.NonSaturatingGANLoss()
            elif entity == 'hinge_gan':
                criterion_atom = loss_module.HingeGANLoss()
            
            criterion_atom.coef_ = coef_
                
            setattr(criterion, entity_label, criterion_atom)
            
        return criterion
    
    def get_writer(self):
        current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.writer_path = str(pb.Path(self.logs_dir) / current_time)
        logger.info(f'Experiment logs dir: {self.writer_path}\n')
        writer = SummaryWriter(self.writer_path)
        return writer
            
    def _generator_train_step(self, image, known_freq, known_image, mask, criterion, **kwargs):
        self.model.train()
        rec_image, z_mu, z_log_var, texture = self.model(image, known_freq, mask)
        gt_texture = self.texture_proxy(known_image)
        
        cache = {}
        
        loss_rec = criterion.rec(rec_image, known_image)
        cache['loss_rec'] = loss_rec.item()
        
        loss_kl = criterion.kl_normal(z_mu, z_log_var)
        cache['loss_kl'] = loss_kl.item()
        
        loss_texture = criterion.texture(texture, gt_texture)
        cache['loss_texture'] = loss_texture.item()
        
        loss = loss_kl * criterion.kl_normal.coef_ \
            + loss_texture * criterion.texture.coef_ \
            + loss_rec * criterion.rec.coef_
            
        if hasattr(criterion, 'adv'):
            fake_scores = self.discriminator(rec_image, image)
            loss_adv = criterion.adv(fake_scores)
            cache['loss_adv'] = loss_adv.item()
            loss += loss_adv * criterion.adv.coef_
            
        cache['reconstruction'] = rec_image.detach()
        
        return loss, cache
    
    def _generator_val_step(self, image, known_freq, known_image, mask, **kwargs):
        self.model.eval()
        # torch.manual_seed(42)  # Maybe not the most elegan way, as it may lead to repetitive noise
        # noise = torch.randn((b, 1, h, w), dtype=image.dtype, device=self.device)  # Explicit noise to make noise injections reproducible
        rec_image, _, _, _ = self.model(image, known_freq, mask, is_deterministic=True)
        rec_image = rec_image.detach()
        
        cache = {}
        
        cache['metric_ssim'] = piq.ssim(rec_image, known_image, data_range=1.).item()
        cache['metric_psnr'] = piq.psnr(rec_image, known_image, data_range=1.).item()
        cache['reconstruction'] = rec_image
        
        return None, cache
    
    def _discriminator_train_step(self, image, known_image, cache, criterion, **kwargs):
        rec_image = cache['reconstruction']
        fake_scores = self.discriminator(rec_image, image)
        real_scores = self.discriminator(known_image, image)
        
        loss = criterion.adv(fake_scores, real_scores)
        cache['loss_dis'] = loss.item()
        
        return loss, cache
    
    def run(
        self
        , epoch
        , dataloader
        , criterion
        , steps
        , optimizers=[None]
        , schedulers=[None]
        , log_prefix=''
        , checkpoint=False
        , writer=None
    ):
        
        dataloader_length = len(dataloader)
        pbar = tqdm.tqdm(enumerate(dataloader), leave=False, desc=f'Epoch: {epoch}', total=dataloader_length)
        loss_to_log = {}
        metric_to_log = {}
        
        for i, batch in pbar:
            image, mask, known_freq, known_image, mean, std, fname = batch
            
            image = image.to(self.device)
            known_freq = known_freq.to(self.device)
            known_image = known_image.to(self.device)
            mask = mask.to(self.device)
            mean = mean.to(self.device)
            std = std.to(self.device)
            
            cache = {}
            for step, optimizer in zip(steps, optimizers):
                loss, cache_ = step(image=image, known_freq=known_freq, known_image=known_image, mask=mask, criterion=criterion, cache=cache)
                cache = {**cache, **cache_}
                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()                
            
            # Update global sum loss statistics
            loss_atoms_to_log = {key: value for key, value in cache.items() if key.startswith('loss_')}    
            if len(loss_to_log) == 0:
                loss_to_log = loss_atoms_to_log
            else:
                for loss_key, loss_value in loss_atoms_to_log.items():
                    loss_to_log[loss_key] += loss_value
                    
            # Update global sum metric statistics
            metric_atoms_to_log = {key: value for key, value in cache.items() if key.startswith('metric_')}
            if len(metric_to_log) == 0:
                metric_to_log = metric_atoms_to_log
            else:
                for metric_key, metric_value in metric_atoms_to_log.items():
                    metric_to_log[metric_key] += metric_value
            
            # Update progres bar and writer
            pbar.set_postfix(**loss_atoms_to_log, **metric_atoms_to_log)
            if writer is not None:
                for loss_key, loss_value in loss_atoms_to_log.items():
                    writer.add_scalar(f'{log_prefix}{loss_key}', loss_value, dataloader_length * epoch + i)
                for metric_key, metric_value in metric_atoms_to_log.items():
                    writer.add_scalar(f'{log_prefix}{metric_key}', metric_value, dataloader_length * epoch + i)
                    
            # if i == dataloader_length: break
                    
                    
        pbar.close()
        
        for scheduler in schedulers:
            if scheduler is not None:
                scheduler.step()
        
        if len(loss_to_log) > 0:     
            final_loss_log_str = ', '.join([
                f'{loss_key} = {loss_value / dataloader_length:.4f}'
                for loss_key, loss_value in loss_to_log.items()
            ])
            logger.info(f"{log_prefix}epoch: {epoch:03d}: {final_loss_log_str}\n")

        if len(metric_to_log) > 0:     
            final_metric_log_str = ', '.join([
                f'{metric_key} = {metric_value / dataloader_length:.4f}'
                for metric_key, metric_value in metric_to_log.items()
            ])
            logger.info(f"{log_prefix}epoch: {epoch:03d}: {final_metric_log_str}\n")
        
        if writer is not None:
            for loss_key, loss_value in loss_to_log.items():
                writer.add_scalar(f'{log_prefix}epoch_{loss_key}', loss_value / dataloader_length, epoch)
            if 'reconstruction' in cache.keys():
                # image = image[:2] * std[:2, None, None, None] + mean[:2, None, None, None]
                image = image[:2]
                # rec = cache['reconstruction'][:2] * std[:2, None, None, None] + mean[:2, None, None, None]
                rec = cache['reconstruction'][:2]
                known_image = known_image[:2]
                
                tensor_to_log = torchvision.utils.make_grid(torch.cat([image, rec, known_image]), nrow=2)
                writer.add_image(f'{log_prefix}epoch_reconstruction', tensor_to_log, epoch)
                
        if checkpoint:
            checkpoint_name = f'{epoch:03d}'
            if 'metric_psnr' in metric_to_log.keys():
                checkpoint_name += f"_{metric_to_log['metric_psnr'] / dataloader_length:.4f}"
            if 'metric_ssim' in metric_to_log.keys():
                checkpoint_name += f"_{metric_to_log['metric_ssim'] / dataloader_length:.4f}"
            checkpoint_name += '.pt'
            
            torch.save(self.model.state_dict(), str(pb.Path(self.writer_path) / checkpoint_name))
