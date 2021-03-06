

import os
import torch
from Pix2PixHD_Networks import Discriminator, Generator, Loss
from Pix2PixHD_Utils import Manager, update_lr, weights_init
import pytorch_lightning as pl
from utils_scripts.utils import save, save_image

class Pix2PixHDCodex(pl.LightningModule):

    def __init__(self, opt, display_step):

        super().__init__()
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.save_hyperparameters()
        self.opt = opt
        
        self.display_step = display_step
        self.gen = Generator(opt).apply(weights_init)
        self.patch_gan = Discriminator(opt).apply(weights_init)

        criterion = Loss(opt)
        self.adversarial_criterion = criterion.disc_loss

        self.recon_criterion = criterion.gen_loss

    def _gen_step(self, conditioned_images, real_images):
        
        gen_loss = self.recon_criterion(self.patch_gan, self.gen, conditioned_images, real_images)
        
        return gen_loss

    def _disc_step(self, conditioned_images, real_images):

        disc_loss = self.adversarial_criterion(self.patch_gan, self.gen, conditioned_images, real_images)
        
        return disc_loss

    def configure_optimizers(self):
        lr = self.opt.lr
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(self.opt.beta1, self.opt.beta2), eps=self.opt.eps)
        disc_opt = torch.optim.Adam(self.patch_gan.parameters(), lr=lr, betas=(self.opt.beta1, self.opt.beta2), eps=self.opt.eps)
        return disc_opt, gen_opt

    def training_step(self, batch, batch_idx, optimizer_idx):
        condition, real = batch
        # if(self.current_epoch==1):
        #     sampleImg=torch.rand((1,1,28,28))
        #     self.logger.experiment.add_graph(Pix2PixHDCodex(self.opt, self.display_step),batch)
        if self.on_gpu:
            condition = condition.cuda(condition.device.index)
            real = real.cuda(real.device.index)
        self.gen.train()
        loss = None
        if optimizer_idx == 0:
            loss = self._disc_step(conditioned_images= condition, real_images  = real)
            self.log('PatchGAN Loss', loss)
        elif optimizer_idx == 1:
            loss = self._gen_step(conditioned_images = condition, real_images = real)
            self.log('Generator Loss', loss)

        if self.current_epoch % 100 == 0 and batch_idx ==0  : # and optimizer_idx==1
            
            print('epoch ', self.current_epoch)
            fake = self.gen(condition).detach()
            self.opt.image_dir = '/'.join(self.opt.image_dir.split('/')[:-1]) + '/Train'
            save(target = real, gen = fake, path = self.opt.image_dir, epoch = self.current_epoch, image=True, train_status='train')
            # print('Image saved')
            # display_progress(condition[0], fake[0], real[0])
        return loss
    
    def validation_step(self, batch, batch_idx):
        condition, real = batch
        
        if self.on_gpu:
            condition = condition.cuda(condition.device.index)
            real = real.cuda(real.device.index)
        self.gen.train()
        loss = None
        
        loss = self._disc_step(conditioned_images= condition, real_images  = real)
        self.log('PatchGAN (val) Loss', loss)
        loss = self._gen_step(conditioned_images = condition, real_images = real)
        self.log('Generator (val) Loss', loss)

        if self.current_epoch % 100 == 0: # and optimizer_idx==1
            
            fake = self.gen(condition).detach()
            self.opt.image_dir = '/'.join(self.opt.image_dir.split('/')[:-1]) + '/Train'
            save(target = real, gen = fake, path = self.opt.image_dir, epoch = self.current_epoch, image=True, train_status='validation')
            # print('Image saved')
            # display_progress(condition[0], fake[0], real[0])
        return loss

    def test_step(self, batch, batch_idx ):
        condition, real = batch
        # if(self.current_epoch==1):
        #     sampleImg=torch.rand((1,1,28,28))
        #     self.logger.experiment.add_graph(Pix2PixHDCodex(self.opt, self.display_step),batch)
        if self.on_gpu:
            condition = condition.cuda(condition.device.index)
            real = real.cuda(real.device.index)
        self.gen.eval()
        loss = None
        loss = self._disc_step(conditioned_images= condition, real_images  = real)
        self.log('PatchGAN (Test) Loss', loss)
        loss = self._gen_step(conditioned_images = condition, real_images = real)
        self.log('Generator (Test) Loss', loss)

        fake = self.gen(condition).detach()
        print(self.current_epoch)
        self.opt.image_dir = '/'.join(self.opt.image_dir.split('/')[:-1]) + '/Test'
        print('Save Path ', self.opt.image_dir )
        
        save(target = real, gen = fake, path = self.opt.image_dir, epoch = self.current_epoch, image=True, train_status='test')
        return loss

    def forward(self, dataloader):
        condition, labels = next(iter(dataloader))
        fake = self.gen(condition).detach()
        print(fake.shape)
        self.opt.image_dir = '/'.join(self.opt.image_dir.split('/')[:-1]) + '/Infer/'
        save_image( image_tensor = fake, path = self.opt.image_dir, save_all = True)