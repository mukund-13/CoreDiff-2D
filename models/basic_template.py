# This part builds heavily on https://github.com/Hzzone/DU-GAN.
import torch
import os.path as osp

import torchvision
import tqdm
import argparse
import torch.distributed as dist
from torch.utils.data import DataLoader
import wandb
from PIL import Image
from utils.dataset import dataset_dict
from utils.loggerx import LoggerX
from utils.measure import compute_measure, gen_visualization_files
from utils.sampler import RandomSampler
from utils.ops import load_network
import numpy as np

class TrainTask(object):

    def __init__(self, opt):
        self.opt = opt
        self.logger = LoggerX(save_root=osp.join(
            osp.dirname(osp.dirname(osp.abspath(__file__))), 'output', '{}_{}'.format(opt.model_name, opt.run_name)))
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        self.set_loader()
        self.set_model()

    @staticmethod
    def build_default_options():
        parser = argparse.ArgumentParser('Default arguments for training of different methods')

        parser.add_argument('--save_freq', type=int, default=2500,
                            help='save frequency')
        parser.add_argument('--batch_size', type=int, default=4,
                            help='batch_size')
        parser.add_argument('--test_batch_size', type=int, default=1,
                            help='test_batch_size')
        parser.add_argument('--num_workers', type=int, default=4,
                            help='num of workers to use')
        parser.add_argument('--max_iter', type=int, default=150000,
                            help='number of training iterations')
        parser.add_argument('--resume_iter', type=int, default=0,
                            help='number of training epochs')
        parser.add_argument('--test_iter', type=int, default=150000,
                            help='number of epochs for test')
        parser.add_argument("--local_rank", type=int, default=0)
        parser.add_argument("--mode", type=str, default='train')
        parser.add_argument('--wandb', action="store_true")

        # run_name and model_name
        parser.add_argument('--run_name', type=str, default='default',
                            help='each run name')
        parser.add_argument('--model_name', type=str, default='corediff',
                            help='the type of method')

        # training parameters for one-shot learning framework
        parser.add_argument("--osl_max_iter", type=int, default=3001,
                            help='number of training iterations for one-shot learning framework training')
        parser.add_argument("--osl_batch_size", type=int, default=8,
                            help='batch size for one-shot learning framework training')
        parser.add_argument("--index", type=int, default=10,
                            help='slice index selected for one-shot learning framework training')
        parser.add_argument("--unpair", action="store_true",
                            help='use unpaired data for one-shot learning framework training')
        parser.add_argument("--patch_size", type=int, default=256,
                            help='patch size used to divide the image')

        # dataset
        # parser.add_argument('--train_dataset', type=str, default='mayo_2016_sim')
        # parser.add_argument('--test_dataset', type=str, default='mayo_2016_sim')   # mayo_2020, piglte, phantom, mayo_2016
        # parser.add_argument('--test_id', type=int, default=9,
                            # help='test patient index for Mayo 2016')
        parser.add_argument('--context', action="store_true",
                            help='use contextual information')   #
        parser.add_argument('--image_size', type=int, default=512)
        parser.add_argument('--dose', type=int, default=5,
                            help='dose% data use for training and testing')
        
        #Added
        parser.add_argument('--hq_dir', type=str, default='/projects/synergy_lab/garvit217/enhancement_data/train/HQ/', help='Path to high-quality dataset')
        parser.add_argument('--lq_dir', type=str, default='/projects/synergy_lab/garvit217/enhancement_data/train/LQ/', help='Path to low-quality dataset')
        parser.add_argument('--hq_dir_test', type=str, default='/Users/ayushchaturvedi/Documents/test_data/HQ/', help='Path to high-quality test dataset')
        parser.add_argument('--lq_dir_test', type=str, default='/Users/ayushchaturvedi/Documents/test_data/LQ/', help='Path to low-quality test dataset')
    
        parser.add_argument('--dataset_length', type=int, default=784, help='Number of images in dataset')
        parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')


        return parser

    @staticmethod
    def build_options():
        pass

    def load_pretrained_dict(self, file_name: str):
        self.project_root = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
        return load_network(osp.join(self.project_root, 'pretrained', file_name))
    
    def set_loader(self, mode=None):
        if mode:
            self.opt.mode = mode
        print(f"Setting loader with mode: {self.opt.mode}")
        opt = self.opt
        if opt.mode == 'train':
            train_dataset = dataset_dict['train'](root_dir_h=opt.hq_dir, root_dir_l=opt.lq_dir, length=opt.dataset_length)
            self.train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.num_workers,
                pin_memory=True
            )

        if opt.mode == 'test' or 'test' in opt.mode:
            print("Initializing test loader...")
            test_dataset = dataset_dict['test'](root_dir_h=opt.hq_dir_test, root_dir_l=opt.lq_dir_test, length=784)
            self.test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=opt.test_batch_size,
                shuffle=False,
                num_workers=opt.num_workers,
                # pin_memory=True
            )
            print(f"Test loader initialized with {len(self.test_loader)} batches.")

    def set_loader3(self):
        opt = self.opt

        if opt.mode == 'train':
            # Call the partial object to create an instance of the dataset
            train_dataset = dataset_dict['train'](root_dir_h=opt.hq_dir, root_dir_l=opt.lq_dir, length=opt.dataset_length)
            self.train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.num_workers,
                pin_memory=True
            )
            print(len(self.train_loader))
            print(opt.dataset_length)

        if opt.mode == 'test':
            # Call the partial object to create an instance of the dataset
            test_dataset = dataset_dict['test'](root_dir_h=opt.hq_dir_test, root_dir_l=opt.lq_dir_test, length=914)
            self.test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=opt.test_batch_size,
                shuffle=False,
                num_workers=opt.num_workers,
                pin_memory=True
            )

            # Preload some test images
            test_images = [test_dataset[i] for i in range(0, min(300, len(test_dataset)), 75)]
            low_dose = torch.stack([torch.from_numpy(x[0]) for x in test_images], dim=0).cuda()
            full_dose = torch.stack([torch.from_numpy(x[1]) for x in test_images], dim=0).cuda()
            self.test_images = (low_dose, full_dose)

        # self.test_dataset = test_dataset

    def set_loader2(self):
        opt = self.opt
        if opt.mode == 'train':
            # train_dataset = CTDataset(root_dir_h=args.hq_dir, root_dir_l=args.lq_dir, length=5120)  # Adjust the length as needed

            train_dataset = dataset_dict['train'](root_dir_h=opt.hq_dir, root_dir_l=opt.lq_dir, length=5120)
            train_sampler = RandomSampler(dataset=train_dataset, batch_size=opt.batch_size,
                                        num_iter=opt.max_iter, restore_iter=opt.resume_iter)
            self.train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

            # self.train_loader = torch.utils.data.DataLoader(
            #     dataset=train_dataset,
            #     batch_size=opt.batch_size,
            #     sampler=train_sampler,
            #     shuffle=False,
            #     drop_last=False,
            #     num_workers=opt.num_workers,
            #     pin_memory=True
            # )

        # Setup test dataset
        # Ensure 'test_dataset' is a key in your dataset_dict that correctly refers to your dataset setup

# Assuming CTDataset is imported and ready to be used
   
        test_dataset = dataset_dict.get('test', None)  # Using a default or a specific key as necessary
        if test_dataset:

            self.test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

            # self.test_loader = torch.utils.data.DataLoader(
            #     dataset=test_dataset,
            #     batch_size=opt.test_batch_size,
            #     shuffle=False,
            #     num_workers=opt.num_workers,
            #     pin_memory=True
            # )

            # test_images = [test_dataset[i] for i in range(0, min(300, 914), 75)]
            test_images = test_dataset
            # test_images = [test_dataset[i] for i in range(0, 914, 75)]
            low_dose = torch.stack([torch.from_numpy(x[0]) for x in test_images], dim=0).cuda()
            full_dose = torch.stack([torch.from_numpy(x[1]) for x in test_images], dim=0).cuda()
            self.test_images = (low_dose, full_dose)

            self.test_dataset = test_dataset

    # def set_loader(self):
    #     opt = self.opt
    #     if opt.mode == 'train':
    #     # Ensure your custom dataset is correctly referenced here
    #         train_dataset = dataset_dict['train'](root_dir_h=opt.hq_dir, root_dir_l=opt.lq_dir, length=opt.dataset_length)
    #         train_sampler = RandomSampler(dataset=train_dataset, batch_size=opt.batch_size,
    #                                     num_iter=opt.max_iter, restore_iter=opt.resume_iter)

    #         self.train_loader = torch.utils.data.DataLoader(
    #             dataset=train_dataset,
    #             batch_size=opt.batch_size,
    #             sampler=train_sampler,
    #             shuffle=False,
    #             drop_last=False,
    #             num_workers=opt.num_workers,
    #             pin_memory=True
    #         )
    #     # opt = self.opt

    #     # if opt.mode == 'train':
    #     #     train_dataset = dataset_dict['train'](
    #     #         dataset=opt.train_dataset,
    #     #         test_id=opt.test_id,
    #     #         dose=opt.dose,
    #     #         context=opt.context,
    #     #     )
    #     #     train_sampler = RandomSampler(dataset=train_dataset, batch_size=opt.batch_size,
    #     #                                   num_iter=opt.max_iter,
    #     #                                   restore_iter=opt.resume_iter)

    #     #     train_loader = torch.utils.data.DataLoader(
    #     #         dataset=train_dataset,
    #     #         batch_size=opt.batch_size,
    #     #         sampler=train_sampler,
    #     #         shuffle=False,
    #     #         drop_last=False,
    #     #         num_workers=opt.num_workers,
    #     #         pin_memory=True
    #     #     )
    #         # self.train_loader = train_loader

    #     test_dataset = dataset_dict[opt.test_dataset](
    #         dataset=opt.test_dataset,
    #         test_id=opt.test_id,
    #         dose=opt.dose,
    #         context=opt.context
    #     )
    #     test_loader = torch.utils.data.DataLoader(
    #         dataset=test_dataset,
    #         batch_size=opt.test_batch_size,
    #         shuffle=False,
    #         num_workers=opt.num_workers,
    #         pin_memory=True
    #     )
    #     self.test_loader = test_loader

    #     test_images = [test_dataset[i] for i in range(0, min(300, len(test_dataset)), 75)]
    #     low_dose = torch.stack([torch.from_numpy(x[0]) for x in test_images], dim=0).cuda()
    #     full_dose = torch.stack([torch.from_numpy(x[1]) for x in test_images], dim=0).cuda()
    #     self.test_images = (low_dose, full_dose)

    #     self.test_dataset = test_dataset

    def fit(self):
        opt = self.opt
        if opt.mode == 'train':
            if opt.resume_iter > 0:
                self.logger.load_checkpoints(opt.resume_iter)

            if not hasattr(self, 'test_loader'):
                self.set_loader(mode='test')  # Ensure the test loader is set up

            # Initialize the DataLoader iterator
            loader_iter = iter(self.train_loader)

            # Training routine with tqdm progress bar
            for n_iter in tqdm.trange(opt.resume_iter + 1, opt.max_iter + 1, disable=(self.rank != 0)):
                try:
                    #get the next batch
                    inputs = next(loader_iter)
                except StopIteration:
                    #reset
                    loader_iter = iter(self.train_loader)
                    inputs = next(loader_iter)

                # Proceed with training using the inputs
                self.train(inputs, n_iter)
                if n_iter % opt.save_freq == 0:
                    self.logger.checkpoints(n_iter)
                    self.test(n_iter)
                    self.generate_images(n_iter)

        elif opt.mode == 'test':
            self.logger.load_checkpoints(opt.test_iter)
            self.test(opt.test_iter)
            self.generate_images(opt.test_iter)

        # Additional training modes
        elif opt.mode == 'train_osl_framework':
            self.logger.load_checkpoints(opt.test_iter)
            self.train_osl_framework(opt.test_iter)

        elif opt.mode == 'test_osl_framework':
            self.logger.load_checkpoints(opt.test_iter)
            self.test_osl_framework(opt.test_iter)

    def fit3(self):
        opt = self.opt
        if opt.mode == 'train':
            if opt.resume_iter > 0:
                self.logger.load_checkpoints(opt.resume_iter)

            if not hasattr(self, 'test_loader'):
                self.set_loader(mode='test')  # Ensure the test loader is set up

            # Repeatedly cycle through the training data
            n_iter = opt.resume_iter + 1
            while n_iter <= opt.max_iter:
                try:
                    loader = iter(self.train_loader)
                    while True:
                        inputs = next(loader)
                        self.train(inputs, n_iter)
                        if n_iter % opt.save_freq == 0:
                            self.logger.checkpoints(n_iter)
                            self.test(n_iter)
                            self.generate_images(n_iter)
                        n_iter += 1
                        if n_iter > opt.max_iter:
                            break
                except StopIteration:
                    # Restart from the beginning if the end of the dataset is reached
                    continue

        elif opt.mode == 'test':
            self.logger.load_checkpoints(opt.test_iter)
            self.test(opt.test_iter)
            self.generate_images(opt.test_iter)

        # Additional training modes
        elif opt.mode == 'train_osl_framework':
            self.logger.load_checkpoints(opt.test_iter)
            self.train_osl_framework(opt.test_iter)

        elif opt.mode == 'test_osl_framework':
            self.logger.load_checkpoints(opt.test_iter)
            self.test_osl_framework(opt.test_iter)

    def fit2(self):
        opt = self.opt
        if opt.mode == 'train':
            if opt.resume_iter > 0:
                self.logger.load_checkpoints(opt.resume_iter)

            if not hasattr(self, 'test_loader'):
                self.set_loader(mode='test')  # Make sure this method can handle mode parameter and set up test_loader appropriately.

            # training routine
            loader = iter(self.train_loader)
            for n_iter in tqdm.trange(opt.resume_iter + 1, opt.max_iter + 1, disable=(self.rank != 0)):
                inputs = next(loader)
                self.train(inputs, n_iter)
                if n_iter % opt.save_freq == 0:
                    self.logger.checkpoints(n_iter)
                    self.test(n_iter)
                    self.generate_images(n_iter)

        elif opt.mode == 'test':
            self.logger.load_checkpoints(opt.test_iter)
            self.test(opt.test_iter)
            self.generate_images(opt.test_iter)
        

        # train one-shot learning framework
        elif opt.mode == 'train_osl_framework':
            self.logger.load_checkpoints(opt.test_iter)
            self.train_osl_framework(opt.test_iter)

        # test one-shot learning framework
        elif opt.mode == 'test_osl_framework':
            self.logger.load_checkpoints(opt.test_iter)
            self.test_osl_framework(opt.test_iter)


    def set_model(opt):
        pass

    def train(self, inputs, n_iter):
        self.model.to(self.opt.device)  # Make sure the model is on the right device
        opt = self.opt
        self.model.train()
        self.ema_model.train()
        # print("Input shape:", inputs['LQ'].shape) 
        low_dose = inputs['LQ'].float().to(self.opt.device)  # Make sure to convert and send to device
        full_dose = inputs['HQ'].float().to(self.opt.device)
        # print("Low dose shape:", low_dose.shape)
        # low_dose, full_dose = inputs
        # low_dose, full_dose = low_dose.cuda(), full_dose.cuda()

        ## training process of CoreDiff
        gen_full_dose, x_mix, gen_full_dose_sub1, x_mix_sub1 = self.model(
            low_dose, full_dose, n_iter,
            only_adjust_two_step=opt.only_adjust_two_step,
            start_adjust_iter=opt.start_adjust_iter
        )

        loss = 0.5 * self.lossfn(gen_full_dose, full_dose) + 0.5 * self.lossfn_sub1(gen_full_dose_sub1, full_dose)
        loss.backward()

        if opt.wandb:
            if n_iter == opt.resume_iter + 1:
                wandb.init(project="your wandb project name")

        self.optimizer.step()
        self.optimizer.zero_grad()

        lr = self.optimizer.param_groups[0]['lr']
        loss = loss.item()
        self.logger.msg([loss, lr], n_iter)

        if opt.wandb:
            wandb.log({'epoch': n_iter, 'loss': loss})

        if n_iter % self.update_ema_iter == 0:
            self.step_ema(n_iter)

    @torch.no_grad()
    def test(self, n_iter):
        opt = self.opt
        self.ema_model.eval()

        psnr= []
        ssim=[]
        rmse =[]
        for batch_samples in tqdm.tqdm(self.test_loader, desc='test'):
            low_dose = batch_samples['LQ'].to(self.opt.device)
            full_dose = batch_samples['HQ'].to(self.opt.device)
            file_name = batch_samples['vol']
            maxs = batch_samples['max']
            mins = batch_samples['min']

            gen_full_dose, direct_recons, imstep_imgs = self.ema_model.sample(
                batch_size = low_dose.shape[0],
                img = low_dose,
                t = self.T,
                sampling_routine = self.sampling_routine,
                n_iter=n_iter,
                start_adjust_iter=opt.start_adjust_iter,
            )

            full_dose = self.transfer_calculate_window(full_dose)
            gen_full_dose = self.transfer_calculate_window(gen_full_dose)
        
            # data_range = full_dose.max() - full_dose.min()
            for m in range(self.batch_size):
                file_name1 = file_name[m]
                file_name1 = file_name1.replace(".IMA", ".tif")
                im = Image.fromarray(gen_full_dose[m, 0, :, :])
                im.save('reconstructed_images/test/' + file_name1)
            gen_visualization_files(gen_full_dose, full_dose, low_dose, file_name, "test", maxs, mins)

            msssim, rmse_score = compute_measure(full_dose, gen_full_dose)
            # psnr += psnr_score / len(self.test_loader)
            msssim += msssim / len(self.test_loader)
            rmse += rmse_score / len(self.test_loader)
        print("~~~~~~~~~~~~~~~~~~ everything completed ~~~~~~~~~~~~~~~~~~~~~~~~")
        data1 = np.loadtxt('./visualize/test/mse_loss_target_out')
        print("size of metrics: " + str(data1.shape))
        data2 = np.loadtxt('./visualize/test/msssim_loss_target_out')
        # print("size of out target: " + str(data2.shape))
        # data3 = np.loadtxt('./visualize/test/ssim_loss_target_out')
        # print("size of out target: " + str(data3.shape))
        # data3 = np.append(data1, data2)
        # print("size of append target: " + str(data3.shape))
        print("Final avergae MSE: ", np.average(data1), "std dev.: ", np.std(data1))
        print("Final average MSSSIM: " + 100 * np.average(data2), 'std dev : ', np.std(data2))

    @torch.no_grad()
    def generate_images(self, n_iter):
        opt = self.opt
        self.ema_model.eval()

        for i, batch_samples in enumerate(self.test_loader):
            low_dose = batch_samples['LQ'].to(self.opt.device)
            full_dose = batch_samples['HQ'].to(self.opt.device)

            gen_full_dose, direct_recons, imstep_imgs = self.ema_model.sample(
                batch_size = low_dose.shape[0],
                img = low_dose,
                t = self.T,
                sampling_routine = self.sampling_routine,
                n_iter=n_iter,
                start_adjust_iter=opt.start_adjust_iter,
            )

            # Optionally perform any necessary transformations for display
            low_dose = self.transfer_display_window(low_dose)
            full_dose = self.transfer_display_window(full_dose)
            gen_full_dose = self.transfer_display_window(gen_full_dose)

            b, c, w, h = low_dose.size()
            fake_imgs = torch.stack([low_dose, full_dose, gen_full_dose])
            fake_imgs = fake_imgs.transpose(1, 0).reshape((-1, c, w, h))

            # Save or show images
            self.logger.save_image(torchvision.utils.make_grid(fake_imgs, nrow=3),
                                n_iter, 'test_{}_{}'.format(self.dose, self.sampling_routine) + '_' + opt.test_dataset)

    def adjust_learning_rate(self, n_iter):
        opt = self.opt
        pass

    # denormalize to [0, 255] for calculating PSNR, SSIM and RMSE
    def transfer_calculate_window(self, img, MIN_B=-1024, MAX_B=3072, cut_min=-1000, cut_max=1000):
        img = img * (MAX_B - MIN_B) + MIN_B
        img[img < cut_min] = cut_min
        img[img > cut_max] = cut_max
        img = 255 * (img - cut_min) / (cut_max - cut_min)
        return img

    # denormalize to [-100, 200]HU for display
    def transfer_display_window(self, img, MIN_B=-1024, MAX_B=3072, cut_min=-100, cut_max=200):
        img = img * (MAX_B - MIN_B) + MIN_B
        img[img < cut_min] = cut_min
        img[img > cut_max] = cut_max
        img = (img - cut_min) / (cut_max - cut_min)
        return img

