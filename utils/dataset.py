import os
import os.path as osp
from glob import glob
from torch.utils.data import Dataset
import numpy as np
import torch
from functools import partial
import torch.nn.functional as F
from PIL import Image
import re

def read_correct_image(path):
    offset = 0
    ct_org = None
    with Image.open(path) as img:
        ct_org = np.float32(np.array(img))
        if 270 in img.tag.keys():
            for item in img.tag[270][0].split("\n"):
                if "c0=" in item:
                    loi = item.strip()
                    offset = re.findall(r"[-+]?\d*\.\d+|\d+", loi)
                    offset = (float(offset[1]))
    ct_org = ct_org + offset
    neg_val_index = ct_org < (-1024)
    ct_org[neg_val_index] = -1024
    return ct_org


class CTDatasetMukund(Dataset):
    def __init__(self, root_dir_h, root_dir_l, length):
        self.data_root_l = root_dir_l + "/"
        self.data_root_h = root_dir_h + "/"
        self.img_list_l = sorted(os.listdir(self.data_root_l))
        self.img_list_h = sorted(os.listdir(self.data_root_h))
        self.img_list_l = self.img_list_l[:length]
        self.img_list_h = self.img_list_h[:length]

    def __len__(self):
        return len(self.img_list_l)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs_np = None
        targets_np = None
        rmin = 0
        rmax = 1

        image_input = read_correct_image(self.data_root_l + self.img_list_l[idx])
        image_target = read_correct_image(self.data_root_h + self.img_list_h[idx])

        # Normalize and process the images if necessary
        image_input = ((image_input - np.min(image_input)) / (np.max(image_input) - np.min(image_input))).astype(np.float32)
        image_target = ((image_target - np.min(image_target)) / (np.max(image_target) - np.min(image_target))).astype(np.float32)

        # Reshape to [C, H, W]
        image_input = image_input[np.newaxis, :, :]
        image_target = image_target[np.newaxis, :, :]

        # Repeat the single channel three times to create three channels
        image_input = np.repeat(image_input, 3, axis=0)
        image_target = np.repeat(image_target, 3, axis=0)

        # Convert to PyTorch tensors
        inputs = torch.from_numpy(image_input)
        targets = torch.from_numpy(image_target)

        return {'LQ': inputs, 'HQ': targets}

class CTDataset(Dataset):
    def __init__(self, root_dir_h, root_dir_l,  length, root_hq_vgg3 = None, root_hq_vgg1= None):
        self.data_root_l = root_dir_l + "/"
        self.data_root_h = root_dir_h + "/"
        # self.data_root_h_vgg_3 = root_hq_vgg3 + "/"
        # self.data_root_h_vgg_1 = root_hq_vgg1 + "/"

        self.img_list_l = os.listdir(self.data_root_l)
        self.img_list_h = os.listdir(self.data_root_h)
        # self.vgg_hq_img3 = os.listdir(self.data_root_h_vgg_3)
        # self.vgg_hq_img1 = os.listdir(self.data_root_h_vgg_1)

        self.img_list_l.sort()
        self.img_list_h.sort()
        # self.vgg_hq_img3.sort()
        # self.vgg_hq_img1.sort()

        self.img_list_l = self.img_list_l[0:length]
        self.img_list_h = self.img_list_h[0:length]
        # self.vgg_hq_img_list3 = self.vgg_hq_img3[0:length]
        # self.vgg_hq_img_list1 = self.vgg_hq_img1[0:length]
        self.sample = dict()

    def __len__(self):
        return len(self.img_list_l)


    def __getitem__(self, idx):
        # print("Dataloader idx: ", idx)
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs_np = None
        targets_np = None
        rmin = 0
        rmax = 1

        # print("HQ", self.data_root_h + self.img_list_h[idx])
        # print("LQ", self.data_root_l + self.img_list_l[idx])
        # image_target = read_correct_image("/groups/synergy_lab/garvit217/enhancement_data/train/LQ//BIMCV_139_image_65.tif")
        # print("test")
        # exit()
        image_target = read_correct_image(self.data_root_h + self.img_list_h[idx])
        # print("low quality {} ".format(self.data_root_h + self.img_list_h[idx]))
        # print("high quality {}".format(self.data_root_h + self.img_list_l[idx]))
        # print("hq vgg b3 {}".format(self.data_root_h_vgg + self.vgg_hq_img_list[idx]))
        image_input = read_correct_image(self.data_root_l + self.img_list_l[idx])
        # vgg_hq_img3 = np.load(self.data_root_h_vgg_3 + self.vgg_hq_img_list3[idx]) ## shape : 1,256,56,56
        # vgg_hq_img1 = np.load(self.data_root_h_vgg_1 + self.vgg_hq_img_list1[idx]) ## shape : 1,64,244,244

        input_file = self.img_list_l[idx]  ## low quality image
        assert (image_input.shape[0] == 512 and image_input.shape[1] == 512)
        assert (image_target.shape[0] == 512 and image_target.shape[1] == 512)
        cmax1 = np.amax(image_target)
        cmin1 = np.amin(image_target)
        image_target = rmin + ((image_target - cmin1) / (cmax1 - cmin1) * (rmax - rmin))
        assert ((np.amin(image_target) >= 0) and (np.amax(image_target) <= 1))
        cmax2 = np.amax(image_input)
        cmin2 = np.amin(image_input)
        image_input = rmin + ((image_input - cmin2) / (cmax2 - cmin2) * (rmax - rmin))
        assert ((np.amin(image_input) >= 0) and (np.amax(image_input) <= 1))
        mins = ((cmin1 + cmin2) / 2)
        maxs = ((cmax1 + cmax2) / 2)

        image_input = image_input[np.newaxis, :, :]
        image_target = image_target[np.newaxis, :, :]

        # Repeat the single channel three times to create three channels
        image_input = np.repeat(image_input, 3, axis=0)
        image_target = np.repeat(image_target, 3, axis=0)
        #
        # image_target = image_target.reshape((1, 512, 512))
        # image_input = image_input.reshape((1, 512, 512))
        inputs_np = image_input
        targets_np = image_target

        inputs = torch.from_numpy(inputs_np)
        targets = torch.from_numpy(targets_np)

        inputs = inputs.type(torch.FloatTensor)
        targets = targets.type(torch.FloatTensor)

        # vgg_hq_b3 =  torch.from_numpy(vgg_hq_img3)
        # vgg_hq_b1 =  torch.from_numpy(vgg_hq_img1)
        #
        # vgg_hq_b3 = vgg_hq_b3.type(torch.FloatTensor)
        # vgg_hq_b1 = vgg_hq_b1.type(torch.FloatTensor)

        # print("hq vgg b3 {} b1 {}".format(vgg_hq_b3.shape , vgg_hq_b1.shape))
        self.sample = {'vol': input_file,
                       'HQ': targets,
                       'LQ': inputs,
                       # 'HQ_vgg_op':vgg_hq_b3, ## 1,256,56,56
                       # 'HQ_vgg_b1': vgg_hq_b1,  ## 1,256,56,56
                       'max': maxs,
                       'min': mins}
        return self.sample
    
# class CTDataset(Dataset):
#     def __init__(self, dataset, mode, test_id=9, dose=5, context=True):
#         self.mode = mode
#         self.context = context
#         print(dataset)

#         if dataset in ['mayo_2016_sim', 'mayo_2016']:
#             if dataset == 'mayo_2016_sim':
#                 data_root = './data_preprocess/gen_data/mayo_2016_sim_npy'
#             elif dataset == 'mayo_2016':
#                 data_root = './data_preprocess/gen_data/mayo_2016_npy'
                
#             patient_ids = [67, 96, 109, 143, 192, 286, 291, 310, 333, 506]
#             if mode == 'train':
#                 patient_ids.pop(test_id)
#             elif mode == 'test':
#                 patient_ids = patient_ids[test_id:test_id + 1]

#             patient_lists = []
#             for ind, id in enumerate(patient_ids):
#                 patient_list = sorted(glob(osp.join(data_root, ('L{:03d}_target_'.format(id) + '*_img.npy'))))
#                 patient_lists = patient_lists + patient_list[1:len(patient_list) - 1]
#             base_target = patient_lists

#             patient_lists = []
#             for ind, id in enumerate(patient_ids):
#                 patient_list = sorted(glob(osp.join(data_root, ('L{:03d}_{}_'.format(id, dose) + '*_img.npy'))))
#                 if context:
#                     cat_patient_list = []
#                     for i in range(1, len(patient_list) - 1):
#                         patient_path = ''
#                         for j in range(-1, 2):
#                             patient_path = patient_path + '~' + patient_list[i + j]
#                         cat_patient_list.append(patient_path)
#                     patient_lists = patient_lists + cat_patient_list
#                 else:
#                     patient_list = patient_list[1:len(patient_list) - 1]
#                     patient_lists = patient_lists + patient_list
#             base_input = patient_lists

#         elif dataset == 'mayo_2020':
#             data_root = './data_preprocess/gen_data/mayo_2020_npy'
#             if dose == 10:
#                 patient_ids = ['C052', 'C232', 'C016', 'C120', 'C050']
#             elif dose == 25:
#                 patient_ids = ['L077', 'L056', 'L186', 'L006', 'L148']

#             patient_lists = []
#             for ind, id in enumerate(patient_ids):
#                 patient_list = sorted(glob(osp.join(data_root, (id + '_target_' + '*_img.npy'))))
#                 patient_lists = patient_lists + patient_list[1:len(patient_list) - 1]
#             base_target = patient_lists

#             patient_lists = []
#             for ind, id in enumerate(patient_ids):
#                 patient_list = sorted(glob(osp.join(data_root, (id + '_{}_'.format(dose) + '*_img.npy'))))
#                 if context:
#                     cat_patient_list = []
#                     for i in range(1, len(patient_list) - 1):
#                         patient_path = ''
#                         for j in range(-1, 2):
#                             patient_path = patient_path + '~' + patient_list[i + j]
#                         cat_patient_list.append(patient_path)
#                     patient_lists = patient_lists + cat_patient_list
#                 else:
#                     patient_list = patient_list[1:len(patient_list) - 1]
#                     patient_lists = patient_lists + patient_list
#                 base_input = patient_lists


#         elif dataset == 'piglet':
#             data_root = './data_preprocess/gen_data/piglet_npy'

#             patient_list = sorted(glob(osp.join(data_root, 'piglet_target_' + '*_img.npy')))
#             base_target = patient_list[1:len(patient_list) - 1]

#             patient_list = sorted(glob(osp.join(data_root, 'piglet_{}_'.format(dose) + '*_img.npy')))
#             if context:
#                 cat_patient_list = []
#                 for i in range(1, len(patient_list) - 1):
#                     patient_path = ''
#                     for j in range(-1, 2):
#                         patient_path = patient_path + '~' + patient_list[i + j]
#                     cat_patient_list.append(patient_path)
#                     base_input = cat_patient_list
#             else:
#                 patient_list = patient_list[1:len(patient_list) - 1]
#                 base_input = patient_list


#         elif dataset == 'phantom':
#             data_root = './data_preprocess/gen_data/xnat_npy'

#             patient_list = sorted(glob(osp.join(data_root, 'xnat_target' + '*_img.npy')))[9:21]
#             base_target = patient_list[1:len(patient_list) - 1]

#             patient_list = sorted(glob(osp.join(data_root, 'xnat_{:0>3d}_'.format(dose) + '*_img.npy')))[9:21]
#             if context:
#                 cat_patient_list = []
#                 for i in range(1, len(patient_list) - 1):
#                     patient_path = ''
#                     for j in range(-1, 2):
#                         patient_path = patient_path + '~' + patient_list[i + j]
#                     cat_patient_list.append(patient_path)
#                     base_input = cat_patient_list
#             else:
#                 patient_list = patient_list[1:len(patient_list) - 1]
#                 base_input = patient_list

#         self.input = base_input
#         self.target = base_target
#         print(len(self.input))
#         print(len(self.target))


#     def __getitem__(self, index):
#         input, target = self.input[index], self.target[index]

#         if self.context:
#             input = input.split('~')
#             inputs = []
#             for i in range(1, len(input)):
#                 inputs.append(np.load(input[i])[np.newaxis, ...].astype(np.float32))
#             input = np.concatenate(inputs, axis=0)  #(3, 512, 512)
#         else:
#             input = np.load(input)[np.newaxis, ...].astype(np.float32) #(1, 512, 512)
#         target = np.load(target)[np.newaxis,...].astype(np.float32) #(1, 512, 512)
#         input = self.normalize_(input)
#         target = self.normalize_(target)

#         return input, target

#     def __len__(self):
#         return len(self.target)

#     def normalize_(self, img, MIN_B=-1024, MAX_B=3072):
#         img = img - 1024
#         img[img < MIN_B] = MIN_B
#         img[img > MAX_B] = MAX_B
#         img = (img - MIN_B) / (MAX_B - MIN_B)
#         return img

#
dataset_dict = {
    'train': partial(CTDataset, root_dir_h="/projects/synergy_lab/garvit217/enhancement_data/train/HQ/",
                      root_dir_l="/projects/synergy_lab/garvit217/enhancement_data/train/LQ", length=5120),
    'test': partial(CTDataset, root_dir_h="/projects/synergy_lab/garvit217/enhancement_data/test/HQ/",
                    root_dir_l="/projects/synergy_lab/garvit217/enhancement_data/test/LQ/", length=914),
}
## LANL
# dataset_dict = {
#     'train': partial(CTDataset, root_dir_h="/projects/synergy_lab/garvit217/enhancement_data/train/HQ/",
#                       root_dir_l="/projects/synergy_lab/garvit217/enhancement_data/train/LQ", length=5120),
#     'test': partial(CTDataset, root_dir_h="/projects/synergy_lab/garvit217/enhancement_data/test/HQ/",
#                     root_dir_l="/projects/synergy_lab/garvit217/enhancement_data/test/LQ/", length=914),
# }

# dataset_dict = {
#     'train': partial(CTDataset, root_dir_h="/Users/ayushchaturvedi/Documents/test_data/HQ/",
#                       root_dir_l="/Users/ayushchaturvedi/Documents/test_data/LQ", length=784),
#     'test': partial(CTDataset, root_dir_h="/Users/ayushchaturvedi/Documents/test_data/HQ/",
#                     root_dir_l="/Users/ayushchaturvedi/Documents/test_data/LQ/", length=784),
# }


# dataset_dict = {
#     'train': partial(CTDataset, dataset='mayo_2016_sim', mode='train', test_id=9, dose=5  , context=True),
#     'mayo_2016_sim': partial(CTDataset, dataset='mayo_2016_sim', mode='test', test_id=9, dose=5, context=True),
#     'mayo_2016': partial(CTDataset, dataset='mayo_2016', mode='test', test_id=9, dose=25, context=True),
#     'mayo_2020': partial(CTDataset, dataset='mayo_2020', mode='test', test_id=None, dose=None, context=True),
#     'piglet': partial(CTDataset, dataset='piglet', mode='test', test_id=None, dose=None, context=True),
#     'phantom': partial(CTDataset, dataset='phantom', mode='test', test_id=None, dose=108, context=True),
# }
