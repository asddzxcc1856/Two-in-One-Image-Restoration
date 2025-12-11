import os
import random
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor

from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation
import torch


class TrainDataset(Dataset):
    def __init__(self, args):
        super(TrainDataset, self).__init__()
        self.args = args
        self.rs_ids = []
        self.hazy_ids = []
        self.D = Degradation(args)
        self.de_temp = 0
        self.de_type = self.args.de_type
        print(self.de_type)

        self.de_dict = {'derain': 0, 'desnow': 1}

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'desnow' in self.de_type:
            self._init_desnow_ids()

        random.shuffle(self.de_type)

    def _init_clean_ids(self):
        raise NotImplementedError("not implemented")

    def _init_hazy_ids(self):
        raise NotImplementedError("not implemented")

    def _init_deblur_ids(self):
        raise NotImplementedError("not implemented")

    def _init_enhance_ids(self):
        raise NotImplementedError("not implemented")

    def _init_rs_ids(self):
        temp_ids = []
        rs = self.args.data_file_dir + "rainy/rainTrain.txt"
        temp_ids += [self.args.derain_dir + id_.strip() for id_ in open(rs)]
        self.rs_ids = [{"clean_id": x, "de_type": 0} for x in temp_ids]
        self.rs_ids = self.rs_ids * 5

        print(f"Total Rainy Ids : {len(self.rs_ids)}")

    def _init_desnow_ids(self):
        temp_ids = []
        snow = self.args.data_file_dir + "snowy/snowTrain.txt"
        temp_ids += [self.args.desnow_dir + id_.strip() for id_ in open(snow)]
        self.snow_ids = [{"clean_id": x, "de_type": 1} for x in temp_ids]

        print(f"Total Snowy Ids : {len(self.snow_ids)}")

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size,
                        ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size,
                        ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_gt_name_rain(self, rainy_name):
        if "rain-" in rainy_name:
            gt_name = rainy_name.split("rainy")[0] + 'gt/rain_clean-' + rainy_name.split('rain-')[-1]
        elif "snow-" in rainy_name:
            gt_name = rainy_name.split("rainy")[0] +'gt/snow_clean-' + rainy_name.split('snow-')[-1]
        # gt_name = rainy_name.split("rainy")[0]
        # + 'gt/rain_clean-' + rainy_name.split('rain-')[-1]
        else:
            raise Exception(f"Invalid rainy name:{rainy_name}")
        return gt_name

    def _get_gt_name_snow(self, snowy_name):
        if "snow-" in snowy_name:
            gt_name = snowy_name.split("snowy")[0]
            + 'gt/snow_clean-' + snowy_name.split('snow-')[-1]
        else:
            raise Exception("Invalid snowy name")
        return gt_name

    def _merge_ids(self):
        self.sample_ids = []
        if "derain" in self.de_type:
            self.sample_ids += self.rs_ids
        if "desnow" in self.de_type:
            self.sample_ids += self.snow_ids

        print(len(self.sample_ids))
    def _apply_cutmix(self, degrad_patch, clean_patch, alpha=1.0):
        lam = np.random.beta(alpha, alpha)

        B, C, H, W = degrad_patch.size()

        # 隨機裁剪一塊區域
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)

        rand_index = torch.randperm(B)

        degrad_patch[:, :, y1:y2, x1:x2] = degrad_patch[rand_index, :, y1:y2, x1:x2]
        clean_patch[:, :, y1:y2, x1:x2] = clean_patch[rand_index, :, y1:y2, x1:x2]

        lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))

        return degrad_patch, clean_patch, lam

    def _apply_mixup(self, degrad_patch, clean_patch, alpha=0.4):
        """对一批图像应用 MixUp"""
        if alpha <= 0:
            return degrad_patch, clean_patch, 1.0

        lam = np.random.beta(alpha, alpha)

        B, C, H, W = degrad_patch.size()
        rand_index = torch.randperm(B)

        degrad_patch_mix = lam * degrad_patch + (1 - lam) * degrad_patch[rand_index]
        clean_patch_mix = lam * clean_patch + (1 - lam) * clean_patch[rand_index]

        return degrad_patch_mix, clean_patch_mix, lam

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]
        if de_id == 0:  # Rain Streak Removal
            degrad_img = crop_img(np.array(
                Image.open(sample["clean_id"]).convert('RGB')), base=16)
            clean_name = self._get_gt_name_rain(sample["clean_id"])
            clean_img = crop_img(np.array(
                Image.open(clean_name).convert('RGB')), base=16)
        elif de_id == 1:  # snow
            degrad_img = crop_img(np.array(
                Image.open(sample["clean_id"]).convert('RGB')), base=16)
            clean_name = self._get_gt_name_snow(sample["clean_id"])
            clean_img = crop_img(np.array(
                Image.open(clean_name).convert('RGB')), base=16)

        degrad_patch, clean_patch = random_augmentation(
            *self._crop_patch(degrad_img, clean_img))

        degrad_patch = self.toTensor(degrad_patch)
        clean_patch = self.toTensor(clean_patch)

        # --- CutMix ---
        if self.args.use_cutmix:
            degrad_patch, clean_patch, lam_cutmix = self._apply_cutmix(
                degrad_patch.unsqueeze(0),
                clean_patch.unsqueeze(0)
            )
            degrad_patch = degrad_patch.squeeze(0)
            clean_patch = clean_patch.squeeze(0)

        # --- MixUp ---
        if self.args.use_mixup:
            degrad_patch, clean_patch, lam_mixup = self._apply_mixup(
                degrad_patch.unsqueeze(0),
                clean_patch.unsqueeze(0),
                alpha=self.args.mixup_alpha
            )
            degrad_patch = degrad_patch.squeeze(0)
            clean_patch = clean_patch.squeeze(0)


        return [clean_name, de_id], degrad_patch, clean_patch


    def __len__(self):
        return len(self.sample_ids)


class ValDataset(Dataset):
    def __init__(self, args):
        super(ValDataset, self).__init__()
        self.args = args
        self.rs_ids = []
        self.hazy_ids = []
        self.D = Degradation(args)
        self.de_temp = 0
        self.de_type = self.args.de_type
        print(self.de_type)

        self.de_dict = {'derain': 0, 'desnow': 1}

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'desnow' in self.de_type:
            self._init_desnow_ids()

        random.shuffle(self.de_type)

    def _init_clean_ids(self):
        raise NotImplementedError("not implemented")

    def _init_hazy_ids(self):
        raise NotImplementedError("not implemented")

    def _init_deblur_ids(self):
        raise NotImplementedError("not implemented")

    def _init_enhance_ids(self):
        raise NotImplementedError("not implemented")

    def _init_rs_ids(self):
        temp_ids = []
        rs = self.args.data_file_dir + "rainy/rainVal.txt"
        temp_ids += [self.args.derain_dir + id_.strip() for id_ in open(rs)]
        self.rs_ids = [{"clean_id": x, "de_type": 0} for x in temp_ids]

        print(f"Total Rainy Ids : {len(self.rs_ids)}")

    def _init_desnow_ids(self):
        temp_ids = []
        snow = self.args.data_file_dir + "snowy/snowVal.txt"
        temp_ids += [self.args.desnow_dir + id_.strip() for id_ in open(snow)]
        self.snow_ids = [{"clean_id": x, "de_type": 1} for x in temp_ids]

        print(f"Total Snowy Ids : {len(self.snow_ids)}")

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size,
                        ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size,
                        ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_gt_name_rain(self, rainy_name):
        if "rain-" in rainy_name:
            gt_name = rainy_name.split("rainy")[0]
            + 'gt/rain_clean-' + rainy_name.split('rain-')[-1]
        elif "snow-" in rainy_name:
            gt_name = rainy_name.split("rainy")[0]
            + 'gt/snow_clean-' + rainy_name.split('snow-')[-1]
        else:
            raise Exception("Invalid rainy name")
        return gt_name

    def _get_gt_name_snow(self, snowy_name):
        if "snow-" in snowy_name:
            gt_name = snowy_name.split("snowy")[0]
            + 'gt/snow_clean-' + snowy_name.split('snow-')[-1]
        else:
            raise Exception("Invalid snowy name")
        return gt_name

    def _merge_ids(self):
        self.sample_ids = []
        if "derain" in self.de_type:
            self.sample_ids += self.rs_ids
        if "desnow" in self.de_type:
            self.sample_ids += self.snow_ids

        print(len(self.sample_ids))

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]
        if de_id == 0:  # Rain Streak Removal
            degrad_img = crop_img(np.array(
                Image.open(sample["clean_id"]).convert('RGB')), base=16)
            clean_name = self._get_gt_name_rain(sample["clean_id"])
            clean_img = crop_img(np.array(
                Image.open(clean_name).convert('RGB')), base=16)
        elif de_id == 1:  # snow
            degrad_img = crop_img(np.array(
                Image.open(sample["clean_id"]).convert('RGB')), base=16)
            clean_name = self._get_gt_name_snow(sample["clean_id"])
            clean_img = crop_img(np.array(
                Image.open(clean_name).convert('RGB')), base=16)

        degrad_patch, clean_patch = random_augmentation(
            *self._crop_patch(degrad_img, clean_img))

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)

        return [clean_name, de_id], degrad_patch, clean_patch

    def __len__(self):
        return len(self.sample_ids)


class TestSpecificDataset(Dataset):
    def __init__(self, args):
        super(TestSpecificDataset, self).__init__()
        self.args = args
        self.degraded_ids = []
        self._init_clean_ids(args.test_path)

        self.toTensor = ToTensor()

    def _init_clean_ids(self, root):
        extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
        if os.path.isdir(root):
            name_list = []
            for image_file in os.listdir(root):
                if any([image_file.endswith(ext) for ext in extensions]):
                    name_list.append(image_file)
            if len(name_list) == 0:
                raise Exception('Input directory does not contain image files')
            self.degraded_ids += [root + id_ for id_ in name_list]
        else:
            if any([root.endswith(ext) for ext in extensions]):
                name_list = [root]
            else:
                raise Exception('Please pass an Image file')
            self.degraded_ids = name_list
        print("Total Images : {}".format(name_list))

        self.num_img = len(self.degraded_ids)

    def __getitem__(self, idx):
        degraded_img = crop_img(np.array(
            Image.open(self.degraded_ids[idx]).convert('RGB')), base=16)
        name = self.degraded_ids[idx].split('/')[-1][:-4]

        degraded_img = self.toTensor(degraded_img)

        return [name], degraded_img

    def __len__(self):
        return self.num_img
