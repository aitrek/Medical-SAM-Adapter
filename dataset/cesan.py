import json
import os
import cv2
import random
import numpy as np
import torch
import torchvision.transforms as transforms

from typing import List, Tuple
from PIL import Image
from torch.utils.data import Dataset
from utils import random_box, random_click


class CESAN(Dataset):

    def __init__(self, data_root, mode='Training', seed: int = 42,
                 test_size: float = 0.1, test_sample_rate: float = None,
                 prompt='click', excluded: List[str] = None,
                 min_mask_region_area: float = 20):

        self.data_root = data_root
        self.excluded = excluded if excluded else []
        self.min_mask_region_area = min_mask_region_area
        self.mode = mode
        self.seed = seed
        self.test_size = test_size
        self.test_smaple_rate = test_sample_rate
        self.prompt = prompt
        self.data_list, self.label_list = self._read_data()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.transform_msk = transforms.Compose([
            transforms.ToTensor(),
        ])

    def _read_data(self) -> Tuple[List[str], List[Tuple[str, int]]]:
        data_paths = []
        label_paths = []
        for name in os.listdir(self.data_root):
            if name in self.excluded:
                print(f"skip excluded dataset: {name}")
                continue

            if name.startswith(".") or name.startswith("__"):
                continue

            dataset_dir = os.path.join(self.data_root, name)
            if not os.path.isdir(dataset_dir):
                continue
            print(f"load {self.mode} dataset: {dataset_dir}")
            split_path = os.path.join(
                dataset_dir,
                f"split_seed-{self.seed}_test_size-{self.test_size}.json"
            )
            if not os.path.exists(split_path):
                print(f"split file not exists: {split_path}")
                continue

            with open(split_path) as f:
                split_data = json.load(f)

            if "train" in self.mode.lower():
                data_list = split_data["train"]
            else:
                data_list = random.choices(
                    split_data["test"],
                    k=int(len(split_data["test"]) * self.test_smaple_rate)
                )

            for data_path, label_path in data_list:
                data_path = os.path.join(dataset_dir, data_path)
                label_path = os.path.join(dataset_dir, label_path)

                mask = np.load(label_path)
                mask_vals = [
                    _ for _ in np.unique(mask)
                    if _ != 0 and (mask == _).sum() > self.min_mask_region_area
                ]
                if not mask_vals:
                    continue
                for mask_val in mask_vals:
                    data_paths.append(data_path)
                    label_paths.append((label_path, mask_val))

        return data_paths, label_paths

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        point_label = 1
        img_path = self.data_list[index]
        mask_path, mask_val = self.label_list[index]
        name = os.path.basename(img_path)

        img = Image.open(img_path).convert('RGB')
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = np.load(mask_path)
        mask[mask != mask_val] = 0
        mask[mask == mask_val] = 255
        mask = mask.astype(np.uint8)
        mask = Image.fromarray(mask)

        if self.prompt == 'click':
            point_label, pt = random_click(np.array(mask) / 255, point_label)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)

            if self.transform_msk:
                mask = self.transform_msk(mask).int()

        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj': name}

        return {
            'image': img,
            'label': mask,
            'p_label': point_label,
            'pt': pt,
            'image_meta_dict': image_meta_dict,
        }

if __name__ == "__main__":
    dataset = CESAN(
        data_root="/Users/zhaojq/Datasets/ALL_Multi",
    )
    print(len(dataset))