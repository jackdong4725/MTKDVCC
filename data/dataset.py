"""
多数据集混合加载器
支持人群计数 + 通用物体计数数据集
"""
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from config import cfg


class VideoCountDataset(Dataset):
    """
    通用视频帧序列数据集
    适用于：MALL, FDST, ShanghaiTech, VSCrowd等
    """

    def __init__(self, dataset_name, split='train', transform=None):
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform

        # 数据路径
        # 容错：允许大小写/后缀不精确匹配数据集目录
        candidate = cfg.data_root / dataset_name
        if not candidate.exists():
            # 搜索 data_root 下与 dataset_name 模糊匹配的目录
            found = None
            name_low = dataset_name.lower()
            for p in cfg.data_root.iterdir():
                if not p.is_dir():
                    continue
                pn = p.name.lower()
                if pn == name_low or name_low in pn or pn.startswith(name_low) or pn.endswith(name_low):
                    found = p
                    break
            if found is not None:
                candidate = found
        # 兼容不同数据集的 split 目录命名
        desired_split = split
        split_candidates = {
            'train': ['train', 'train_data', 'rgbtrain', 'train_images'],
            'val': ['val', 'test', 'test_data', 'rgbtest', 'test_images']
        }

        split_path = None
        # 先尝试直接使用 candidate/split
        direct = candidate / split
        if direct.exists():
            split_path = direct
        else:
            for alt in split_candidates.get(split, []):
                alt_path = candidate / alt
                if alt_path.exists():
                    split_path = alt_path
                    break

        # 如果仍无合适 split，直接使用 candidate（某些数据集可能是扁平结构）
        if split_path is None:
            split_path = candidate

        self.data_root = split_path

        # 加载样本列表
        self.samples = self._load_samples()

    def _load_samples(self):
        """加载所有样本路径"""
        samples = []

        # 假设数据组织为：dataset_name/split/video_id/frame_xxxx.jpg
        video_dirs = sorted(self.data_root.glob('*'))

        for video_dir in video_dirs:
            if not video_dir.is_dir():
                continue

            # 过滤隐藏/临时文件（例如以 ._ 开头）并只保留常见图像扩展
            frames = []
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
                frames.extend(list(video_dir.glob(ext)))
            # 排序并移除以 '._' 开头或 '.' 隐藏文件
            frames = sorted([p for p in frames if not p.name.startswith('._') and not p.name.startswith('.')])

            # 每 num_frames 帧组成一个样本
            for i in range(0, len(frames) - cfg.num_frames + 1, cfg.num_frames):
                sample = {
                    'frames': frames[i:i + cfg.num_frames],
                    'density_maps': video_dir / 'density' / f'density_{i:04d}.npy',  # 对应密度图
                    'video_id': video_dir.name,
                    'start_frame': i
                }
                samples.append(sample)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载视频帧
        frames = []
        for frame_path in sample['frames']:
            try:
                img = Image.open(frame_path).convert('RGB')
                img = img.resize(cfg.img_size, Image.BILINEAR)
                img = np.array(img) / 255.0
                frames.append(img)
            except Exception:
                # 无法读取的图像使用零填充占位，避免评估/训练崩溃
                frames.append(np.zeros((cfg.img_size[1], cfg.img_size[0], 3), dtype=np.float32))

        frames = np.stack(frames, axis=0)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()

        # 加载密度图
        if sample['density_maps'].exists():
            density_maps = np.load(sample['density_maps'])
            density_maps = torch.from_numpy(density_maps).float()
        else:
            density_maps = torch.zeros(cfg.num_frames, cfg.img_size[0], cfg.img_size[1])

        if self.transform:
            frames = self.transform(frames)

        return {
            'frames': frames,
            'density_maps': density_maps,
            'video_id': sample['video_id'],
            'dataset': self.dataset_name 
        }


class MultiDatasetLoader:
    """
    多数据集混合加载器

    新增参数 `sample_fraction` 可以用来在训练时只使用数据集的部分样本，
    便于进行低数据比例实验。
    """

    def __init__(self, dataset_names, split='train', batch_size=4, shuffle=True, sample_fraction=None):
        self.datasets = []

        for name in dataset_names:
            try:
                dataset = VideoCountDataset(name, split)
                if len(dataset) == 0:
                    print(f"[警告] 数据集 {name} 在拆分 '{split}' 中没有样本，将被忽略。")
                    continue
                self.datasets.append(dataset)
                print(f"[数据加载] {name} - {split}: {len(dataset)} 样本")
            except Exception as e:
                print(f"[警告] 加载 {name} 失败: {e}")

        if len(self.datasets) == 0:
            raise RuntimeError(f"没有可用的数据集。请检查 dataset_names={dataset_names} 和数据路径。")

        # 合并数据集
        self.combined_dataset = ConcatDataset(self.datasets)

        # 根据 sample_fraction 可能只采样部分数据
        if sample_fraction is not None and 0 < sample_fraction < 1.0:
            total = len(self.combined_dataset)
            subset_size = int(total * sample_fraction)
            if subset_size < 1:
                subset_size = 1
            # ensure at least one sample from each individual dataset when possible
            num_ds = len(self.datasets)
            if subset_size < num_ds and total >= num_ds:
                subset_size = num_ds
            # random initial subset
            all_indices = torch.randperm(total).tolist()
            indices = all_indices[:subset_size]

            # helper to map global idx to dataset idx
            cum_sizes = self.combined_dataset.cumulative_sizes
            def which_ds(idx):
                for ds_i, cum in enumerate(cum_sizes):
                    if idx < cum:
                        return ds_i
                return len(cum_sizes)-1

            # ensure each dataset appears at least once
            present = set(which_ds(i) for i in indices)
            missing = set(range(num_ds)) - present
            ptr = subset_size
            while missing and ptr < total:
                ds_i = which_ds(all_indices[ptr])
                if ds_i in missing:
                    # replace a random existing index from a dataset with >1 sample
                    for j, idx in enumerate(indices):
                        if which_ds(idx) not in missing:
                            indices[j] = all_indices[ptr]
                            missing.discard(ds_i)
                            break
                ptr += 1

            sampler = torch.utils.data.SubsetRandomSampler(indices)
            print(f"[数据加载] 使用 {sample_fraction*100:.1f}% 的样本 ({subset_size}/{total})")
            self.loader = DataLoader(
                self.combined_dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=4,
                pin_memory=True
            )
        else:
            self.loader = DataLoader(
                self.combined_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=4,
                pin_memory=True
            )

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)

    @property
    def batch_size(self):
        return self.loader.batch_size
