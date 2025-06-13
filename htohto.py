import os
import argparse
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import segmentation_models_pytorch as smp
from torch import optim
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

# Конфигурация
N_CLASSES = 5  # Фон + 4 класса
TILE_SIZE = 1024  # Размер изображения
BATCH_SIZE = 2
NUM_EPOCHS = 10
LEARNING_RATE = 3e-4
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_folder = os.path.join(root_dir)
        self.mask_folder = os.path.join(root_dir)
        self.image_names = os.listdir(self.image_folder)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_names[idx])
        mask_name = self.image_names[idx].replace('.jpg', '_mask.png')
        mask_path = os.path.join(self.mask_folder, mask_name)

        image = np.array(Image.open(image_path).convert("RGB"))
        # mask = np.array(Image.open(mask_path).convert("L"))

        mask = np.array(Image.open(mask_path))

        assert set(np.unique(mask)).issubset({0, 1, 2, 3, 4}), f"Invalid mask values: {np.unique(mask)}"
        mask = mask.astype(np.int64)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']

        return image, mask

def setup(rank, world_size, master_addr, master_port='12355'):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(
        backend="gloo",  # Используем NCCL для GPU
        init_method="env://",
        rank=rank,
        world_size=world_size
    )

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, args):
    # Инициализация распределённого обучения
    setup(rank, world_size, args.master_addr, args.master_port)

    # Фиксация seed для воспроизводимости
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    random.seed(42 + rank)
    torch.cuda.manual_seed_all(42 + rank)

    # Аугментации
    train_transform = A.Compose([
        A.Resize(TILE_SIZE, TILE_SIZE),
        A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.2, rotate_limit=90, p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.VerticalFlip(),
        A.HorizontalFlip(),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(TILE_SIZE, TILE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Даталодеры
    train_dataset = CustomDataset(root_dir=args.data_dir, transform=train_transform)
    val_dataset = CustomDataset(root_dir=args.val_data_dir, transform=val_transform) if args.val_data_dir else None

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    if val_dataset and rank == 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

    # Модель
    model = smp.UnetPlusPlus(
        encoder_name="timm-efficientnet-b0",
        encoder_weights=None,
        classes=N_CLASSES,
        activation=None #"softmax"
    ).cpu()

    model = DDP(model, device_ids=None, find_unused_parameters=True)

    # Оптимизатор и функция потерь

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    #criterion = smp.losses.JaccardLoss(mode="multiclass")
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Тренировочный цикл
    for epoch in range(NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        model.train()

        epoch_loss = 0.0
        #progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", disable=rank != 0)
        #progress_bar = tqdm(train_loader, desc=f"Rank {rank} | Epoch {epoch+1}/{NUM_EPOCHS}")
        progress_bar = tqdm(
            train_loader,
            desc=f"Rank {rank} | Epoch {epoch+1}/{NUM_EPOCHS}",
            position=rank,  # Каждый процесс пишет в свою строку
            leave=False     # Не оставлять полоску после завершения
        )
        for images, masks in progress_bar:
            images, masks = images.cpu(), masks.cpu().long()

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)

            if masks.dim() == 4:  # Если маски стали [B,1,H,W]
                masks = masks.squeeze(1)  # Делаем [B,H,W]

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        scheduler.step()

        # Валидация (только на главном процессе)
        if val_dataset and rank == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.cpu(), masks.cpu()
                    outputs = model(images)
                    val_loss += criterion(outputs, masks).item()

            print(f"Epoch {epoch+1} | Train Loss: {epoch_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

        # Сохранение модели
        if rank == 0 and (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss/len(train_loader),
            }, f'{args.save_dir}/checkpoint_epoch_{epoch+1}_{loss}.pth')

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distributed Training')
    parser.add_argument('--world_size', type=int, required=True, help='Total number of processes')
    parser.add_argument('--rank', type=int, required=True, help='Process rank')
    parser.add_argument('--master_addr', type=str, required=True, help='Master node IP address')
    parser.add_argument('--master_port', type=str, default='12355', help='Master port')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to training data')
    parser.add_argument('--val_data_dir', type=str, help='Path to validation data')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers per process')

    args = parser.parse_args()

    # Создание директории для сохранения
    if args.rank == 0 and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train(args.rank, args.world_size, args)
