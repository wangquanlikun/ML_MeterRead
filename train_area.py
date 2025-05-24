import os
import time
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AreaDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.images = {}  # 缓存图像
    
    def __len__(self):
        return len(self.data)
    
    def preload_images(self):
        for idx in range(len(self.data)):
            img_name = self.data.iloc[idx]['filename']
            img_path = os.path.join(self.img_dir, img_name)
            self.images[idx] = Image.open(img_path).convert('RGB')
    
    def __getitem__(self, idx):
        if idx in self.images:
            image = self.images[idx]
            img_name = self.data.iloc[idx]['filename']
        else:
            img_name = self.data.iloc[idx]['filename']
            img_path = os.path.join(self.img_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            self.images[idx] = image  # 缓存图像

        xmin = int(self.data.iloc[idx]['xmin'])
        xmax = int(self.data.iloc[idx]['xmax'])
        ymin = int(self.data.iloc[idx]['ymin'])
        ymax = int(self.data.iloc[idx]['ymax'])

        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor([xmin, xmax, ymin, ymax], dtype=torch.float16), img_name


class MeterAreaModel(nn.Module):
    def __init__(self, input_channels=3, dropout=0.2):
        super(MeterAreaModel, self).__init__()
        self.features = nn.Sequential(
            # Block 1 (224->112)
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2 (112->56)
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3 (56->28)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 4 (28->14)
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 4),
            nn.Sigmoid()  # 输出范围 [0, 1]
        )
    
    def forward(self, x):
        feats = self.features(x)
        coords = self.regressor(feats) * 400.0  # 缩放到 [0, 400]
        return coords
    
def train_areaModel(model, train_loader, criterion, optimizer, device, num_epochs=300):
    model.to(device)
    best_loss = float('inf')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for images, area, _ in train_loader:
            images = images.to(device)
            area = area.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, area)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        scheduler.step(epoch_loss)

        elapsed = time.time() - start_time
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Time: {elapsed:.1f}s')
        loss_record.append(epoch_loss)
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), 'best_meter_area_model.pth')
    
    print(f'Training complete. Best Loss: {best_loss:.4f}')
    model.load_state_dict(torch.load('best_meter_area_model.pth', weights_only=True))
    return model

def lossfunc(pred, target):
    # GIoU Loss
    pred_x1, pred_x2, pred_y1, pred_y2 = pred.unbind(-1)
    target_x1, target_x2, target_y1, target_y2 = target.unbind(-1)
    
    # 计算交集面积
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_y2 = torch.min(pred_y2, target_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # 计算并集面积
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union_area = pred_area + target_area - inter_area
    
    # 计算最小封闭框面积
    enclose_x1 = torch.min(pred_x1, target_x1)
    enclose_x2 = torch.max(pred_x2, target_x2)
    enclose_y1 = torch.min(pred_y1, target_y1)
    enclose_y2 = torch.max(pred_y2, target_y2)
    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
    
    # GIoU计算
    iou = inter_area / (union_area + 1e-6)
    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-6)
    giou_loss = 1 - giou
    
    # 中心点距离损失
    pred_center_x = (pred_x1 + pred_x2) / 2
    pred_center_y = (pred_y1 + pred_y2) / 2
    target_center_x = (target_x1 + target_x2) / 2
    target_center_y = (target_y1 + target_y2) / 2
    center_loss = torch.sqrt((pred_center_x - target_center_x)**2 + (pred_center_y - target_center_y)**2) / 400.0
    
    # 尺寸比例损失
    pred_w = pred_x2 - pred_x1
    pred_h = pred_y2 - pred_y1
    target_w = target_x2 - target_x1
    target_h = target_y2 - target_y1
    size_loss = torch.abs(torch.log(pred_w/target_w)) + torch.abs(torch.log(pred_h/target_h))
    
    # 组合损失
    total_loss = giou_loss + 0.5 * center_loss + 0.1 * size_loss
    return total_loss.mean()

if __name__ == "__main__":
    train_csv = "./train_part.csv"
    train_image_dir = "./Dataset"
    loss_record = []
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), # 统一缩放到 224×224 像素
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # 随机调整亮度、对比度、饱和度、色相，用于增强鲁棒性。
        transforms.ToTensor(), # 将 PIL 图像转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]) # 归一化处理
    ])
    
    train_dataset = AreaDataset(csv_file=train_csv, img_dir=train_image_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)

    areaModel = MeterAreaModel()
    criterion = lossfunc
    optimizer = optim.Adam(areaModel.parameters(), lr=1e-4)
    areaModel = train_areaModel(areaModel, train_loader, criterion, optimizer, device, num_epochs=500)
    torch.save(areaModel.state_dict(), 'meter_area_final_model.pth')
    print("Model training complete. Final model saved as 'meter_area_final_model.pth'.")

    # 画出损失曲线：从第5次迭代开始
    plt.plot(loss_record[5:])
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.show()