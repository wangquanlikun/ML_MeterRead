import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')
from transformers import ViTForImageClassification, ViTImageProcessor, ViTConfig
import shutil
from sklearn.model_selection import train_test_split

# 全局配置
CONFIG = {
    'CSV_PATH': './train_part.csv',
    'IMG_DIR': './Dataset',
    'OUTPUT_DIR': './output',
    'MODEL_PATH': './vit-base-patch16-224-in21k',
    'TRAIN_DATASET_DIR': './digit_dataset',
    'BATCH_SIZE': 32,
    'NUM_EPOCHS': 30,
    'LEARNING_RATE': 2e-5,
    'MODEL_INPUT_SIZE': 224,
    'NUM_CLASSES': 10,  # 0-9
    'TEST_SIZE': 0.2,
    'DEBUG': True
}

# 数字识别器类 - 用于准备训练数据
class DigitExtractor:
    def segment_digits(self, roi):
        """等分为6块进行数字分割（不使用二值化）"""
        # 1. 转为灰度图
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 2. 直接等分为6块
        height, width = gray.shape
        digit_width = width // 6
        
        digit_rois = []
        for i in range(6):
            start_col = i * digit_width
            end_col = (i + 1) * digit_width if i < 5 else width  # 最后一块取到末尾
            
            # 提取当前数字区域
            digit_img = gray[:, start_col:end_col]
            
            # 计算水平投影，找到数字的上下边界
            h_proj = np.sum(digit_img < 128, axis=1)  # 假设数字较暗
            if np.any(h_proj > 0):
                top = np.argmax(h_proj > 0)
                bottom = len(h_proj) - np.argmax(h_proj[::-1] > 0)
                digit_img = digit_img[top:bottom, :]
            
            digit_rois.append(digit_img)
        
        return digit_rois

# 自定义数据集类
class DigitDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.class_names = sorted(os.listdir(dataset_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        self.idx_to_class = {idx: cls_name for idx, cls_name in enumerate(self.class_names)}
        
        # 收集所有图像路径和标签
        self.image_paths = []
        self.labels = []
        
        for class_name in self.class_names:
            class_dir = os.path.join(dataset_dir, class_name)
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 加载图像
        img = Image.open(img_path).convert('L')  # 转换为灰度
        img = Image.merge('RGB', (img, img, img))  # 转换为RGB
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

# 准备训练数据集
def prepare_training_dataset():
    print("准备训练数据集...")
    
    # 创建输出目录
    dataset_dir = CONFIG['TRAIN_DATASET_DIR']
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 创建类别目录
    for digit in range(10):
        os.makedirs(os.path.join(dataset_dir, str(digit)), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, '.'), exist_ok=True)
    
    # 初始化数字提取器
    extractor = DigitExtractor()
    
    # 读取CSV数据
    df = pd.read_csv(CONFIG['CSV_PATH'])
    
    # 计数器
    digit_count = {str(i): 0 for i in range(10)}
    total_images = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理图像"):
        try:
            # 获取图像路径
            img_path = os.path.join(CONFIG['IMG_DIR'], row['filename'])
            if not os.path.exists(img_path):
                continue
            
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # 获取边界框
            bbox = [int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])]
            
            # 提取ROI
            x1, y1, x2, y2 = bbox
            meter_roi = img[y1:y2, x1:x2]
            
            # 分割数字
            digit_rois = extractor.segment_digits(meter_roi)
            
            # 获取真实值
            true_value = str(row['number'])
            # 移除小数点并补齐到6位
            true_value = true_value.replace('.', '')
            if len(true_value) < 6:
                true_value = true_value.zfill(6)
            elif len(true_value) > 6:
                true_value = true_value[:6]
            
            # 保存数字图像
            for i, digit_img in enumerate(digit_rois[:6]):
                if i >= len(true_value):
                    continue
                
                # 获取当前数字的真实标签
                digit_label = true_value[i]
                
                # 保存图像
                save_path = os.path.join(dataset_dir, digit_label, f"{row['filename']}_{i}.png")
                cv2.imwrite(save_path, digit_img)
                
                digit_count[digit_label] += 1
            
            total_images += 1
            
        except Exception as e:
            print(f"处理 {row['filename']} 时出错: {str(e)}")
    
    print("\n数据集统计:")
    print(f"处理的总图像数: {total_images}")
    for digit, count in digit_count.items():
        print(f"数字 '{digit}': {count} 个样本")
    
    return dataset_dir

# 训练ViT模型
def train_vit_model():
    print("训练ViT模型...")
    
    # 准备数据集目录
    dataset_dir = CONFIG['TRAIN_DATASET_DIR']
    
    # 如果数据集不存在，先创建
    if not os.path.exists(dataset_dir) or not any(os.listdir(dataset_dir)):
        dataset_dir = prepare_training_dataset()
    
    # 创建训练和验证数据集
    class_dirs = os.listdir(dataset_dir)
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 为每个类别创建子目录
    for class_name in class_dirs:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
    
    # 分割数据集
    for class_name in class_dirs:
        class_dir = os.path.join(dataset_dir, class_name)
        images = os.listdir(class_dir)
        
        # 分割训练集和验证集
        train_imgs, val_imgs = train_test_split(images, test_size=CONFIG['TEST_SIZE'], random_state=42)
        
        # 复制到训练集
        for img in train_imgs:
            src = os.path.join(class_dir, img)
            dst = os.path.join(train_dir, class_name, img)
            if os.path.isfile(src):  # 只复制文件
                shutil.copy(src, dst)
        
        # 复制到验证集
        for img in val_imgs:
            src = os.path.join(class_dir, img)
            dst = os.path.join(val_dir, class_name, img)
            if os.path.isfile(src):  # 只复制文件
                shutil.copy(src, dst)
    
    # 定义数据转换
    transform = transforms.Compose([
        transforms.Resize((CONFIG['MODEL_INPUT_SIZE'], CONFIG['MODEL_INPUT_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 创建数据集
    train_dataset = DigitDataset(train_dir, transform=transform)
    val_dataset = DigitDataset(val_dir, transform=transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=2)
    
    # 加载预训练模型
    model = ViTForImageClassification.from_pretrained(
        CONFIG['MODEL_PATH'],
        num_labels=CONFIG['NUM_CLASSES'],
        ignore_mismatched_sizes=True,
        local_files_only=True
    )
    # 如果有GPU，将模型移到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"使用设备: {device}")
    
    # 定义优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'])
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    best_val_acc = 0.0
    train_losses = []
    val_accs = []
    
    for epoch in range(CONFIG['NUM_EPOCHS']):
        print(f"\nEpoch {epoch+1}/{CONFIG['NUM_EPOCHS']}")
        print('-' * 20)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc="训练"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        
        print(f'训练损失: {epoch_loss:.4f}, 准确率: {epoch_acc:.4f}')
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="验证"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs.logits, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_dataset)
        val_acc = val_correct / val_total
        val_accs.append(val_acc)
        
        print(f'验证损失: {val_loss:.4f}, 准确率: {val_acc:.4f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CONFIG['OUTPUT_DIR'], 'best_model.pth'))
            print(f"保存新的最佳模型，验证准确率: {val_acc:.4f}")
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(CONFIG['OUTPUT_DIR'], 'final_model.pth'))
    print("训练完成! 最终模型已保存")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.title('训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='验证准确率', color='orange')
    plt.title('验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig(os.path.join(CONFIG['OUTPUT_DIR'], 'training_curve.png'))
    plt.show()
    
    return model

# 评估模型
def evaluate_model(model):
    print("评估模型...")
    
    # 加载验证集
    val_dir = os.path.join(CONFIG['TRAIN_DATASET_DIR'], 'val')
    transform = transforms.Compose([
        transforms.Resize((CONFIG['MODEL_INPUT_SIZE'], CONFIG['MODEL_INPUT_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    val_dataset = DigitDataset(val_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=2)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(os.path.join(CONFIG['OUTPUT_DIR'], 'best_model.pth'), map_location='cuda'))
    model = model.to(device)
    model.eval()
    
    # 评估指标
    correct = 0
    total = 0
    class_correct = [0] * CONFIG['NUM_CLASSES']
    class_total = [0] * CONFIG['NUM_CLASSES']
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="评估"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.logits, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 统计每个类别的准确率
            for i in range(labels.size(0)):
                label = labels[i]
                pred = predicted[i]
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1
    
    # 总体准确率
    accuracy = correct / total
    print(f"总体准确率: {accuracy:.4f}")
    
    # 打印每个类别的准确率
    print("\n每个类别的准确率:")
    for i in range(CONFIG['NUM_CLASSES']):
        if class_total[i] > 0:
            class_acc = class_correct[i] / class_total[i]
            class_name = val_dataset.idx_to_class[i]
            print(f"类别 '{class_name}': {class_acc:.4f} ({class_correct[i]}/{class_total[i]})")
        else:
            print(f"类别 {i}: 无样本")
    
    # 保存评估结果
    eval_results = {
        'overall_accuracy': accuracy,
        'class_accuracy': {val_dataset.idx_to_class[i]: class_correct[i]/class_total[i] 
                            for i in range(CONFIG['NUM_CLASSES']) if class_total[i] > 0}
    }
    
    return eval_results

# 主函数
def main():
    print("电表数字识别模型训练")
    print("=" * 50)
    
    # 训练模型
    trained_model = train_vit_model()
    
    # 评估模型
    evaluate_model(trained_model)
    
    print("\n训练完成!")

if __name__ == "__main__":
    main()