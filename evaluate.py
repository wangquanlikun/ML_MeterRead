import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')
from transformers import ViTForImageClassification, ViTImageProcessor

# 全局配置
CONFIG = {
    'CSV_PATH': './flag.csv',
    'IMG_DIR': './Dataset',
    'OUTPUT_DIR': './output',
    'MODEL_PATH': './vit-base-patch16-224-in21k',
    'EXPECTED_DIGITS': 6,  # 电表固定为6位数字
    'DEBUG': False,  # 是否显示调试图像
    'MIN_CONFIDENCE': 0.7,  # ViT模型最低置信度阈值
    'MODEL_INPUT_SIZE': 224,  # ViT模型要求的输入尺寸
    'NUM_CLASSES': 10
}

# 数字识别器类 - 使用纯ViT模型
class DigitRecognizer:
    def __init__(self):
        """初始化数字识别器"""
        # 加载ViT模型
        self.model, self.processor = self.load_vit_model()
        # 类别映射
        self.class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        self.idx_to_class = {idx: cls_name for idx, cls_name in enumerate(self.class_names)}
    
    def load_vit_model(self):
        """加载ViT模型和处理器"""
        try:
            # 加载预训练模型（只在首次训练时加载预训练权重）
            best_model_path = os.path.join(CONFIG['OUTPUT_DIR'], 'best_model.pth')
            if os.path.exists(best_model_path):
                # 断点续训或评估时加载自己保存的权重
                model = ViTForImageClassification.from_pretrained(
                    CONFIG['MODEL_PATH'],
                    num_labels=CONFIG['NUM_CLASSES'],
                    ignore_mismatched_sizes=True,
                    local_files_only=True
                )
                print("加载已训练权重")
                model.load_state_dict(torch.load(best_model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
            else:
                # 首次训练加载预训练ViT主干
                model = ViTForImageClassification.from_pretrained(
                    CONFIG['MODEL_PATH'],
                    num_labels=CONFIG['NUM_CLASSES'],
                    ignore_mismatched_sizes=True,
                    local_files_only=True
                )
                print("加载ViT预训练权重")
            processor = ViTImageProcessor.from_pretrained(CONFIG['MODEL_PATH'])
            
            # 设置为评估模式
            model.eval()
            
            # 获取模型要求的输入尺寸
            self.model_input_size = processor.size.get('height', 224)
            print(f"ViT模型要求输入尺寸: {self.model_input_size}x{self.model_input_size}")
            
            # 如果有GPU，将模型移到GPU
            if torch.cuda.is_available():
                model = model.to('cuda')
                print("ViT模型已加载到GPU")
            else:
                print("ViT模型已加载到CPU")
            
            return model, processor
            
        except Exception as e:
            raise RuntimeError(f"加载ViT模型失败: {e}")

    def segment_digits(self, roi):
        """等分为6块进行数字分割（不使用二值化）"""
        # 1. 转为灰度图
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 显示原始灰度图
        if CONFIG['DEBUG']:
            plt.figure(figsize=(10, 4))
            plt.imshow(gray, cmap='gray')
            plt.title("原始灰度图")
            plt.show()
        
        # 2. 直接等分为6块
        height, width = gray.shape
        digit_width = width // 6
        
        digit_rois = []
        valid_digits = []  # 用于DEBUG显示
        
        for i in range(6):
            start_col = i * digit_width
            end_col = (i + 1) * digit_width if i < 5 else width  # 最后一块取到末尾
            
            # 提取当前数字区域
            digit_img = gray[:, start_col:end_col]
            
            # 可选：去除上下空白区域（基于灰度值变化）
            # 计算水平投影，找到数字的上下边界
            h_proj = np.sum(digit_img < 128, axis=1)  # 假设数字较暗
            if np.any(h_proj > 0):
                top = np.argmax(h_proj > 0)
                bottom = len(h_proj) - np.argmax(h_proj[::-1] > 0)
                digit_img = digit_img[top:bottom, :]
            
            # 确保有足够的尺寸
            if digit_img.shape[0] < 10 or digit_img.shape[1] < 5:
                digit_img = np.zeros((50, 30), dtype=np.uint8)  # 创建默认大小的图像
            
            digit_rois.append(digit_img)
            valid_digits.append((start_col, end_col))
        
        # 显示分割结果
        if CONFIG['DEBUG']:
            display_img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).copy()
            for i, (start, end) in enumerate(valid_digits):
                cv2.rectangle(display_img, (start, 0), (end, roi.shape[0]), (0, 255, 0), 2)
                cv2.putText(display_img, str(i), (start+5, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
            plt.figure(figsize=(10, 6))
            plt.imshow(display_img)
            plt.title(f"等分为 {len(valid_digits)} 个数字区域")
            plt.show()
            
            plt.figure(figsize=(12, 4))
            for i, digit_img in enumerate(digit_rois):
                plt.subplot(1, len(digit_rois), i+1)
                plt.imshow(digit_img, cmap='gray')
                plt.title(f"数字 {i+1} (尺寸: {digit_img.shape[1]}x{digit_img.shape[0]})")
                plt.axis('off')
            plt.suptitle("分割出的数字区域")
            plt.show()
        
        return digit_rois
    
    def preprocess_for_vit(self, digit_img):
        """为ViT模型预处理数字图像，使用处理器的预处理功能"""
        try:
            # 转换为RGB图像
            if len(digit_img.shape) == 2:
                rgb_img = cv2.cvtColor(digit_img, cv2.COLOR_GRAY2RGB)
            else:
                rgb_img = digit_img
    
            if CONFIG['DEBUG']:
                print(f"原始分割图像尺寸: {rgb_img.shape[1]}x{rgb_img.shape[0]}")
    
            # 强制resize到ViT模型要求的输入尺寸
            inputs = self.processor(
                images=rgb_img,
                return_tensors="pt",
                size=CONFIG['MODEL_INPUT_SIZE']  # 这里强制指定输入尺寸
            )
    
            return inputs
            
        except Exception as e:
            print(f"预处理失败: {e}")
            return None
    
    def recognize_digit(self, digit_img):
        """使用ViT模型识别数字"""
        try:
            # 预处理图像
            inputs = self.preprocess_for_vit(digit_img)
            
            if inputs is None:
                return 'x', 0.0  # 无效区域
            
            if CONFIG['DEBUG']:
                print(f"输入ViT的图像尺寸: {inputs['pixel_values'].shape}")
            
            # 移到GPU（如果可用）
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            # 预测
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # 获取预测结果
            probabilities = F.softmax(logits, dim=-1)
            k = min(3, logits.shape[-1])
            top_probs, top_idxs = torch.topk(probabilities, k)
            pred_idx = top_idxs[0][0].item()
            confidence = top_probs[0][0].item()
            
            # 转换为类别名
            pred_class = self.idx_to_class.get(pred_idx, 'x')
            
            if CONFIG['DEBUG']:
                # 显示前3个预测结果
                print(f"预测: {pred_class} (置信度: {confidence:.4f})")
                for i in range(1, k):
                    alt_class = self.idx_to_class.get(top_idxs[0][i].item(), 'x')
                    alt_conf = top_probs[0][i].item()
                    print(f"备选 {i}: {alt_class} ({alt_conf:.4f})")
            
            return pred_class, confidence
            
        except Exception as e:
            print(f"ViT识别失败: {e}")
            return 'x', 0.0
    
    def extract_reading(self, img):
        """从图像中提取电表读数"""
        # 分割为数字
        digit_rois = self.segment_digits(img)
        
        # 识别每个数字
        digit_values = []
        confidences = []
        for i, roi in enumerate(digit_rois):
            digit, conf = self.recognize_digit(roi)
            digit_values.append(digit)
            confidences.append(conf)
        
        # 打印识别的数字序列
        if CONFIG['DEBUG']:
            print(f"识别的数字序列: {digit_values}")
            print(f"置信度: {[f'{c:.2f}' for c in confidences]}")
        
        # 处理低置信度数字
        for i in range(len(digit_values)):
            if confidences[i] < CONFIG['MIN_CONFIDENCE']:
                digit_values[i] = 'x'  # 标记为无效
        
        # 组合为完整读数
        valid_digits = [d for d in digit_values if d != 'x']
        
        # 如果有效数字不足，尝试重新组合
        if len(valid_digits) < CONFIG['EXPECTED_DIGITS']:
            # 使用第一个有效数字作为参考，填充缺失位置
            first_valid = next((d for d in digit_values if d != 'x'), '0')
            reading = ''.join([d if d != 'x' else first_valid for d in digit_values])
        else:
            reading = ''.join(digit_values)
        
        # 确保有6位数字
        if len(reading) != 6:
            print(f"警告: 识别数字位数不足 ({len(reading)}), 自动补0")
            reading = reading + '0' * (6 - len(reading))
        
        # 格式化为XXXXX.X
        return f"{reading[:5]}.{reading[5]}"

# 电表读数类
class MeterReader:
    def __init__(self):
        """初始化电表读数识别器"""
        self.digit_recognizer = DigitRecognizer()
    
    def read_meter(self, image_path, bbox=None, csv_data=None):
        """
        读取电表数值
        
        参数:
            image_path: 图像路径
            bbox: 边界框 [x1, y1, x2, y2]，如果为None则从csv_data查找
            csv_data: 包含边界框信息的DataFrame
        """
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像: {image_path}")
            return 0.0
        
        # 如果未提供边界框，从CSV查找
        if bbox is None and csv_data is not None:
            try:
                filename = os.path.basename(image_path)
                row = csv_data[csv_data['filename'] == filename]
                if len(row) > 0:
                    row = row.iloc[0]
                    bbox = [int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])]
                else:
                    print(f"在CSV中未找到图像 {filename} 的边界框")
                    return 0.0
            except Exception as e:
                print(f"从CSV获取边界框失败: {e}")
                return 0.0
        
        if bbox is None:
            print(f"未提供边界框且无法从CSV获取: {image_path}")
            return 0.0
        
        # 提取ROI
        x1, y1, x2, y2 = bbox
        meter_roi = img[y1:y2, x1:x2]
        
        # 识别电表读数
        reading_str = self.digit_recognizer.extract_reading(meter_roi)
        
        try:
            return float(reading_str)
        except:
            print(f"无法转换读数: {reading_str}")
            return 0.0
    
    def evaluate_dataset(self, csv_path, img_dir, output_path=None, sample_count=None):
        """评估整个数据集"""
        # 读取CSV数据
        df = pd.read_csv(csv_path)
        
        # 可选择只处理部分样本
        if sample_count is not None and sample_count < len(df):
            df = df.sample(sample_count, random_state=42)
        
        # 读取所有边界框数据
        results = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="评估数据集"):
            try:
                # 获取图像路径
                img_path = os.path.join(img_dir, row['filename'])
                if not os.path.exists(img_path):
                    print(f"图像不存在: {img_path}")
                    continue
                
                # 获取边界框
                bbox = [int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])]
                
                # 识别电表读数
                pred_value = self.read_meter(img_path, bbox)
                pred_value = float(pred_value)
                
                results.append({
                    'filename': row['filename'],
                    'pred_value': pred_value,
                })
                
                if CONFIG["DEBUG"]:
                    print(f"图像 {row['filename']}: 预测={pred_value}")
                
            except Exception as e:
                print(f"处理 {row['filename']} 时出错: {str(e)}")
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)

        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f"结果已保存到 {output_path}")

# 主函数
def main():
    print("电表读数识别系统 - ViT模型专用版")
    print("=" * 50)
    print(f"使用模型: {CONFIG['MODEL_PATH']}")
    print(f"最低置信度阈值: {CONFIG['MIN_CONFIDENCE']}")
    
    # 创建电表读数识别器
    try:
        meter_reader = MeterReader()
    except Exception as e:
        print(f"初始化失败: {e}")
        return
    
    # 评估数据集
    meter_reader.evaluate_dataset(
        csv_path=CONFIG['CSV_PATH'],
        img_dir=CONFIG['IMG_DIR'],
        output_path=f"{CONFIG['OUTPUT_DIR']}/meter_predictions.csv",
        sample_count=840
    )
    
    print("\n任务完成!")
    print(f"结果保存在: {CONFIG['OUTPUT_DIR']}/meter_predictions.csv")

if __name__ == "__main__":
    main()