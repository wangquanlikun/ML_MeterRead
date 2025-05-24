import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from time import sleep

from train_area import AreaDataset, MeterAreaModel, lossfunc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_areaModel(image_dir, model, test_loader): # 读取实际和训练得出的xmin, xmax, ymin, ymax
    with torch.no_grad():
        for images, area, fname in test_loader:
            train_area = model(images)
            train_xmin, train_xmax, train_ymin, train_ymax = train_area[0][0], train_area[0][1], train_area[0][2], train_area[0][3]
            train_xmin = train_xmin.detach().cpu().numpy()
            train_xmax = train_xmax.detach().cpu().numpy()
            train_ymin = train_ymin.detach().cpu().numpy()
            train_ymax = train_ymax.detach().cpu().numpy()
            xmin, xmax, ymin, ymax = area[0][0], area[0][1], area[0][2], area[0][3]

            print(f"Predicted: {train_xmin:.2f}, {train_xmax:.2f}, {train_ymin:.2f}, {train_ymax:.2f}")
            print(f"Actual: {xmin:.2f}, {xmax:.2f}, {ymin:.2f}, {ymax:.2f}")
            running_loss = lossfunc(train_area, area).item() * images.size(0)
            print(f"Loss: {running_loss:.2f}")

            real_image = Image.open(image_dir + "/" + str(fname[0])).convert('RGB')
            _, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(real_image)
            ax.add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none'))
            ax.add_patch(patches.Rectangle((train_xmin, train_ymin), train_xmax - train_xmin, train_ymax - train_ymin, linewidth=2, edgecolor='b', facecolor='none'))
            ax.set_title(f"Image: {fname[0]}")
            ax.axis('off')
            plt.show()

def evaluate_areaModel_loss(model, test_loader):
    total_loss = 0.0
    with torch.no_grad():
        for images, area, _ in test_loader:
            outputs = model(images)
            loss = lossfunc(outputs, area)
            total_loss += loss.item() * images.size(0)
    average_loss = total_loss / len(test_loader.dataset)
    print(f"Average Loss: {average_loss:.4f}")

if __name__ == "__main__":
    test_csv = "./test_part.csv"
    test_image_dir = "./Dataset"

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = AreaDataset(csv_file=test_csv, img_dir=test_image_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2) # batch_size: 1张图片，shuffle=False 不打乱顺序，num_workers=2 多线程读取数据

    model = MeterAreaModel()
    model.load_state_dict(torch.load('meter_area_final_model.pth', weights_only=True))
    model.eval() # 设置模型为评估模式
    evaluate_areaModel_loss(model, test_loader)
    evaluate_areaModel(test_image_dir, model, test_loader)