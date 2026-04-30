import torch
import torchvision.transforms as T
from PIL import Image

def get_transform():
    """
    Quy trình chuẩn hóa ảnh y tế chuyên sâu.
    """
    return T.Compose([
        T.Resize((224, 224)), # Khớp với kích thước đầu vào của ResNet50
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406], # Chuẩn hóa theo ImageNet
            std=[0.229, 0.224, 0.225]
        )
    ])

def prepare_bag(image_paths):
    """
    Chuyển đổi danh sách các mảnh ảnh (patches) thành một 'Bag' dữ liệu 
    để đưa vào mô hình MIL (Multiple Instance Learning).
    """
    transform = get_transform()
    patches = []
    
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        patches.append(transform(img))
    
    # Tạo tensor dạng [1, N_patches, 3, 224, 224]
    bag = torch.stack(patches).unsqueeze(0)
    return bag