import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class PathoBackbone(nn.Module):
    """
    Module trích xuất đặc trưng hình thái học từ ảnh y tế.
    Sử dụng ResNet50 làm mạng cơ sở để nhận diện cấu trúc tế bào.
    """
    def __init__(self, pretrained=True):
        super(PathoBackbone, self).__init__()
        
        # Tải mô hình ResNet50 với trọng số đã được huấn luyện (ImageNet)
        # Trong thuyết minh, hãy nhấn mạnh việc tinh chỉnh (fine-tuning) cho ảnh Patho
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base_model = resnet50(weights=weights)
        
        # Loại bỏ lớp Fully Connected (FC) cuối cùng và lớp Global Average Pooling
        # Chúng ta chỉ lấy các đặc trưng thô (feature maps)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.flatten = nn.Flatten()

    def forward(self, x):
        # x: [Batch_size, 3, H, W]
        features = self.feature_extractor(x)
        features = self.flatten(features) # Đầu ra: [Batch_size, 2048]
        return features

    def freeze_layers(self, num_layers=6):
        """
        Đóng băng một số tầng đầu để giữ lại khả năng nhận diện hình dạng cơ bản,
        tập trung huấn luyện các tầng sau cho đặc trưng y tế chuyên sâu.
        """
        child_list = list(self.feature_extractor.children())
        for i in range(num_layers):
            for param in child_list[i].parameters():
                param.requires_grad = False