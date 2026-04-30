import torch
import torch.nn as nn
from .backbone import PathoBackbone
from .attention import PathoAttention

class PathoInsightNet(nn.Module):
    """
    Kiến trúc tổng thể của hệ thống PathoInsight AI.
    Kết hợp giữa Backbone (ResNet50) và Sparse Attention.
    """
    def __init__(self, top_k=16):
        super(PathoInsightNet, self).__init__()
        # 1. Khởi tạo bộ trích xuất đặc trưng
        self.backbone = PathoBackbone(pretrained=True)
        
        # 2. Khởi tạo bộ lọc vùng chú ý (Sparse Routing)
        self.attention_module = PathoAttention(input_dim=2048, hidden_dim=256, k=top_k)
        
        # 3. Lớp phân loại cuối cùng (Classifier)
        # Kết quả: 0 (Lành tính) hoặc 1 (Ác tính)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        # x: [1, N_patches, 3, H, W] - Dữ liệu dạng 'Bag' trong y tế
        x = x.squeeze(0) # Loại bỏ dimension batch giả: [N_patches, 3, H, W]
        
        # Bước 1: Trích xuất đặc trưng từng mảnh ảnh
        h = self.backbone(x) # [N_patches, 2048]
        
        # Bước 2: Tính toán trọng số chú ý để tìm vùng bệnh
        # routed_scores chứa điểm số của các vùng được chọn (Top-K)
        routed_scores = self.attention_module(h) # [1, N_patches]
        
        # Bước 3: Tổng hợp thông tin (Aggregation)
        # Chỉ những vùng có trọng số > 0 mới đóng góp vào kết luận cuối
        # Điều này giúp loại bỏ nhiễu từ các mô lành hoặc khoảng trắng trên slide
        weights = torch.softmax(routed_scores[routed_scores > 0], dim=0)
        top_k_indices = torch.where(routed_scores[0] > 0)[0]
        
        selected_features = h[top_k_indices]
        # Nhân đặc trưng với trọng số chú ý
        m_represent = torch.mm(weights.unsqueeze(0), selected_features) # [1, 2048]
        
        # Bước 4: Đưa ra dự đoán lâm sàng
        logits = self.classifier(m_represent)
        
        return logits, routed_scores