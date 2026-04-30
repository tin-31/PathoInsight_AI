import torch
import sys
import os

# Thêm đường dẫn để chạy test
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.core.model import PathoInsightNet

def test_model_output_shape():
    """Kiểm tra xem mô hình có trả về đúng cấu trúc dữ liệu không."""
    model = PathoInsightNet(top_k=5)
    # Giả lập 10 mảnh ảnh (patches) kích thước 224x224
    dummy_input = torch.randn(1, 10, 3, 224, 224)
    
    logits, attention = model(dummy_input)
    
    assert logits.shape == (1, 1), "Lỗi: Đầu ra dự đoán sai kích thước!"
    assert attention.shape[1] == 10, "Lỗi: Trọng số attention sai số lượng mảnh ảnh!"
    print("✅ Kiểm tra Model: ĐẠT")

if __name__ == "__main__":
    test_model_output_shape()