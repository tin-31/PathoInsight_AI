import torch
import os
from src.core.model import PathoInsightNet
from src.utils.preprocess import prepare_bag

def run_inference(image_folder):
    # 1. Khởi tạo thiết bị và mô hình
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PathoInsightNet(top_k=16).to(device)
    model.eval()

    # 2. Thu thập danh sách ảnh
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                   if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_paths:
        print("Không tìm thấy ảnh bệnh phẩm trong thư mục!")
        return

    # 3. Chạy dự đoán
    print(f"Đang phân tích {len(image_paths)} mảnh ảnh...")
    input_bag = prepare_bag(image_paths).to(device)
    
    with torch.no_grad():
        logits, _ = model(input_bag)
        probability = torch.sigmoid(logits).item()

    print("-" * 30)
    print(f"KẾT QUẢ CHẨN ĐOÁN: {'ÁC TÍNH' if probability > 0.5 else 'LÀNH TÍNH'}")
    print(f"Độ tin cậy: {probability*100:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    # Đường dẫn thư mục ảnh test (ngài có thể thay đổi)
    TEST_FOLDER = "data/raw" 
    if os.path.exists(TEST_FOLDER):
        run_inference(TEST_FOLDER)
    else:
        print(f"Vui lòng để ảnh vào thư mục {TEST_FOLDER} để bắt đầu.")