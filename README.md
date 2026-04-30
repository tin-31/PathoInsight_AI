# PathoInsight AI 🧬
**Hệ thống hỗ trợ chẩn đoán ung thư qua ảnh giải phẫu bệnh**
*Dự án tham dự Hội thi Tin học trẻ toàn quốc lần thứ XXXII - 2026*

## 🌟 Giới thiệu
PathoInsight AI ứng dụng kiến trúc **Sparse Routing Attention** để phân tích các mẫu ảnh mô bệnh học khổng lồ, giúp bác sĩ xác định nhanh chóng các vùng tế bào ác tính với độ chính xác cao và có khả năng giải thích (XAI).

## 🛠 Cấu trúc dự án
- `src/core/`: Thuật toán cốt lõi và kiến trúc mạng Neural.
- `src/ui/`: Giao diện người dùng Streamlit cho bác sĩ[cite: 1].
- `src/utils/`: Tiền xử lý dữ liệu và tạo bản đồ nhiệt[cite: 1].

## 🚀 Hướng dẫn cài đặt
1. Cài đặt thư viện: `pip install -r requirements.txt`
2. Chạy giao diện Web: `streamlit run src/ui/app.py`
3. Chạy kiểm tra nhanh: `python main.py`

## 👨‍ Tác giả
- **Thí sinh:** Lê Vũ Anh Tin
- **Bảng dự thi:** D3 - Sản phẩm sáng tạo
