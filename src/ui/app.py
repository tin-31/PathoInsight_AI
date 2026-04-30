import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import gdown
import os
from torchvision.models import resnet18

# =====================================================================
# 1. CẤU HÌNH & TẢI MODEL
# =====================================================================
MODEL_ID = '1J77dVnIjj_iVjDdWIpE3TLWmz063ae4s'
MODEL_PATH = 'clinical_robust_mil.pth' # App sẽ tìm file này trong thư mục gốc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_clinical_model():
    # Kiểm tra nếu file model chưa có thì mới tải từ Google Drive
    if not os.path.exists(MODEL_PATH):
        st.warning("Đang tải model từ Google Drive, vui lòng chờ giây lát...")
        url = f'https://google.com{MODEL_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)
    
    class SparseRoutingTopK(torch.autograd.Function):
        @staticmethod
        def forward(ctx, attention_scores, k):
            topk_vals, topk_indices = torch.topk(attention_scores, k, dim=1)
            mask = torch.zeros_like(attention_scores).scatter_(1, topk_indices, 1.0)
            ctx.save_for_backward(topk_indices)
            ctx.shape = attention_scores.shape
            return attention_scores * mask
        @staticmethod
        def backward(ctx, grad_output):
            topk_indices, = ctx.saved_tensors
            grad_input = torch.zeros(ctx.shape, device=grad_output.device).scatter_(1, topk_indices, grad_output.gather(1, topk_indices))
            return grad_input, None

    class ClinicalGigapixelMIL(nn.Module):
        def __init__(self, top_k=16):
            super().__init__()
            self.top_k = top_k
            self.instance_norm = nn.InstanceNorm2d(3, affine=True)
            resnet = resnet18(weights=None)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
            self.att_V, self.att_U, self.att_weights = nn.Linear(512, 128), nn.Linear(512, 128), nn.Linear(128, 1)
            self.classifier = nn.Linear(512, 1) 

        def forward(self, bag):
            bag = bag.squeeze(0)
            h = self.backbone(self.instance_norm(bag))
            raw_scores = self.att_weights(torch.tanh(self.att_V(h)) * torch.sigmoid(self.att_U(h))).T
            num_patches = raw_scores.shape[1]
            adaptive_k = min(self.top_k, num_patches)
            routed_scores = SparseRoutingTopK.apply(raw_scores, adaptive_k)
            topk_indices = torch.topk(raw_scores, adaptive_k, dim=1)[1]
            A_softmax = F.softmax(routed_scores.gather(1, topk_indices), dim=1)
            M = torch.mm(A_softmax, h[topk_indices.squeeze(0)])
            return self.classifier(M), routed_scores

    model = ClinicalGigapixelMIL(top_k=16).to(device)
    # Map_location giúp chạy được trên cả máy có GPU hoặc chỉ có CPU
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    return model

# =====================================================================
# 2. GIAO DIỆN CHUYÊN NGHIỆP
# =====================================================================
st.set_page_config(page_title="PathoInsight AI | Clinical Support", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #007bff; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .report-card { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; border: 1px solid #dee2e6; }
    h1, h2, h3 { color: #1e293b; }
    </style>
    """, unsafe_allow_html=True)

st.write("## 🧬 PathoInsight AI: Trợ lý Phân tích Giải phẫu bệnh")
st.caption("Phiên bản chạy Local - Hỗ trợ chẩn đoán ngoại tuyến.")

with st.sidebar:
    st.header("📋 Nhập dữ liệu")
    uploaded_files = st.file_uploader("Tải các mảnh cắt (Patches)...", 
                                      type=['png', 'jpg', 'jpeg'], 
                                      accept_multiple_files=True)
    if uploaded_files:
        st.success(f"Đã nạp {len(uploaded_files)} ảnh.")

if uploaded_files:
    model = load_clinical_model()
    
    preprocess = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    patches_list = [preprocess(Image.open(f).convert('RGB')) for f in uploaded_files]
    input_bag = torch.stack(patches_list).unsqueeze(0).to(device)

    with st.spinner('Đang phân tích dữ liệu lâm sàng...'):
        with torch.no_grad():
            logits, attention = model(input_bag)
            prob = torch.sigmoid(logits).item()

    # Báo cáo kết quả
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    with c1:
        color = "#dc3545" if prob > 0.5 else "#198754"
        st.markdown(f"<h1 style='color:{color}; text-align:center; font-size: 60px;'>{prob*100:.1f}%</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;'><b>XÁC SUẤT BỆNH LÝ</b></p>", unsafe_allow_html=True)
    with c2:
        if prob > 0.5:
            st.error("### 🚨 CẢNH BÁO NGUY CƠ CAO")
            st.write("Mẫu bệnh phẩm có dấu hiệu bất thường. Cần kiểm tra kỹ các vùng được đánh dấu trọng số cao.")
        else:
            st.success("### ✅ NGUY CƠ THẤP")
            st.write("Chưa phát hiện dấu hiệu ác tính trong các mảnh cắt hiện tại.")
        st.progress(prob)
    st.markdown('</div>', unsafe_allow_html=True)

    # Hiển thị Attention Heatmaps
    st.subheader("📍 Phân tích vùng tập trung (Attention Map)")
    raw_scores = attention.cpu().numpy().flatten()
    norm_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-8)
    
    n_display = min(4, len(uploaded_files))
    cols = st.columns(n_display)
    top_indices = np.argsort(raw_scores)[-n_display:][::-1]
    
    for i, idx in enumerate(top_indices):
        with cols[i]:
            patch_img = Image.open(uploaded_files[idx])
            fig, ax = plt.subplots()
            ax.imshow(patch_img)
            overlay = np.zeros((*np.array(patch_img).shape[:2], 3))
            overlay[:] = [1, 0, 0] 
            ax.imshow(overlay, alpha=norm_scores[idx] * 0.4) 
            ax.set_title(f"Vùng #{i+1} (Score: {raw_scores[idx]:.3f})")
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)
else:
    st.info("Vui lòng tải ảnh lên từ Sidebar để bắt đầu phân tích.")

st.markdown("---")
st.caption("© 2026 PathoInsight AI | Local Clinical Decision Support System")
