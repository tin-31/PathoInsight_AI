import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseRoutingTopK(torch.autograd.Function):
    """
    Cơ chế Sparse Routing tùy chỉnh để lọc các vùng đặc trưng quan trọng nhất.
    Giúp mô hình minh bạch (XAI) và tối ưu hiệu suất tính toán.
    """
    @staticmethod
    def forward(ctx, attention_scores, k):
        # Lấy ra Top-K vùng có điểm chú ý cao nhất
        topk_vals, topk_indices = torch.topk(attention_scores, k, dim=1)
        
        # Tạo mask để loại bỏ các vùng không quan trọng (vùng nhiễu)
        mask = torch.zeros_like(attention_scores).scatter_(1, topk_indices, 1.0)
        
        ctx.save_for_backward(topk_indices)
        ctx.shape = attention_scores.shape
        return attention_scores * mask

    @staticmethod
    def backward(ctx, grad_output):
        topk_indices, = ctx.saved_tensors
        grad_input = torch.zeros(ctx.shape, device=grad_output.device).scatter_(
            1, topk_indices, grad_output.gather(1, topk_indices)
        )
        return grad_input, None

class PathoAttention(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=256, k=16):
        super(PathoAttention, self). __init__()
        self.k = k
        # Cơ chế Gated Attention để tăng độ nhạy với các dấu hiệu bệnh lý
        self.att_v = nn.Linear(input_dim, hidden_dim)
        self.att_u = nn.Linear(input_dim, hidden_dim)
        self.att_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [Số lượng mảnh ảnh, 2048]
        v = torch.tanh(self.att_v(x))
        u = torch.sigmoid(self.att_u(x))
        
        # Tính toán điểm số chú ý (Attention Scores)
        raw_scores = self.att_weights(v * u).T # [1, N_patches]
        
        # Áp dụng Sparse Routing để lọc Top-K
        num_patches = raw_scores.shape[1]
        k_effective = min(self.k, num_patches)
        routed_scores = SparseRoutingTopK.apply(raw_scores, k_effective)
        
        return routed_scores