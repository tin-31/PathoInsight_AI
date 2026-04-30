import streamlit as st

def apply_custom_style():
    st.markdown("""
        <style>
        /* Tùy chỉnh phông chữ và nền */
        .stApp {
            background-color: #f4f7f6;
        }
        
        /* Sidebar chuyên nghiệp */
        [data-testid="stSidebar"] {
            background-color: #1e293b;
            color: white;
        }

        /* Thẻ kết quả (Result Cards) */
        .metric-card {
            background-color: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            border-top: 4px solid #3b82f6;
            text-align: center;
        }

        /* Nút bấm (Buttons) */
        .stButton>button {
            border-radius: 8px;
            background-color: #3b82f6;
            color: white;
            font-weight: bold;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            background-color: #2563eb;
            transform: translateY(-2px);
        }
        </style>
    """, unsafe_allow_html=True)