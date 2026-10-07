import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
import json
import os

# ---------------------------------------------------------
# 1. CẤU HÌNH TRANG STREAMLIT
# ---------------------------------------------------------
st.set_page_config(
    page_title="Chẩn Đoán Bệnh Lá Cà Chua AI",
    page_icon="🌱",
    layout="wide"
)

# ---------------------------------------------------------
# 2. THỐNG KÊ 34 TỈNH THÀNH & TỔNG SỐ XÃ/PHƯỜNG SAU SÁP NHẬP (01/07/2025)
# ---------------------------------------------------------
PROVINCE_SUMMARY = {
    "TP Hà Nội": {"total": 126, "sample": ["Phường Ngọc Hà", "Phường Đội Cấn", "Phường Kim Mã", "Phường Thới Hòa", "Xã Tân Hương"]},
    "TP Hồ Chí Minh": {"total": 168, "sample": ["Phường Đông Hưng Thuận", "Phường Hòa Hưng", "Phường Vườn Lài", "Phường Nhiêu Lộc", "Phường Xuân Hòa", "Phường Bình Tây", "Phường Bình Tiên", "Đặc khu Thạnh An"]},
    "TP Hải Phòng": {"total": 114, "sample": ["Phường Minh Khai", "Phường Hoàng Văn Thụ", "Đặc khu Cát Hải", "Xã An Đồng"]},
    "TP Đà Nẵng": {"total": 94, "sample": ["Phường Hải Châu 1", "Phường Thuận Phước", "Xã Tam Hải", "Xã Tân Hiệp"]},
    "TP Cần Thơ": {"total": 103, "sample": ["Phường Tân An", "Phường Tân Lộc", "Xã Trường Long", "Xã Thạnh Phú", "Xã Phong Nẫm"]},
    "TP Huế": {"total": 40, "sample": ["Phường Dương Nỗ", "Phường Thuận Lộc", "Phường Phú Xuân", "Xã Hương Thọ"]},
    "Tỉnh An Giang": {"total": 102, "sample": ["Phường Mỹ Bình", "Xã Mỹ Hòa Hưng", "Xã Bình Giang", "Xã Bình Sơn", "Xã Hòn Nghệ"]},
    "Tỉnh Bắc Ninh": {"total": 99, "sample": ["Phường Suối Hoa", "Xã Tuấn Đạo", "Xã Quế Võ", "Xã Yên Phong"]},
    "Tỉnh Cà Mau": {"total": 64, "sample": ["Phường 1", "Xã Hồ Thị Kỷ", "Xã Đất Mũi", "Xã Tân Thành"]},
    "Tỉnh Cao Bằng": {"total": 56, "sample": ["Phường Hợp Giang", "Xã Bảo Lạc", "Xã Bằng Thành", "Xã Trùng Khánh"]},
    "Tỉnh Đắk Lắk": {"total": 102, "sample": ["Phường Tân An", "Xã Ea H'Leo", "Xã Ea Trang", "Xã Ia Lốp", "Xã Ia Rvê"]},
    "Tỉnh Điện Biên": {"total": 45, "sample": ["Phường Mường Thanh", "Xã Nậm Pồ", "Xã Tủa Chùa", "Xã Mường Lay"]},
    "Tỉnh Đồng Nai": {"total": 95, "sample": ["Phường Phước Tân", "Phường Tam Phước", "Xã Thanh Sơn", "Xã Đak Lua", "Xã Phú Lý"]},
    "Tỉnh Đồng Tháp": {"total": 102, "sample": ["Phường 1", "Phường 2", "Xã Mỹ Tân", "Xã Bình Thành"]},
    "Tỉnh Gia Lai": {"total": 135, "sample": ["Phường Pleiku", "Xã Ia O", "Xã Nhơn Châu", "Xã Ia Púch", "Xã Ia Mơ"]},
    "Tỉnh Hà Tĩnh": {"total": 69, "sample": ["Phường Bắc Hà", "Xã Sơn Kim 1", "Xã Sơn Kim 2", "Xã Thạch Trung"]},
    "Tỉnh Hưng Yên": {"total": 104, "sample": ["Phường Hiến Nam", "Phường Lam Sơn", "Xã Ân Thi", "Xã Nhân Hòa"]},
    "Tỉnh Khánh Hòa": {"total": 65, "sample": ["Phường Lộc Thọ", "Phường Phước Tiến", "Đặc khu Trường Sa", "Xã Phước Đồng"]},
    "Tỉnh Lai Châu": {"total": 38, "sample": ["Phường Tân Phong", "Xã Mù Cả", "Xã Tà Tổng", "Xã Sìn Hồ"]},
    "Tỉnh Lâm Đồng": {"total": 124, "sample": ["Phường 1", "Phường 2", "Xã Quảng Hòa", "Xã Quảng Sơn", "Xã Quảng Trực", "Xã Ninh Gia"]},
    "Tỉnh Lạng Sơn": {"total": 65, "sample": ["Phường Hoàng Văn Thụ", "Phường Tam Thanh", "Xã Đồng Đăng", "Xã Khau San"]},
    "Tỉnh Lào Cai": {"total": 99, "sample": ["Phường Sa Pa", "Xã Nậm Xé", "Xã Ngũ Chỉ Sơn", "Xã Chế Tạo", "Xã Lao Chải"]},
    "Tỉnh Nghệ An": {"total": 130, "sample": ["Phường Quang Trung", "Xã Keng Đu", "Xã Mỹ Lý", "Xã Bắc Lý", "Xã Huồi Tụ"]},
    "Tỉnh Ninh Bình": {"total": 129, "sample": ["Phường Nam Thành", "Phường Đông Thành", "Xã Gia Vân", "Xã Khánh Nhai"]},
    "Tỉnh Phú Thọ": {"total": 148, "sample": ["Phường Tân Dân", "Xã Thu Cúc", "Xã Trung Sơn", "Xã Hy Nông"]},
    "Tỉnh Quảng Ngãi": {"total": 96, "sample": ["Phường Nguyễn Nghiêm", "Xã Đăk Long", "Xã Ba Xa", "Xã Rờ Kơi", "Xã Mô Rai"]},
    "Tỉnh Quảng Ninh": {"total": 54, "sample": ["Phường Hồng Gai", "Phường Bãi Cháy", "Xã Cái Chiên", "Đặc khu Vân Đồn"]},
    "Tỉnh Quảng Trị": {"total": 78, "sample": ["Phường 1", "Xã Tân Thành", "Đặc khu Cồn Cỏ", "Xã Hướng Lập"]},
    "Tỉnh Sơn La": {"total": 75, "sample": ["Phường Quyết Thắng", "Xã Mường Lạn", "Xã Phiêng Khoài", "Xã Suối Tọ", "Xã Ngọc Chiến"]},
    "Tỉnh Tây Ninh": {"total": 96, "sample": ["Phường Tân Phú", "Phường Tiến Thành", "Xã Bình Minh", "Xã Tân Bình"]},
    "Tỉnh Thái Nguyên": {"total": 92, "sample": ["Phường Phan Đình Phùng", "Xã Sảng Mộc", "Xã Thượng Quan", "Xã Linh Sơn"]},
    "Tỉnh Thanh Hóa": {"total": 166, "sample": ["Phường Lam Sơn", "Xã Phú Xuân", "Xã Mường Chanh", "Xã Quang Chiểu", "Xã Tam Chung"]},
    "Tỉnh Tuyên Quang": {"total": 124, "sample": ["Phường Minh Xuân", "Xã Trung Hà", "Xã Kiến Thiết", "Xã Hùng Đức", "Xã Minh Sơn"]},
    "Tỉnh Vĩnh Long": {"total": 124, "sample": ["Phường 1", "Xã Long Hòa", "Xã Đông Hải", "Xã Long Vĩnh", "Xã Hòa Minh"]}
}

# Tải file JSON chứa đầy đủ 3.321 xã/phường nếu có trong thư mục project
FULL_DATA_FILE = "dia_chinh_3321.json"
ADMIN_FULL_DATA = {}

if os.path.exists(FULL_DATA_FILE):
    try:
        with open(FULL_DATA_FILE, "r", encoding="utf-8") as f:
            ADMIN_FULL_DATA = json.load(f)
    except Exception:
        ADMIN_FULL_DATA = {}

# ---------------------------------------------------------
# 3. BẢNG MÔ TẢ VÀ ĐIỀU TRỊ 11 LỚP BỆNH
# ---------------------------------------------------------
DISEASE_INFO = {
    "Bacterial_spot": {
        "vn_name": "Bệnh Đốm Vi Khuẩn (Bacterial Spot)",
        "symptoms": "Xuất hiện các đốm nhỏ màu nâu đen, mọng nước trên bề mặt lá.",
        "remedy": "Sử dụng thuốc gốc đồng (Copper Hydroxide hoặc Kasugamycin). Tỉa bớt lá già để tạo độ thông thoáng."
    },
    "Early_blight": {
        "vn_name": "Bệnh Đốm Vòng / Cháy Lá Sớm (Early Blight)",
        "symptoms": "Vết bệnh hình tròn có các vòng đồng tâm màu nâu đen, lá bị vàng xung quanh.",
        "remedy": "Phun thuốc gốc Mancozeb, Chlorothalonil hoặc Difenoconazole. Luân canh cây trồng."
    },
    "Late_blight": {
        "vn_name": "Bệnh Sương Mai / Cháy Lá Muộn (Late Blight)",
        "symptoms": "Vết đốm màu xám xanh mọng nước, phát triển nhanh làm cháy khô lá.",
        "remedy": "Sử dụng Metalaxyl, Dimethomorph hoặc Ridomil Gold. Tránh tưới nước lên lá vào buổi tối."
    },
    "Leaf_mold": {
        "vn_name": "Bệnh Mốc Lá (Leaf Mold)",
        "symptoms": "Mặt trên lá có đốm vàng nhạt, mặt dưới xuất hiện lớp mốc màu xám xỉn hoặc nâu nhạt.",
        "remedy": "Phun thuốc gốc Đồng hoặc Carbendazim. Giảm độ ẩm nhà kính, tăng cường thông gió."
    },
    "Septoria_leaf_spot": {
        "vn_name": "Bệnh Đốm Lá Septoria (Septoria Leaf Spot)",
        "symptoms": "Các đốm tròn nhỏ màu xám nhạt ở giữa, viền nâu đen xuất hiện nhiều ở lá gốc.",
        "remedy": "Phun Azoxystrobin hoặc Difenoconazole. Thu gom và tiêu hủy lá bệnh dưới gốc."
    },
    "Spider_mites": {
        "vn_name": "Nhện Đỏ Hại Lá (Two-Spotted Spider Mites)",
        "symptoms": "Lá xuất hiện các chấm nhỏ lốm đốm màu vàng, mặt dưới lá có màng tơ mỏng.",
        "remedy": "Phun thuốc đặc trị nhện như Abamectin, Hexythiazox hoặc Propargite. Tăng độ ẩm vườn."
    },
    "Target_Spot": {
        "vn_name": "Bệnh Đốm Mục Tiêu (Target Spot)",
        "symptoms": "Vết bệnh hình tròn màu nâu với tâm sáng hơn, tạo hình dạng giống bia bắn.",
        "remedy": "Sử dụng thuốc trừ nấm chứa Chlorothalonil hoặc Pyraclostrobin."
    },
    "Yellow_Leaf_Curl_Virus": {
        "vn_name": "Bệnh Xoăn Lá Vàng Do Virus (Yellow Leaf Curl Virus)",
        "symptoms": "Lá bị xoăn ngửa lên trên, phiến lá nhỏ lại, màu vàng chanh, cây lùn còi cọc.",
        "remedy": "Chưa có thuốc trị virus. Cần tiêu diệt bọ phấn trắng (môi giới truyền bệnh) bằng Imidacloprid."
    },
    "Mosaic_virus": {
        "vn_name": "Bệnh Khảm Lá Do Virus (Mosaic Virus)",
        "symptoms": "Lá loang lổ các vệt màu xanh đậm lục nhạt xen kẽ, lá bị biến dạng nhăn nheo.",
        "remedy": "Nhổ bỏ cây bệnh để tránh lây lan. Khử trùng dụng cụ cắt tỉa và kiểm soát rệp muỗi."
    },
    "Powdery_mildew": {
        "vn_name": "Bệnh Phấn Trắng (Powdery Mildew)",
        "symptoms": "Lớp bột trắng như phấn bao phủ trên bề mặt lá, làm lá khô xơ và rụng.",
        "remedy": "Phun Sulfur (Lưu huỳnh), Hexaconazole hoặc Myclobutanil."
    },
    "Healthy": {
        "vn_name": "Lá Khỏe Mạnh (Healthy)",
        "symptoms": "Lá xanh tươi, không có dấu hiệu bị nấm, vi khuẩn hay sâu bệnh tấn công.",
        "remedy": "Tiếp tục chăm sóc, tưới nước và bón phân định kỳ cân đối N-P-K."
    }
}

# ---------------------------------------------------------
# 4. LOAD MODEL TỪ GITHUB RELEASES
# ---------------------------------------------------------
MODEL_URL = "https://github.com/Tranthihoaithuong2009/tomato-disease-ai/releases/download/v1.0/tomato_model_best.pth"

@st.cache_resource
def load_model():
    model_path = "tomato_model_best.pth"
    if not os.path.exists(model_path):
        with st.spinner("Đang tải mô hình AI từ GitHub..."):
            res = requests.get(MODEL_URL)
            with open(model_path, "wb") as f:
                f.write(res.content)
    
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    class_names = checkpoint.get('class_names', list(DISEASE_INFO.keys()))
    num_classes = len(class_names)

    model = models.efficientnet_v2_s(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, num_classes)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, class_names

try:
    model, class_names = load_model()
except Exception as e:
    st.error(f"Lỗi khi tải mô hình: {e}")
    st.stop()

# ---------------------------------------------------------
# 5. GIAO DIỆN BÊN THANH TRÁI (SIDEBAR) - CHỌN ĐỊA PHƯƠNG
# ---------------------------------------------------------
st.sidebar.title("📌 Vị Trí Vườn Trồng")
st.sidebar.info("Ghi nhận địa chính thuộc 34 tỉnh thành & 3.321 xã/phường:")

# 1. Chọn Tỉnh / Thành phố
selected_province = st.sidebar.selectbox("1. Chọn Tỉnh/Thành phố (34 tỉnh thành):", list(PROVINCE_SUMMARY.keys()))

province_info = PROVINCE_SUMMARY[selected_province]
st.sidebar.caption(f"ℹ️ Tổng số đơn vị cấp xã: **{province_info['total']} xã/phường/đặc khu**")

# 2. Lấy danh sách xã/phường (từ JSON đầy đủ nếu có, hoặc danh sách gợi ý)
if selected_province in ADMIN_FULL_DATA:
    ward_options = ADMIN_FULL_DATA[selected_province]
else:
    ward_options = province_info["sample"]

ward_options_with_custom = ward_options + ["-- Nhập tên Xã/Phường khác --"]
selected_ward_choice = st.sidebar.selectbox("2. Chọn Xã/Phường/Đặc khu:", ward_options_with_custom)

if selected_ward_choice == "-- Nhập tên Xã/Phường khác --":
    custom_ward = st.sidebar.text_input("Nhập chính xác tên Xã/Phường/Đặc khu của bạn:")
    selected_ward = custom_ward if custom_ward.strip() != "" else "Chưa xác định"
else:
    selected_ward = selected_ward_choice

st.sidebar.markdown("---")
st.sidebar.write(f"📍 **Địa chỉ đã ghi nhận:** {selected_ward}, {selected_province}")

# ---------------------------------------------------------
# 6. GIAO DIỆN CHÍNH - CHẨN ĐOÁN VÀ KẾT QUẢ
# ---------------------------------------------------------
st.title("🌱 Chẩn Đoán Bệnh Lá Cà Chua Bằng AI")
st.write("Tải ảnh lá cà chua lên để hệ thống nhận diện tự động và đưa ra hướng điều trị.")

uploaded_file = st.file_uploader("Chọn ảnh lá cà chua (JPG, PNG, JPEG)...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Hình ảnh lá đã tải lên", use_container_width=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0)

    with st.spinner("AI đang phân tích hình ảnh..."):
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)

    predicted_raw = class_names[predicted_idx.item()]
    conf_percent = confidence.item() * 100

    info = DISEASE_INFO.get(predicted_raw, {
        "vn_name": predicted_raw,
        "symptoms": "Chưa có thông tin mô tả chi tiết.",
        "remedy": "Tham khảo ý kiến chuyên gia nông nghiệp địa phương."
    })

    with col2:
        st.subheader("📋 Kết Quả Phân Tích")
        if predicted_raw == "Healthy":
            st.success(f"**Kết quả:** {info['vn_name']}")
        else:
            st.error(f"**Kết quả:** {info['vn_name']}")
        
        st.metric(label="Độ tin cậy của AI", value=f"{conf_percent:.2f}%")
        st.write(f"📍 **Khu vực ghi nhận:** {selected_ward}, {selected_province}")

        st.markdown("---")
        st.markdown("### 🔍 Triệu Chứng Nhận Biết:")
        st.write(info["symptoms"])

        st.markdown("### 💊 Hướng Dẫn Điều Trị / Phòng Ngừa:")
        st.info(info["remedy"])
