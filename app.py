import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
import io
import os

# ---------------------------------------------------------
# 1. CAU HINH TRANG STREAMLIT
# ---------------------------------------------------------
st.set_page_config(
    page_title="Hệ Thống Chẩn Đoán Bệnh Của Cây Cà Chua",
    page_icon="🌱",
    layout="wide"
)

# ---------------------------------------------------------
# 2. DANH SACH 34 TINH THANH VIET NAM MOI (SAU SAP NHAP)
# ---------------------------------------------------------
ADMIN_DATA = {
    "An Giang": {"Thành phố Long Xuyên": ["Phường Mỹ Bình", "Phường Mỹ Long", "Xã Mỹ Khánh"]},
    "Bà Rịa - Vũng Tàu": {"Thành phố Vũng Tàu": ["Phường 1", "Phường 2", "Phường Thắng Tam"]},
    "Bắc Giang": {"Thành phố Bắc Giang": ["Phường Dĩnh Kế", "Phường Xương Giang", "Xã Song Mai"]},
    "Bắc Kạn": {"Thành phố Bắc Kạn": ["Phường Nguyễn Thị Minh Khai", "Phường Phùng Chí Kiên"]},
    "Bạc Liêu": {"Thành phố Bạc Liêu": ["Phường 1", "Phường 3", "Xã Hiệp Thành"]},
    "Bắc Ninh": {"Thành phố Bắc Ninh": ["Phường Suối Hoa", "Phường Tiền An", "Phường Vũ Ninh"]},
    "Bến Tre": {"Thành phố Bến Tre": ["Phường An Hội", "Phường Phú Khương", "Xã Bình Phú"]},
    "Bình Định": {"Thành phố Quy Nhơn": ["Phường Quy Nhơn", "Phường Nhơn Phú", "Xã Nhơn Lý"]},
    "Bình Dương": {"Thành phố Thủ Dầu Một": ["Phường Hiệp Thành", "Phường Phú Cường"]},
    "Bình Phước": {"Thành phố Đồng Xoài": ["Phường Tân Phú", "Phường Tiến Thành"]},
    "Bình Thuận": {"Thành phố Phan Thiết": ["Phường Mũi Né", "Phường Hàm Tiến", "Xã Tiến Thành"]},
    "Cà Mau": {"Thành phố Cà Mau": ["Phường 1", "Phường 5", "Xã Tân Thành"]},
    "Cần Thơ": {"Quận Ninh Kiều": ["Phường Tân An", "Phường An Khánh", "Phường Xuân Khánh"]},
    "Cao Bằng": {"Thành phố Cao Bằng": ["Phường Hợp Giang", "Phường Sông Bằng"]},
    "Đà Nẵng": {"Quận Hải Châu": ["Phường Hải Châu 1", "Phường Thuận Phước", "Phường Hòa Cường"]},
    "Đắk Lắk": {"Thành phố Buôn Ma Thuột": ["Phường Tân An", "Phường Thắng Lợi", "Xã Cư Ebur"]},
    "Đắk Nông": {"Thành phố Gia Nghĩa": ["Phường Nghĩa Đức", "Phường Nghĩa Thành"]},
    "Điện Biên": {"Thành phố Điện Biên Phủ": ["Phường Mường Thanh", "Phường Nam Thanh"]},
    "Đồng Nai": {"Thành phố Biên Hòa": ["Phường Trảng Dài", "Phường Tân Phong", "Xã Long Hưng"]},
    "Đồng Tháp": {"Thành phố Cao Lãnh": ["Phường 1", "Phường 2", "Xã Mỹ Tân"]},
    "Gia Lai": {"Thành phố Pleiku": ["Phường Diên Hồng", "Phường Tây Sơn", "Xã Biển Hồ"]},
    "Hà Giang": {"Thành phố Hà Giang": ["Phường Trần Phú", "Phường Nguyễn Trãi"]},
    "Hà Nam": {"Thành phố Phủ Lý": ["Phường Minh Khai", "Phường Quang Trung"]},
    "Hà Nội": {"Quận Ba Đình": ["Phường Đội Cấn", "Phường Kim Mã", "Phường Liễu Giai"]},
    "Hà Tĩnh": {"Thành phố Hà Tĩnh": ["Phường Bắc Hà", "Phường Nam Hà", "Xã Thạch Trung"]},
    "Hải Dương": {"Thành phố Hải Dương": ["Phường Trần Phú", "Phường Lê Thanh Nghị"]},
    "Hải Phòng": {"Quận Hồng Bàng": ["Phường Minh Khai", "Phường Hoàng Văn Thụ"]},
    "Hậu Giang": {"Thành phố Vị Thanh": ["Phường 1", "Phường 3", "Xã Vị Tân"]},
    "Hòa Bình": {"Thành phố Hòa Bình": ["Phường Phương Lâm", "Phường Tân Thịnh"]},
    "Hưng Yên": {"Thành phố Hưng Yên": ["Phường Hiến Nam", "Phường Lam Sơn"]},
    "Khánh Hòa": {"Thành phố Nha Trang": ["Phường Lộc Thọ", "Phường Vĩnh Nguyên", "Xã Phước Đồng"]},
    "Kiên Giang": {"Thành phố Rạch Giá": ["Phường Vĩnh Thanh", "Phường Rạch Sỏi"]},
    "Thành phố Hồ Chí Minh": {"Quận 1": ["Phường Bến Nghé", "Phường Bến Thành", "Phường Phạm Ngũ Lão"]},
    "Lâm Đồng": {"Thành phố Đà Lạt": ["Phường 1", "Phường 2", "Phường 10", "Xã Tà Nung"]}
}

# ---------------------------------------------------------
# 3. BANG DICH VA HUONG DAN DIEU TRI 11 LOP BENH
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
        "symptoms": "Mặt trên lá có đốm vàng nhạt, mặt dưới xuất hiện lớp mốc màu xám xịn hoặc nâu nhạt.",
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
        "remedy": "Phun thuốc đặc trị nhện như Abamectin, Hexythiazox hoặc Propargite. Tăng độ ẩm ẩm độ vườn."
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
# 5. GIAO DIEN BÊN THANH TRÁI (SIDEBAR) - CHỌN ĐỊA PHƯƠNG
# ---------------------------------------------------------
st.sidebar.title(" Vị Trí Vườn Trồng")
st.sidebar.info("Chọn địa chính để ghi nhận nhật ký dịch bệnh:")

selected_province = st.sidebar.selectbox("1. Tỉnh/Thành phố (34 tỉnh thành mới):", list(ADMIN_DATA.keys()))

districts = list(ADMIN_DATA[selected_province].keys())
selected_district = st.sidebar.selectbox("2. Quận/Huyện/Thị xã:", districts)

wards = ADMIN_DATA[selected_province][selected_district]
selected_ward = st.sidebar.selectbox("3. Phường/Xã/Thị trấn:", wards)

st.sidebar.markdown("---")
st.sidebar.write(f"📍 **Địa chỉ đã chọn:** {selected_ward}, {selected_district}, {selected_province}")

# ---------------------------------------------------------
# 6. GIAO DIEN CHINH - CHẨN ĐOÁN VÀ KẾT QUẢ
# ---------------------------------------------------------
st.title("🌱 Chẩn Đoán Bệnh Lá Cà Chua Bằng AI")
st.write("Tải ảnh lá cà chua lên để hệ thống nhận diện tự động và đưa ra hướng điều trị.")

uploaded_file = st.file_uploader("Chọn ảnh lá cà chua (JPG, PNG, JPEG)...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns()

    with col1:
        st.image(image, caption="Hình ảnh lá đã tải lên", use_container_width=True)

    # Tiền xử lý ảnh
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

    # Khớp kết quả với từ điển
    info = DISEASE_INFO.get(predicted_raw, {
        "vn_name": predicted_raw,
        "symptoms": "Chưa có thông tin mô tả chi tiết.",
        "remedy": "Tham khảo ý kiến chuyên gia nông nghiệp địa phương."
    })

    with col2:
        st.subheader("📋 Kết Quả Phân Tích")
        if predicted_raw == "Healthy":
            st.success(f"**{info['vn_name']}**")
        else:
            st.error(f"**{info['vn_name']}**")
        
        st.metric(label="Độ tin cậy của AI", value=f"{conf_percent:.2f}%")
        st.write(f"📍 **Khu vực ghi nhận:** {selected_ward}, {selected_district}, {selected_province}")

        st.markdown("---")
        st.markdown("### 🔍 Triệu Chứng Nhận Biết:")
        st.write(info["symptoms"])

        st.markdown("### 💊 Hướng Dẫn Điều Trị / Phòng Ngừa:")
        st.info(info["remedy"])
