import os
import requests
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Cấu hình

st.set_page_config(
    page_title="Chẩn Đoán Bệnh Lá Cà Chua AI",
    page_icon="icon.png",
    layout="centered"
)

# 2. Bảng tọa độ địa lí của các tỉnh thành ở Việt Nam

PROVINCE_COORDS = {
    "TP Hà Nội": {"lat": 21.0285, "lon": 105.8542},
    "TP Hồ Chí Minh": {"lat": 10.8231, "lon": 106.6297},
    "TP Đà Nẵng": {"lat": 16.0544, "lon": 108.2022},
    "TP Hải Phòng": {"lat": 20.8449, "lon": 106.6881},
    "TP Cần Thơ": {"lat": 10.0452, "lon": 105.7469},
    "TP Huế": {"lat": 16.4637, "lon": 107.5909},
    "Tỉnh An Giang": {"lat": 10.5365, "lon": 105.1259},
    "Tỉnh Bắc Ninh": {"lat": 21.1861, "lon": 106.0763},
    "Tỉnh Cao Bằng": {"lat": 22.6658, "lon": 105.9036},
    "Tỉnh Cà Mau": {"lat": 9.1769, "lon": 105.1524},
    "Tỉnh Đắk Lắk": {"lat": 12.6667, "lon": 108.0500},
    "Tỉnh Điện Biên": {"lat": 21.3853, "lon": 103.0188},
    "Tỉnh Đồng Nai": {"lat": 10.9574, "lon": 106.8427},
    "Tỉnh Đồng Tháp": {"lat": 10.4938, "lon": 105.6882},
    "Tỉnh Gia Lai": {"lat": 13.9833, "lon": 108.0000},
    "Tỉnh Hà Tĩnh": {"lat": 18.3430, "lon": 105.9058},
    "Tỉnh Hưng Yên": {"lat": 20.6464, "lon": 106.0511},
    "Tỉnh Khánh Hòa": {"lat": 12.2388, "lon": 109.1967},
    "Tỉnh Lai Châu": {"lat": 22.3964, "lon": 103.4583},
    "Tỉnh Lâm Đồng": {"lat": 11.9404, "lon": 108.4583},
    "Tỉnh Lạng Sơn": {"lat": 21.8478, "lon": 106.7583},
    "Tỉnh Lào Cai": {"lat": 22.4856, "lon": 103.9707},
    "Tỉnh Nghệ An": {"lat": 19.2342, "lon": 104.8387},
    "Tỉnh Ninh Bình": {"lat": 20.2506, "lon": 105.9745},
    "Tỉnh Phú Thọ": {"lat": 21.3167, "lon": 105.2167},
    "Tỉnh Quảng Ngãi": {"lat": 15.1205, "lon": 108.7922},
    "Tỉnh Quảng Ninh": {"lat": 21.0069, "lon": 107.2925},
    "Tỉnh Quảng Trị": {"lat": 16.7431, "lon": 107.1861},
    "Tỉnh Sơn La": {"lat": 21.3256, "lon": 103.9186},
    "Tỉnh Tây Ninh": {"lat": 11.3122, "lon": 106.0983},
    "Tỉnh Thái Nguyên": {"lat": 21.5928, "lon": 105.8442},
    "Tỉnh Thanh Hóa": {"lat": 19.8067, "lon": 105.7850},
    "Tỉnh Tuyên Quang": {"lat": 21.8239, "lon": 105.2173},
    "Tỉnh Vĩnh Long": {"lat": 10.2537, "lon": 105.9722}
}


# 3. CSDL của 11 loại bệnh và học các loại bệnh

DISEASE_DATABASE = {
    "Bacterial_spot": {
        "vn_name": "Bệnh Đốm Vi Khuẩn (Bacterial Spot)",
        "type": "Vi khuẩn",
        "symptoms": "Xuất hiện các đốm nhỏ màu nâu đen, mọng nước trên bề mặt lá. Lá vàng và rụng sớm dưới gốc.",
        "remedy": "Phun thuốc gốc đồng (Copper Hydroxide hoặc Kasugamycin). Tỉa bớt lá già, tạo độ thông thoáng và tránh tưới phun mưa trực tiếp lên lá."
    },
    "Early_blight": {
        "vn_name": "Bệnh Đốm Vòng / Cháy Lá Sớm (Early Blight)",
        "type": "Nấm",
        "symptoms": "Vết bệnh hình tròn có các vòng đồng tâm màu nâu đen, viền lá xung quanh đốm bị vàng khè.",
        "remedy": "Phun thuốc gốc Mancozeb, Chlorothalonil hoặc Difenoconazole. Thu gom lá bệnh rụng và luân canh cây trồng."
    },
    "Late_blight": {
        "vn_name": "Bệnh Sương Mai / Cháy Lá Muộn (Late Blight)",
        "type": "Nấm",
        "symptoms": "Vết đốm màu xám xanh mọng nước, phát triển rất nhanh làm cháy khô toàn bộ lá, cành và thân cây.",
        "remedy": "Phun ngay thuốc đặc trị chứa Metalaxyl, Dimethomorph hoặc Ridomil Gold. Ngừng tưới nước lên lá và cách ly cây bệnh."
    },
    "Leaf_mold": {
        "vn_name": "Bệnh Mốc Lá (Leaf Mold)",
        "type": "Nấm",
        "symptoms": "Mặt trên lá có đốm vàng nhạt, mặt dưới xuất hiện lớp mốc màu xám xỉn hoặc nâu nhạt.",
        "remedy": "Phun thuốc gốc Đồng hoặc Carbendazim. Giảm độ ẩm vườn, tăng cường thông gió và tỉa lá gốc."
    },
    "Septoria_leaf_spot": {
        "vn_name": "Bệnh Đốm Lá Septoria (Septoria Leaf Spot)",
        "type": "Nấm",
        "symptoms": "Các đốm tròn nhỏ màu xám nhạt ở giữa, viền nâu đen xuất hiện nhiều ở các lá tán gốc.",
        "remedy": "Phun Azoxystrobin hoặc Difenoconazole. Thu gom tiêu hủy lá rụng và dọn sạch cỏ dại xung quanh."
    },
    "Spider_mites": {
        "vn_name": "Nhện Đỏ Hại Lá (Two-Spotted Spider Mites)",
        "type": "Côn trùng hại",
        "symptoms": "Lá xuất hiện các chấm nhỏ lốm đốm màu vàng nhạt, mặt dưới lá có màng tơ mỏng và nhện nhỏ bò.",
        "remedy": "Phun thuốc đặc trị nhện như Abamectin, Hexythiazox hoặc Propargite. Tăng độ ẩm phun sương để hạn chế nhện phát triển."
    },
    "Target_Spot": {
        "vn_name": "Bệnh Đốm Mục Tiêu (Target Spot)",
        "type": "Nấm",
        "symptoms": "Vết bệnh hình tròn màu nâu với tâm sáng hơn, tạo hình dạng giống như bia bắn.",
        "remedy": "Sử dụng thuốc trừ nấm chứa Chlorothalonil, Pyraclostrobin hoặc Mancozeb."
    },
    "Yellow_Leaf_Curl_Virus": {
        "vn_name": "Bệnh Xoăn Lá Vàng Do Virus (Yellow Leaf Curl Virus)",
        "type": "Virus",
        "symptoms": "Lá bị xoăn ngửa lên trên, phiến lá nhỏ lại, màu vàng chanh, cây lùn còi cọc và không thể ra quả.",
        "remedy": "Không thể chữa khỏi bằng thuốc hóa học. Cần phun Imidacloprid để tiêu diệt bọ phấn trắng (vật trung gian truyền bệnh) và nhổ bỏ cây bệnh."
    },
    "Mosaic_virus": {
        "vn_name": "Bệnh Khảm Lá Do Virus (Mosaic Virus)",
        "type": "Virus",
        "symptoms": "Lá loang lổ các vệt màu xanh đậm và xanh nhạt xen kẽ, lá bị biến dạng, nhăn nheo.",
        "remedy": "Nhổ bỏ cây bệnh để tránh lây lan. Khử trùng dụng cụ cắt tỉa và diệt rệp muỗi truyền bệnh."
    },
    "Powdery_mildew": {
        "vn_name": "Bệnh Phấn Trắng (Powdery Mildew)",
        "type": "Nấm",
        "symptoms": "Lớp bột trắng như phấn bao phủ trên bề mặt lá, làm lá khô xơ, chuyển vàng và rụng.",
        "remedy": "Phun Sulfur (Lưu huỳnh), Hexaconazole hoặc Myclobutanil khi vừa phát hiện vết phấn đầu tiên."
    },
    "Healthy": {
        "vn_name": "Lá Khỏe Mạnh (Healthy)",
        "type": "Không có bệnh",
        "symptoms": "Lá xanh tươi, phiến lá phẳng, không có dấu hiệu bị nấm, vi khuẩn hay sâu bệnh tấn công.",
        "remedy": "Tiếp tục duy trì chế độ chăm sóc, tưới nước vừa đủ và bón phân định kỳ cân đối N-P-K."
    }
}

NORMALIZED_DB = {k.lower().replace("_", "").replace(" ", "").replace("-", ""): v for k, v in DISEASE_DATABASE.items()}


# 4. Hàm lấy thời tiết theo tọa độ địa lí trên api

def get_realtime_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,rain"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            current_data = res.json().get("current", {})
            temp = current_data.get("temperature_2m", 28.0)
            humidity = current_data.get("relative_humidity_2m", 78)
            rain = current_data.get("rain", 0.0)
            return temp, humidity, rain
    except Exception:
        pass
    return 28.5, 80.0, 0.0

# 5. Hàm đánh giá khả năng lây lan của bệnh trong tương lai:

def evaluate_spread_forecast(pred_key, temp, humidity, rain):
    info = NORMALIZED_DB.get(pred_key, {
        "vn_name": "Không xác định",
        "type": "Không rõ"
    })
    
    disease_type = info["type"]

    if pred_key == "healthy":
        return "KHÔNG CÓ (0%)", "Cây trồng đang khỏe mạnh. Hãy tiếp tục duy trì vệ sinh vườn và theo dõi định kỳ."

    if disease_type == "Virus":
        return (
            "RẤT CAO (Tốc độ bùng phát nhanh)",
            "Bệnh do Virus lây truyền qua bọ phấn trắng và rệp muỗi. Nếu không phun thuốc diệt côn trùng trung gian và nhổ bỏ cây bệnh, mầm bệnh sẽ lan ra toàn bộ vườn trong vòng 3-5 ngày."
        )

    if disease_type == "Nấm":
        if humidity >= 75 or rain > 0:
            return (
                "CỰC KỲ CAO (Nguy cơ dịch bùng phát 80-95%)",
                f"Độ ẩm không khí thời gian thực cao ({humidity}%) và có mưa ({rain} mm) là điều kiện lý tưởng cho bào tử nấm phát tán mạnh qua giọt nước và gió. Bệnh sẽ lây lan rất nhanh sang các cây lân cận trong 24-48 giờ tới."
            )
        else:
            return (
                "TRUNG BÌNH (30-50%)",
                f"Thời tiết hiện tại (Độ ẩm {humidity}%, Nhiệt độ {temp}°C) làm chậm sự phát tán bào tử nấm. Tuy nhiên cần phun thuốc phòng ngừa trước khi có đợt mưa mới."
            )

    if disease_type == "Vi khuẩn":
        if rain > 0 or humidity >= 80:
            return (
                "CAO (Nguy cơ lây lan 70-85%)",
                "Vi khuẩn lây lan mạnh qua nước tưới, giọt mưa bắn và dụng cụ cắt tỉa. Cần cách ly ngay vùng cây bệnh."
            )
        else:
            return (
                "TRUNG BÌNH (40%)",
                "Vi khuẩn phát triển chậm hơn khi môi trường khô ráo. Tránh tưới nước lên lá để hạn chế vi khuẩn văng sang cây khác."
            )

    if disease_type == "Côn trùng hại":
        if temp >= 30 and humidity < 70:
            return (
                "RẤT CAO (Bùng phát do thời tiết khô nóng)",
                f"Nhiệt độ cao ({temp}°C) và không khí khô thúc đẩy nhện đỏ sinh sản bùng phát rất nhanh. Cần phun thuốc đặc trị nhện và tăng độ ẩm vườn."
            )
        else:
            return (
                "TRUNG BÌNH (45%)",
                "Mật độ nhện đang ở mức gia tăng. Phun thuốc đặc trị để chặn đứng dòng sinh sản."
            )

    return "TRUNG BÌNH", "Cần theo dõi sát sao biểu hiện của vườn trong các ngày tới."

# 6. ADD MÔ HÌNH AI

MODEL_URL = "https://github.com/Tranthihoaithuong2009/tomato-disease-ai/releases/download/v1.0/tomato_model_best.pth"

def load_checkpoint_file(file_source):
    try:
        checkpoint = torch.load(file_source, map_location=torch.device('cpu'), weights_only=False)
    except TypeError:
        checkpoint = torch.load(file_source, map_location=torch.device('cpu'))

    class_names = checkpoint.get('class_names', list(DISEASE_DATABASE.keys()))
    num_classes = len(class_names)

    model = models.efficientnet_v2_s(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, num_classes)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, class_names

@st.cache_resource
def get_ai_model():
    # Tự động nạp file .pth có sẵn trong repo
    pth_files = [f for f in os.listdir(".") if f.endswith(".pth") and os.path.getsize(f) > 5000000]
    if pth_files:
        try:
            return load_checkpoint_file(pth_files)
        except Exception:
            pass

    # Nếu chưa có file local thì tự động tải từ GitHub Release ẩn đằng sau
    local_default = "tomato_model_best.pth"
    if not os.path.exists(local_default) or os.path.getsize(local_default) < 5000000:
        res = requests.get(MODEL_URL, allow_redirects=True, timeout=20)
        if res.status_code == 200 and len(res.content) > 5000000:
            with open(local_default, "wb") as f:
                f.write(res.content)

    return load_checkpoint_file(local_default)


# 7. GIAO DIỆN CHÍNH TRÊN TRANG WEB


def main():
    st.title("Hệ Thống Chẩn Đoán Bệnh Lá Cà Chua Dựa Trên Hình Ảnh, Thời Tiết Tại Vị Trí Trồng")
    st.write("Ứng dụng tự động chẩn đoán bệnh cây cà chua, tra cứu thời tiết thời gian thực và dự báo nguy cơ lây lan dịch bệnh.")
    st.markdown("---")

    # BƯỚC 1: CHỌN TỈNH/THÀNH PHỐ
    st.subheader("1. Chọn Tỉnh/Thành phố")
    selected_province = st.selectbox(
        "Vui lòng chọn địa phương của bạn:",
        options=list(PROVINCE_COORDS.keys())
    )

    # Lấy tọa độ địa lý
    coords = PROVINCE_COORDS[selected_province]
    lat, lon = coords["lat"], coords["lon"]

    st.markdown("---")

   # BƯỚC 2: TẢI ẢNH LÁ CÀ CHUA 
    st.subheader("2. Chọn ảnh lá cà chua") 
    uploaded_file = st.file_uploader( 
        "Chọn ảnh lá cà chua từ thiết bị của bạn:",
        type=["jpg", "png", "jpeg"], 
    )
    
    if uploaded_file is not None:
        st.markdown("---")
        st.subheader("3. Kết Quả Phân Tích Chi Tiết")

        image = Image.open(uploaded_file).convert('RGB')
        
        # Nạp mô hình AI ẩn phía sau (Người dùng không phải làm gì)
        try:
            with st.spinner("Hệ thống đang khởi tạo mô hình AI và phân tích ảnh..."):
                model, class_names = get_ai_model()
        except Exception as e:
            st.error(f"Không thể khởi tạo mô hình AI: {e}. Vui lòng kiểm tra file tomato_model_best.pth trên GitHub.")
            return

        # Xử lý hình ảnh
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_raw = class_names[predicted_idx.item()]
        conf_percent = confidence.item() * 100

        pred_key = predicted_raw.lower().replace("_", "").replace(" ", "").replace("-", "")
        info = NORMALIZED_DB.get(pred_key, {
            "vn_name": predicted_raw,
            "type": "Không rõ",
            "symptoms": "Chưa có thông tin mô tả chi tiết.",
            "remedy": "Tham khảo ý kiến chuyên gia nông nghiệp địa phương."
        })

        # Lấy thời tiết thời gian thực theo tọa độ
        temp, humidity, rain = get_realtime_weather(lat, lon)
        spread_risk, spread_detail = evaluate_spread_forecast(pred_key, temp, humidity, rain)

        # Hiển thị kết quả dạng 2 cột trực quan
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Hình ảnh lá đã tải lên", use_container_width=True)

        with col2:
            if pred_key == "healthy":
                st.success(f"**Kết quả:** {info['vn_name']}")
            else:
                st.error(f"**Kết quả:** {info['vn_name']}")

            st.metric(label="Độ tin cậy của AI", value=f"{conf_percent:.2f}%")
            st.write(f"**Tác nhân gây bệnh:** {info['type']}")
            st.write(f"**Vị trí địa lý:** {selected_province} (Tọa độ: {lat}°N, {lon}°E)")

        st.markdown("---")
        
        # BẢNG THÔNG TIN THỜI TIẾT THỜI GIAN THỰC
        st.subheader("Thông Số Thời Tiết Thời Gian Thực Tại Địa Phương")
        w_col1, w_col2, w_col3 = st.columns(3)
        w_col1.metric("Nhiệt độ hiện tại", f"{temp} °C")
        w_col2.metric("Độ ẩm không khí", f"{humidity} %")
        w_col3.metric("Lượng mưa", f"{rain} mm")

        st.markdown("---")

        # KHẢ NĂNG LÂY LAN TRONG TƯƠNG LAI
        st.subheader("Khả Năng Lây Lan Trong Tương Lai")
        st.warning(f"**Mức độ nguy cơ:** {spread_risk}")
        st.write(spread_detail)

        st.markdown("---")

        # TRIỆU CHỨNG & ĐỀ XUẤT GIẢI PHÁP
        st.subheader("Triệu Chứng Đặc Trưng")
        st.write(info["symptoms"])

        st.subheader("Đề Xuất Giải Pháp Khắc Phục")
        st.info(info["remedy"])

if __name__ == "__main__":
    main()
