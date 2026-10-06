import streamlit as st
import requests
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Trợ Lý AI Cà Chua", page_icon="🍅", layout="wide")

# Tải mô hình YOLOv8
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Từ điển ánh xạ tên lớp YOLO -> Tên tiếng Việt & Phác đồ điều trị
DISEASE_INFO = {
    "Tomato___Bacterial_spot": {
        "name_vi": "Bệnh đốm vi khuẩn",
        "symptoms": "Đốm nhỏ màu nâu đen trên lá, viền vàng, lá dễ bị cháy sém.",
        "treatment": "Phun các loại thuốc gốc đồng (Copper Hydroxide, Kasugamycin), tỉa bỏ lá bệnh."
    },
    "Tomato___Early_blight": {
        "name_vi": "Bệnh đốm vòng",
        "symptoms": "Đốm bệnh có các vòng đồng tâm màu nâu đen giống như bia bắn.",
        "treatment": "Phun thuốc gốc Mancozeb, Chlorothalonil, Azoxystrobin."
    },
    "Tomato___Late_blight": {
        "name_vi": "Bệnh mốc sương",
        "symptoms": "Đốm mọng nước màu xám đen, mặt dưới lá có lớp mốc trắng khi độ ẩm cao.",
        "treatment": "CẤP BÁO! Phun ngay Metalaxyl, Dimethomorph, Ridomil Gold, giảm tưới nước."
    },
    "Tomato___Leaf_Mold": {
        "name_vi": "Bệnh mốc lá",
        "symptoms": "Mặt trên lá xuất hiện đốm vàng, mặt dưới lá có lớp nấm màu ô-liu/xám.",
        "treatment": "Tăng cường thông thoáng vườn, phun Copper oxychloride hoặc Difenoconazole."
    },
    "Tomato___Septoria_leaf_spot": {
        "name_vi": "Bệnh đốm lá Septoria",
        "symptoms": "Nhiều đốm nhỏ tròn màu xám viền đen đậm rải rác trên lá.",
        "treatment": "Phun Chlorothalonil, Difenoconazole, vệ sinh tàn dư thực vật."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "name_vi": "Nhện đỏ hại cà chua",
        "symptoms": "Lá bị lấm chấm vàng nhỏ, mặt dưới có tơ nhện mỏng, lá khô cháy.",
        "treatment": "Phun thuốc trị nhện đỏ chuyên dụng (Abamectin, Pyridaben), tăng độ ẩm tưới."
    },
    "Tomato___Target_Spot": {
        "name_vi": "Bệnh đốm bia",
        "symptoms": "Đốm tròn màu nâu có viền rõ ràng trên lá và thân.",
        "treatment": "Phun thuốc trừ nấm chứa Pyraclostrobin, Chlorothalonil."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "name_vi": "Bệnh xoăn vàng lá Virus",
        "symptoms": "Lá bị xoăn ngửa, nhỏ lại, rìa lá vàng nhạt, cây rụt đọt.",
        "treatment": "Phun diệt bọ phấn trắng (vật trung gian) bằng Imidacloprid, Thiamethoxam và nhổ bỏ cây bệnh."
    },
    "Tomato___Tomato_mosaic_virus": {
        "name_vi": "Bệnh khảm Virus",
        "symptoms": "Lá xuất hiện các vệt loang nổ xanh sáng - xanh đậm, lá biến dạng.",
        "treatment": "Nhổ bỏ cây bệnh, tiêu độc dụng cụ, diệt rệp muội lây truyền."
    },
    "Tomato___healthy": {
        "name_vi": "Lá khỏe mạnh",
        "symptoms": "Lá xanh tốt, không phát hiện dấu hiệu nấm hay virus.",
        "treatment": "Tiếp tục chăm sóc, bón phân cân đối và theo dõi định kỳ."
    }
}

# Cấu trúc Tỉnh/Thành & Quận/Huyện kèm tọa độ GPS
LOCATIONS = {
    "Lâm Đồng": {
        "TP. Đà Lạt": (11.9404, 108.4583),
        "TP. Bảo Lộc": (11.5461, 107.8082),
        "Huyện Đức Trọng": (11.7282, 108.3742),
        "Huyện Đơn Dương": (11.8385, 108.5367),
        "Huyện Lạc Dương": (12.0634, 108.4891),
        "Huyện Lâm Hà": (11.8329, 108.1884),
        "Huyện Di Linh": (11.5235, 108.0805)
    },
    "Quảng Bình": {
        "TP. Đồng Hới": (17.4760, 106.5982),
        "Thị xã Ba Đồn": (17.7551, 106.4258),
        "Huyện Bố Trạch": (17.5583, 106.3023),
        "Huyện Lệ Thủy": (17.2281, 106.6841),
        "Huyện Quảng Ninh": (17.3015, 106.5862),
        "Huyện Tuyên Hóa": (17.8872, 105.9961),
        "Huyện Minh Hóa": (17.7712, 105.8882)
    },
    "Bắc Giang": {
        "TP. Bắc Giang": (21.2731, 106.1946),
        "Huyện Lục Nam": (21.2825, 106.4021),
        "Huyện Lục Ngạn": (21.3654, 106.5882),
        "Huyện Hiệp Hòa": (21.3524, 105.9723),
        "Huyện Lạng Giang": (21.3782, 106.2731)
    },
    "Gia Lai": {
        "TP. Pleiku": (13.9833, 108.0000),
        "Thị xã An Khê": (13.9482, 108.6531),
        "Huyện Đăk Đoa": (13.9882, 108.1251),
        "Huyện Chư Sê": (13.6521, 108.1205)
    },
    "Hải Dương": {
        "TP. Hải Dương": (20.9382, 106.3211),
        "TP. Chí Linh": (21.1182, 106.3982),
        "Huyện Cẩm Giàng": (20.9521, 106.2102)
    },
    "Hà Nội": {
        "Quận Ba Đình": (21.0341, 105.8306),
        "Quận Cầu Giấy": (21.0362, 105.7905),
        "Huyện Gia Lâm": (21.0182, 105.9421)
    },
    "TP Hồ Chí Minh": {
        "TP. Thủ Đức": (10.8492, 106.7537),
        "Quận 1": (10.7756, 106.7004),
        "Huyện Củ Chi": (11.0062, 106.5121)
    }
}

def get_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,rain"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()["current"]
            return data["temperature_2m"], data["relative_humidity_2m"], data["rain"]
    except Exception:
        pass
    return 25.0, 85.0, 0.0

st.title("🍅 Trợ Lý AI Chẩn Đoán Bệnh & Cảnh Báo Dịch Tễ Cà Chua")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Vị trí & Tải ảnh lá")
    selected_province = st.selectbox("📍 Chọn Tỉnh / Thành phố:", list(LOCATIONS.keys()), index=0)
    districts_in_province = LOCATIONS[selected_province]
    selected_district = st.selectbox("🏡 Chọn Quận / Huyện / TP:", list(districts_in_province.keys()), index=0)
    
    lat, lon = districts_in_province[selected_district]
    st.caption(f"📍 Tọa độ GPS: {lat:.4f}, {lon:.4f}")

    uploaded_file = st.file_uploader("📸 Tải ảnh lá cà chua:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Ảnh gốc tải lên", use_container_width=True)

with col2:
    st.subheader("2. Kết quả phân tích AI & Thời tiết")
    if uploaded_file:
        temp, humidity, rain = get_weather(lat, lon)
        
        st.markdown(f"**📍 Thời tiết tại {selected_district} ({selected_province}):**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Nhiệt độ", f"{temp} °C")
        m2.metric("Độ ẩm", f"{humidity} %")
        m3.metric("Lượng mưa", f"{rain} mm")
        st.markdown("---")
        
        # Nhận diện với YOLOv8
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp.name)
            results = model.predict(tmp.name, conf=0.15)
            result = results
            
            # Lấy ảnh đã khoanh vùng từ YOLO (chuyển BGR sang RGB)
            annotated_img = result.plot()[:, :, ::-1]
            
            # Tính toán tên bệnh và độ tin cậy % thật từ YOLO
            if len(result.boxes) > 0:
                # Lấy khung hình có độ tin cậy cao nhất
                best_box = max(result.boxes, key=lambda b: float(b.conf))
                cls_id = int(best_box.cls)
                raw_conf = float(best_box.conf) * 100  # Tính % chính xác
                raw_class_name = model.names[cls_id]
            else:
                raw_class_name = "Tomato___healthy"
                raw_conf = 0.0

            os.remove(tmp.name)

        # Lấy thông tin bệnh tiếng Việt
        info = DISEASE_INFO.get(raw_class_name, {
            "name_vi": raw_class_name,
            "symptoms": "Chưa có dữ liệu.",
            "treatment": "Liên hệ cán bộ nông nghiệp địa phương."
        })
        
        disease_display_name = info["name_vi"]
        is_healthy = "healthy" in raw_class_name.lower()

        # Hiển thị ảnh kèm Tiêu đề Matplotlib giống hệt hình mẫu
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(annotated_img)
        
        # Chọn màu chữ: Xanh nếu khỏe, Đỏ nếu bị bệnh
        title_color = "green" if is_healthy else "red"
        
        if raw_conf > 0:
            title_text = f"Predicted: {disease_display_name} ({raw_conf:.1f}%)"
        else:
            title_text = f"Predicted: {disease_display_name} (Không phát hiện vệt bệnh)"
            
        ax.set_title(title_text, color=title_color, fontsize=14, fontweight="bold", pad=12)
        ax.axis("off")  # Ẩn trục tọa độ
        
        st.pyplot(fig)
        plt.close(fig)

        # Chi tiết phác đồ điều trị
        st.markdown("### Chi tiết chẩn đoán & Khuyên dùng:")
        if is_healthy:
            st.success(f" **Trạng thái:** {disease_display_name} (Độ tin cậy: {raw_conf:.1f}%)")
            st.write(f" **Hướng dẫn:** {info['treatment']}")
        else:
            st.error(f" **Phát hiện bệnh:** {disease_display_name} (Độ tin cậy: {raw_conf:.1f}%)")
            st.write(f" **Triệu chứng:** {info['symptoms']}")
            st.write(f" **Phác đồ điều trị:** {info['treatment']}")

        st.markdown("---")
        st.markdown("###  Cảnh báo nguy cơ dịch tễ:")
        if humidity > 80 and 18 <= temp <= 25:
            st.error(f"🔴 **CẤP BÁO:** Độ ẩm ({humidity}%) và Nhiệt độ ({temp}°C) rất dễ làm nấm bệnh bùng phát mạnh trong 24-48h!")
        elif humidity > 70:
            st.warning(f"🟠 **CẢNH BÁO:** Độ ẩm khá cao ({humidity}%). Cần chú ý thông thoáng vườn.")
        else:
            st.success("🟢 **THỜI TIẾT AN TOÀN:** Ít nguy cơ lây lan dịch bệnh.")
