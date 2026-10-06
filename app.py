import streamlit as st
import requests
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

st.set_page_config(page_title="Trợ Lý AI Cà Chua", page_icon="🍅", layout="wide")

# Tải mô hình YOLOv8
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Từ điển thông tin chi tiết các bệnh cà chua (Tên Việt, triệu chứng, thuốc trị)
DISEASE_INFO = {
    "Tomato___Bacterial_spot": {
        "name_vi": "Bệnh đốm vi khuẩn (Bacterial Spot)",
        "symptoms": "Đốm nhỏ màu nâu đen trên lá, viền vàng, lá dễ bị cháy sém.",
        "treatment": "Phun các loại thuốc gốc đồng (Copper Hydroxide, Kasugamycin), tỉa bỏ lá bệnh."
    },
    "Tomato___Early_blight": {
        "name_vi": "Bệnh đốm vòng (Early Blight)",
        "symptoms": "Đốm bệnh có các vòng đồng tâm màu nâu đen giống như bia bắn.",
        "treatment": "Phun thuốc gốc Mancozeb, Chlorothalonil, Azoxystrobin."
    },
    "Tomato___Late_blight": {
        "name_vi": "Bệnh mốc sương / Sương mai (Late Blight)",
        "symptoms": "Đốm mọng nước màu xám đen, mặt dưới lá có lớp mốc trắng khi độ ẩm cao.",
        "treatment": "CẤP BÁO! Phun ngay Metalaxyl, Dimethomorph, Ridomil Gold, giảm tưới nước."
    },
    "Tomato___Leaf_Mold": {
        "name_vi": "Bệnh mốc lá (Leaf Mold)",
        "symptoms": "Mặt trên lá xuất hiện đốm vàng, mặt dưới lá có lớp nấm màu ô-liu/xám.",
        "treatment": "Tăng cường thông thoáng vườn, phun Copper oxychloride hoặc Difenoconazole."
    },
    "Tomato___Septoria_leaf_spot": {
        "name_vi": "Bệnh đốm lá Septoria",
        "symptoms": "Nhiều đốm nhỏ tròn màu xám viền đen đậm rải rác trên lá.",
        "treatment": "Phun Chlorothalonil, Difenoconazole, vệ sinh tàn dư thực vật."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "name_vi": "Bệnh nhện đỏ (Spider Mites)",
        "symptoms": "Lá bị lấm chấm vàng nhỏ, mặt dưới có tơ nhện mỏng, lá khô cháy.",
        "treatment": "Phun thuốc trị nhện đỏ chuyên dụng (Abamectin, Pyridaben), tăng độ ẩm tưới."
    },
    "Tomato___Target_Spot": {
        "name_vi": "Bệnh đốm bia (Target Spot)",
        "symptoms": "Đốm tròn màu nâu có viền rõ ràng trên lá và thân.",
        "treatment": "Phun thuốc trừ nấm chứa Pyraclostrobin, Chlorothalonil."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "name_vi": "Bệnh xoăn vàng lá do Virus (TYLCV)",
        "symptoms": "Lá bị xoăn ngửa, nhỏ lại, rìa lá vàng nhạt, cây rụt đọt không phát triển.",
        "treatment": "Không có thuốc chữa virus. Phải diệt bọ phấn trắng (vật trung gian) bằng Imidacloprid, Thiamethoxam và nhổ bỏ cây bệnh."
    },
    "Tomato___Tomato_mosaic_virus": {
        "name_vi": "Bệnh Khảm Virus (Mosaic Virus)",
        "symptoms": "Lá xuất hiện các vệt loang nổ xanh sáng - xanh đậm, lá biến dạng.",
        "treatment": "Nhổ bỏ cây bệnh, tiêu độc dụng cụ, diệt rệp muội lây truyền."
    },
    "Tomato___healthy": {
        "name_vi": "Lá khỏe mạnh (Healthy)",
        "symptoms": "Lá xanh tốt, không phát hiện dấu hiệu nấm hay virus.",
        "treatment": "Tiếp tục chăm sóc, bón phân cân đối và theo dõi định kỳ."
    }
}

# Cấu trúc dữ liệu Tỉnh/Thành và Quận/Huyện
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
        "Huyện Lạng Giang": (21.3782, 106.2731),
        "Huyện Yên Dũng": (21.2012, 106.2415),
        "Huyện Việt Yên": (21.2801, 106.1102)
    },
    "Gia Lai": {
        "TP. Pleiku": (13.9833, 108.0000),
        "Thị xã An Khê": (13.9482, 108.6531),
        "Thị xã Ayun Pa": (13.5412, 108.4421),
        "Huyện Đăk Đoa": (13.9882, 108.1251),
        "Huyện Chư Sê": (13.6521, 108.1205),
        "Huyện Ia Grai": (13.9982, 107.7812)
    },
    "Hải Dương": {
        "TP. Hải Dương": (20.9382, 106.3211),
        "TP. Chí Linh": (21.1182, 106.3982),
        "Thị xã Kinh Môn": (21.0021, 106.5201),
        "Huyện Cẩm Giàng": (20.9521, 106.2102),
        "Huyện Nam Sách": (21.0012, 106.3382),
        "Huyện Gia Lộc": (20.8712, 106.3012)
    },
    "Hà Nội": {
        "Quận Ba Đình": (21.0341, 105.8306),
        "Quận Hoàn Kiếm": (21.0285, 105.8542),
        "Quận Cầu Giấy": (21.0362, 105.7905),
        "Huyện Gia Lâm": (21.0182, 105.9421),
        "Huyện Đông Anh": (21.1382, 105.8421),
        "Thị xã Sơn Tây": (21.1362, 105.5021)
    },
    "TP Hồ Chí Minh": {
        "TP. Thủ Đức": (10.8492, 106.7537),
        "Quận 1": (10.7756, 106.7004),
        "Huyện Củ Chi": (11.0062, 106.5121),
        "Huyện Hóc Môn": (10.8851, 106.5912),
        "Huyện Bình Chánh": (10.6862, 106.5782)
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
    st.subheader("1. Chọn vị trí & Tải ảnh lá")
    selected_province = st.selectbox("Chọn Tỉnh / Thành phố:", list(LOCATIONS.keys()), index=0)
    districts_in_province = LOCATIONS[selected_province]
    selected_district = st.selectbox("Chọn Quận / Huyện / TP thuộc tỉnh:", list(districts_in_province.keys()), index=0)
    
    lat, lon = districts_in_province[selected_district]
    st.caption(f"Tọa độ GPS: {lat:.4f}, {lon:.4f}")

    uploaded_file = st.file_uploader("📸 Tải ảnh lá cà chua:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Ảnh gốc tải lên", use_container_width=True)

with col2:
    st.subheader("2. Kết quả phân tích AI & Dự báo thời tiết")
    if uploaded_file:
        temp, humidity, rain = get_weather(lat, lon)
        
        st.markdown(f"**Thời tiết thực tế tại {selected_district} ({selected_province}):**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Nhiệt độ", f"{temp} °C")
        m2.metric("Độ ẩm", f"{humidity} %")
        m3.metric("Lượng mưa", f"{rain} mm")
        st.markdown("---")
        
        # Nhận diện với YOLOv8
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp.name)
            results = model.predict(tmp.name, conf=0.25)
            annotated_img = results.plot()[:, :, ::-1]
            st.image(annotated_img, caption="Ảnh khoanh vùng nhận diện YOLOv8", use_container_width=True)
            
            # Bóc tách danh sách các bệnh phát hiện được
            boxes = results.boxes
            detected_list = []
            
            if len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls)
                    conf = float(box.conf) * 100
                    class_name = model.names[cls_id]
                    detected_list.append((class_name, conf))
            
            os.remove(tmp.name)
            
        st.markdown("### 🔍 Kết quả chẩn đoán chi tiết từ AI:")
        if detected_list:
            for cls_name, conf in detected_list:
                info = DISEASE_INFO.get(cls_name, {
                    "name_vi": cls_name,
                    "symptoms": "Chưa có thông tin chi tiết.",
                    "treatment": "Cần tham khảo thêm ý kiến chuyên gia nông nghiệp."
                })
                
                if "healthy" in cls_name.lower():
                    st.success(f"**{info['name_vi']}** (Độ tin cậy: {conf:.1f}%)")
                    st.write(f"Trạng thái: {info['symptoms']}")
                    st.write(f"Khuyên dùng: {info['treatment']}")
                else:
                    st.error(f"Phát hiện: {info['name_vi']} (Độ tin cậy: {conf:.1f}%)")
                    st.write(f"Triệu chứng: {info['symptoms']}")
                    st.write(f"Phác đồ điều trị: {info['treatment']}")
        else:
            st.info("AI không phát hiện vệt bệnh nào rõ ràng (có thể lá bình thường hoặc độ tin cậy < 25%).")

        st.markdown("---")
        st.markdown("###  Cảnh báo nguy cơ lây lan theo thời tiết:")
        if humidity > 80 and 18 <= temp <= 25:
            st.error(f"🔴 **CẤP BÁO THỜI TIẾT:** Độ ẩm ({humidity}%) và Nhiệt độ ({temp}°C) cực kỳ thuận lợi cho nấm bệnh lây lan mạnh trong 24-48h tới!")
        elif humidity > 70:
            st.warning(f"🟠 **CẢNH BÁO THỜI TIẾT:** Độ ẩm khá cao ({humidity}%). Cần chú ý thông thoáng vườn.")
        else:
            st.success("🟢 **THỜI TIẾT AN TOÀN:** Điều kiện môi trường hiện tại ít nguy cơ bùng phát dịch.")
