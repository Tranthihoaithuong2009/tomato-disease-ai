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

# Cấu trúc dữ liệu Tỉnh/Thành phố và các Quận/Huyện/Thành phố trực thuộc (kèm tọa độ GPS)
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

# Hàm lấy dữ liệu thời tiết thực tế từ Open-Meteo
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
    
    # 1. Chọn Tỉnh / Thành phố
    selected_province = st.selectbox(" Chọn Tỉnh / Thành phố:", list(LOCATIONS.keys()), index=0)
    
    # 2. Tự động cập nhật danh sách Quận / Huyện dựa trên Tỉnh đã chọn
    districts_in_province = LOCATIONS[selected_province]
    selected_district = st.selectbox(" Chọn Quận / Huyện / TP thuộc tỉnh:", list(districts_in_province.keys()), index=0)
    
    # Lấy tọa độ GPS chính xác của Quận/Huyện được chọn
    lat, lon = districts_in_province[selected_district]
    st.caption(f"Tọa độ GPS đã chọn: {lat:.4f}, {lon:.4f}")

    uploaded_file = st.file_uploader("📸 Tải ảnh lá cà chua:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Ảnh gốc tải lên", use_container_width=True)

with col2:
    st.subheader("2. Kết quả phân tích AI & Dự báo thời tiết")
    if uploaded_file:
        temp, humidity, rain = get_weather(lat, lon)
        
        st.markdown(f"** Thời tiết thực tế tại {selected_district} ({selected_province}):**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Nhiệt độ", f"{temp} °C")
        m2.metric("Độ ẩm", f"{humidity} %")
        m3.metric("Lượng mưa", f"{rain} mm")
        st.markdown("---")
        
        # Nhận diện với YOLOv8
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp.name)
            results = model.predict(tmp.name, conf=0.25)
            # Lấy kết quả ảnh đầu tiên và chuyển hệ màu BGR -> RGB
            annotated_img = results.plot()[:, :, ::-1]
            st.image(annotated_img, caption="Kết quả nhận diện YOLOv8", use_container_width=True)
            os.remove(tmp.name)
            
        # Đánh giá nguy cơ dịch tễ
        st.markdown(" Đánh giá nguy cơ bùng phát dịch bệnh:")
        if humidity > 80 and 18 <= temp <= 25:
            st.error("🔴 **CẤP BÁO:** Độ ẩm cao kết hợp nhiệt độ thuận lợi! Nguy cơ bùng phát bệnh mốc sương/đốm lá diện rộng trong 48h.")
            st.write("**Hành động:** Giảm tưới nước, phun thuốc phòng trừ diện rộng ngay lập tức.")
        elif humidity > 70:
            st.warning("🟠 **CẢNH BÁO:** Nguy cơ trung bình. Cần theo dõi sát các luống cây lân cận.")
        else:
            st.success("🟢 **AN TOÀN:** Điều kiện thời tiết ít nguy cơ lây lan diện rộng.")
