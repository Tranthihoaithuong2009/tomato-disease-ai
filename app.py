import streamlit as st
import requests
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

st.set_page_config(page_title="Trợ Lý AI Cà Chua", page_icon="🍅", layout="wide")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

LOCATION_DATA = {
    "Đà Lạt (Lâm Đồng)": {"lat": 11.9404, "lon": 108.4583},
    "Đơn Dương (Lâm Đồng)": {"lat": 11.8333, "lon": 108.5333},
    "Bắc Giang": {"lat": 21.2731, "lon": 106.1946},
    "Hải Dương": {"lat": 20.9382, "lon": 106.3211},
    "Gia Lai": {"lat": 13.9833, "lon": 108.0000},
}

def get_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,rain"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()["current"]
            return data["temperature_2m"], data["relative_humidity_2m"], data["rain"]
    except:
        pass
    return 25.0, 85.0, 0.0

st.title("Trợ Lý AI Chẩn Đoán Bệnh & Cảnh Báo Dịch Tễ Cà Chua")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Thông tin đầu vào")
    loc = st.selectbox("Chọn vị trí nông trại:", list(LOCATION_DATA.keys()))
    uploaded_file = st.file_uploader("Tải ảnh lá cà chua:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Ảnh gốc tải lên", use_container_width=True)

with col2:
    st.subheader("2. Kết quả phân tích AI")
    if uploaded_file:
        coords = LOCATION_DATA[loc]
        temp, humidity, rain = get_weather(coords["lat"], coords["lon"])
        
        st.markdown(f"** Thời tiết thực tế tại {loc}:**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Nhiệt độ", f"{temp} °C")
        m2.metric("Độ ẩm", f"{humidity} %")
        m3.metric("Mưa", f"{rain} mm")
        st.markdown("---")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp.name)
            results = model.predict(tmp.name, conf=0.25)
            annotated_img = results.plot()
            st.image(annotated_img, caption="Kết quả nhận diện YOLOv8", use_container_width=True)
            os.remove(tmp.name)
            
        st.markdown("** Đánh giá nguy cơ bùng phát dịch bệnh:**")
        if humidity > 80 and 18 <= temp <= 25:
            st.error(" **CẤP BÁO:** Độ ẩm cao kết hợp nhiệt độ thuận lợi! Nguy cơ bùng phát bệnh mốc sương/đốm lá diện rộng trong 48h.")
            st.write(" **Hành động:** Giảm tưới nước, phun thuốc phòng trừ diện rộng ngay lập tức.")
        elif humidity > 70:
            st.warning(" **CẢNH BÁO:** Nguy cơ trung bình. Cần theo dõi sát các luống cây lân cận.")
        else:
            st.success(" **AN TOÀN:** Điều kiện thời tiết ít nguy cơ lây lan diện rộng.")
