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

# Danh sách tọa độ trung tâm 63 Tỉnh/Thành Việt Nam
PROVINCES = {
    "An Giang": (10.5361, 105.1013), "Bà Rịa - Vũng Tàu": (10.5417, 107.2429),
    "Bắc Giang": (21.2731, 106.1946), "Bắc Kạn": (22.147, 105.8348),
    "Bạc Liêu": (9.294, 105.7244), "Bắc Ninh": (21.1861, 106.0763),
    "Bến Tre": (10.2434, 106.3751), "Bình Định": (13.782, 109.2194),
    "Bình Dương": (11.1622, 106.6489), "Bình Phước": (11.751, 106.9184),
    "Bình Thuận": (11.0903, 108.0718), "Cà Mau": (9.1769, 105.15),
    "Cần Thơ": (10.0452, 105.7469), "Cao Bằng": (22.6657, 105.9722),
    "Đà Nẵng": (16.0544, 108.2022), "Đắk Lắk": (12.6667, 108.05),
    "Đắk Nông": (12.0042, 107.6875), "Điện Biên": (21.3857, 103.0189),
    "Đồng Nai": (11.0503, 107.037), "Đồng Tháp": (10.4938, 105.6882),
    "Gia Lai": (13.9833, 108.0), "Hà Giang": (22.8233, 104.9839),
    "Hà Nam": (20.5835, 105.9229), "Hà Nội": (21.0285, 105.8542),
    "Hà Tĩnh": (18.3428, 105.9057), "Hải Dương": (20.9382, 106.3211),
    "Hải Phòng": (20.8449, 106.6881), "Hậu Giang": (9.7839, 105.4701),
    "Hòa Bình": (20.8172, 105.3376), "Hưng Yên": (20.6464, 106.0511),
    "Khánh Hòa": (12.2388, 109.1967), "Kiên Giang": (10.0125, 105.0809),
    "Kon Tum": (14.3503, 108.0002), "Lai Châu": (22.3964, 103.4582),
    "Lâm Đồng": (11.9404, 108.4583), "Lạng Sơn": (21.8537, 106.7615),
    "Lào Cai": (22.4809, 103.978), "Long An": (10.5333, 106.4082),
    "Nam Định": (20.4326, 106.1772), "Nghệ An": (19.2342, 104.8387),
    "Ninh Bình": (20.2539, 105.975), "Ninh Thuận": (11.567, 108.9897),
    "Phú Thọ": (21.3227, 105.228), "Phú Yên": (13.0882, 109.3087),
    "Quảng Bình": (17.476, 106.5982), "Quảng Nam": (15.5802, 108.2096),
    "Quảng Ngãi": (15.1205, 108.7924), "Quảng Ninh": (21.0069, 107.2925),
    "Quảng Trị": (16.7431, 107.1855), "Sóc Trăng": (9.6033, 105.98),
    "Sơn La": (21.3258, 103.9188), "Tây Ninh": (11.31, 106.0983),
    "Thái Bình": (20.4463, 106.3366), "Thái Nguyên": (21.5928, 105.8442),
    "Thanh Hóa": (19.8067, 105.7851), "Thừa Thiên Huế": (16.4674, 107.5905),
    "Tiền Giang": (10.4283, 106.3422), "TP Hồ Chí Minh": (10.8231, 106.6297),
    "Trà Vinh": (9.9348, 106.3458), "Tuyên Quang": (21.8231, 105.2158),
    "Vĩnh Long": (10.2537, 105.9722), "Vĩnh Phúc": (21.3089, 105.6049),
    "Yên Bái": (21.705, 104.8814)
}

# Hàm tìm tọa độ địa danh chi tiết (Phường/Xã/Huyện)
def search_location(query):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={query}&count=1&language=vi&format=json"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200 and "results" in res.json():
            result = res.json()["results"]
            return result["latitude"], result["longitude"], result.get("name", query)
    except:
        pass
    return None, None, None

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

st.title("🍅 Trợ Lý AI Chẩn Đoán Bệnh & Cảnh Báo Dịch Tễ Cà Chua")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Thông tin vị trí & Ảnh lá")
    
    selected_province = st.selectbox("Chọn Tỉnh / Thành phố:", list(PROVINCES.keys()), index=list(PROVINCES.keys()).index("Lâm Đồng"))
    detailed_loc = st.text_input("Nhập Phường/Xã/Huyện cụ thể (không bắt buộc):", placeholder="Ví dụ: Phường 10 Đà Lạt, Xã Hiệp Thạnh...")
    
    lat, lon = PROVINCES[selected_province]
    loc_display = selected_province
    
    if detailed_loc.strip():
        search_query = f"{detailed_loc}, {selected_province}, Vietnam"
        searched_lat, searched_lon, found_name = search_location(search_query)
        if searched_lat and searched_lon:
            lat, lon = searched_lat, searched_lon
            loc_display = f"{detailed_loc} ({selected_province})"
            st.caption(f"📍 Đã định vị chính xác tọa độ GPS: {lat:.4f}, {lon:.4f}")
        else:
            st.caption("⚠️ Không tìm thấy tọa độ chi tiết, sử dụng tọa độ trung tâm tỉnh.")

    uploaded_file = st.file_uploader("Tải ảnh lá cà chua:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Ảnh gốc tải lên", use_container_width=True)

with col2:
    st.subheader("2. Kết quả phân tích AI & Dự báo thời tiết")
    if uploaded_file:
        temp, humidity, rain = get_weather(lat, lon)
        
        st.markdown(f"**📍 Thời tiết thời gian thực tại {loc_display}:**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Nhiệt độ", f"{temp} °C")
        m2.metric("Độ ẩm", f"{humidity} %")
        m3.metric("Lượng mưa", f"{rain} mm")
        st.markdown("---")
        
        # Nhận diện với YOLOv8
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp.name)
            results = model.predict(tmp.name, conf=0.25) 
            annotated\_img = results[0].plot()[:, :, ::-1]
            st.image(annotated_img, caption="Kết quả nhận diện YOLOv8", use_container_width=True)
            os.remove(tmp.name)
            
        # Đánh giá nguy cơ dịch tễ
        st.markdown("**⚡ Đánh giá nguy cơ bùng phát dịch bệnh:**")
        if humidity > 80 and 18 <= temp <= 25:
            st.error("🔴 **CẤP BÁO:** Độ ẩm cao kết hợp nhiệt độ thuận lợi! Nguy cơ bùng phát bệnh mốc sương/đốm lá diện rộng trong 48h.")
            st.write("👉 **Hành động:** Giảm tưới nước, phun thuốc phòng trừ diện rộng ngay lập tức.")
        elif humidity > 70:
            st.warning("🟠 **CẢNH BÁO:** Nguy cơ trung bình. Cần theo dõi sát các luống cây lân cận.")
        else:
            st.success("🟢 **AN TOÀN:** Điều kiện thời tiết ít nguy cơ lây lan diện rộng.")
