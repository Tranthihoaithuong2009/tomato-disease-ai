import os
import urllib.request
import requests
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Trợ Lý AI Chẩn Đoán Bệnh Cà Chua", page_icon="🍅", layout="wide")

# 1. Đường link tải file trọng số từ GitHub Releases
MODEL_URL = "https://github.com/Tranthihoaithuong2009/tomato-disease-ai/releases/download/v1.0/tomato_model_best.pth"
MODEL_PATH = "tomato_model_best.pth"

# 2. Danh sách 63 Tỉnh / Thành Phố Việt Nam kèm tọa độ GPS
PROVINCES_GPS = {
    "An Giang": (10.5364, 105.1110), "Bà Rịa - Vũng Tàu": (10.5417, 107.2429), "Bắc Giang": (21.2731, 106.1946),
    "Bắc Kạn": (22.1470, 105.8348), "Bạc Liêu": (9.2941, 105.7244), "Bắc Ninh": (21.1861, 106.0763),
    "Bến Tre": (10.2432, 106.3751), "Bình Định": (13.7830, 109.2197), "Bình Dương": (11.1604, 106.6521),
    "Bình Phước": (11.7512, 106.9184), "Bình Thuận": (11.0904, 108.0722), "Cà Mau": (9.1769, 105.1524),
    "Cần Thơ": (10.0452, 105.7469), "Cao Bằng": (22.6657, 105.9182), "Đà Nẵng": (16.0544, 108.2022),
    "Đắc Lắk": (12.6667, 108.0500), "Đắk Nông": (12.0042, 107.6875), "Điện Biên": (21.3842, 103.0232),
    "Đồng Nai": (11.0500, 107.0000), "Đồng Tháp": (10.4938, 105.6413), "Gia Lai": (13.9833, 108.0000),
    "Hà Giang": (22.8233, 104.9839), "Hà Nam": (20.5839, 105.9228), "Hà Nội": (21.0285, 105.8542),
    "Hà Tĩnh": (18.3559, 105.8877), "Hải Dương": (20.9372, 106.3146), "Hải Phòng": (20.8449, 106.6881),
    "Hậu Giang": (9.7833, 105.4667), "Hòa Bình": (20.8172, 105.3378), "Hưng Yên": (20.6464, 106.0511),
    "Khánh Hòa": (12.2388, 109.1967), "Kiên Giang": (10.0125, 105.0809), "Kon Tum": (14.3500, 108.0000),
    "Lai Châu": (22.3964, 103.4581), "Lâm Đồng": (11.9404, 108.4583), "Lạng Sơn": (21.8537, 106.7611),
    "Lào Cai": (22.4856, 103.9707), "Long An": (10.5333, 106.4000), "Nam Định": (20.4200, 106.1683),
    "Nghệ An": (19.2342, 104.8920), "Ninh Bình": (20.2506, 105.9744), "Ninh Thuận": (11.5670, 108.9880),
    "Phú Thọ": (21.3228, 105.2280), "Phú Yên": (13.0882, 109.0929), "Quảng Bình": (17.4760, 106.5982),
    "Quảng Nam": (15.5667, 108.0000), "Quảng Ngãi": (15.1205, 108.7924), "Quảng Ninh": (21.0069, 107.2925),
    "Quảng Trị": (16.7444, 107.1853), "Sóc Trăng": (9.6033, 105.9800), "Sơn La": (21.3256, 103.9188),
    "Tây Ninh": (11.3100, 106.0983), "Thái Bình": (20.4464, 106.3364), "Thái Nguyên": (21.5928, 105.8442),
    "Thanh Hóa": (19.8000, 105.7667), "Thừa Thiên Huế": (16.4637, 107.5909), "Tiền Giang": (10.4200, 106.3400),
    "TP Hồ Chí Minh": (10.8231, 106.6297), "Trà Vinh": (9.9347, 106.3453), "Tuyên Quang": (21.8233, 105.2158),
    "Vĩnh Long": (10.2537, 105.9722), "Vĩnh Phúc": (21.3089, 105.6049), "Yên Bái": (21.7050, 104.8742)
}

# 3. Ánh xạ ĐẦY ĐỦ 10 LỚP BỆNH CÀ CHUA CHUẨN
DISEASE_DETAILS = {
    # 1. Đốm vi khuẩn
    "Bacterial Spot": ("Bệnh Đốm Vi Khuẩn", "Đốm nhỏ màu nâu đen trên lá, viền vàng xung quanh.", "Phun thuốc gốc đồng (Copper Hydroxide, Kasugamycin), tỉa bỏ lá bệnh."),
    "Tomato___Bacterial_spot": ("Bệnh Đốm Vi Khuẩn", "Đốm nhỏ màu nâu đen trên lá, viền vàng xung quanh.", "Phun thuốc gốc đồng (Copper Hydroxide, Kasugamycin), tỉa bỏ lá bệnh."),
    
    # 2. Đốm vòng
    "Early Blight": ("Bệnh Đốm Vòng", "Đốm lá có các vòng đồng tâm màu nâu đen.", "Phun Mancozeb, Chlorothalonil hoặc Azoxystrobin."),
    "Tomato___Early_blight": ("Bệnh Đốm Vòng", "Đốm lá có các vòng đồng tâm màu nâu đen.", "Phun Mancozeb, Chlorothalonil hoặc Azoxystrobin."),
    
    # 3. Mốc sương / Sương mai
    "Late Blight": ("Bệnh Mốc Sương (Sương Mai)", "Đốm mọng nước xám đen, mặt dưới lá có lớp mốc trắng.", "CẤP BÁO! Phun Metalaxyl, Dimethomorph hoặc Ridomil Gold."),
    "Tomato___Late_blight": ("Bệnh Mốc Sương (Sương Mai)", "Đốm mọng nước xám đen, mặt dưới lá có lớp mốc trắng.", "CẤP BÁO! Phun Metalaxyl, Dimethomorph hoặc Ridomil Gold."),
    
    # 4. Mốc lá
    "Leaf Mold": ("Bệnh Mốc Lá", "Mặt trên lá xuất hiện các đốm vàng nhạt, mặt dưới có lớp mốc xám nâu.", "Giảm độ ẩm nhà kính, tỉa lá gốc, phun Difenoconazole hoặc Copper Oxychloride."),
    "Tomato___Leaf_Mold": ("Bệnh Mốc Lá", "Mặt trên lá xuất hiện các đốm vàng nhạt, mặt dưới có lớp mốc xám nâu.", "Giảm độ ẩm nhà kính, tỉa lá gốc, phun Difenoconazole hoặc Copper Oxychloride."),
    
    # 5. Đốm lá Septoria
    "Septoria Leaf Spot": ("Bệnh Đốm Lá Septoria", "Nhiều đốm tròn nhỏ màu xám nhạt với viền đen sẫm trên lá già.", "Tỉa bỏ lá già bị nhiễm, phun Chlorothalonil hoặc Copper Fungicide."),
    "Tomato___Septoria_leaf_spot": ("Bệnh Đốm Lá Septoria", "Nhiều đốm tròn nhỏ màu xám nhạt với viền đen sẫm trên lá già.", "Tỉa bỏ lá già bị nhiễm, phun Chlorothalonil hoặc Copper Fungicide."),
    
    # 6. Nhện đỏ
    "Spider Mites": ("Bệnh Nhện Đỏ Cắn Phá", "Mặt trên lá lấm chấm đốm vàng/trắng nhỏ, có tơ mỏng ở mặt dưới.", "Phun thuốc trừ nhện (Abamectin, Fenpyroximate), tăng độ ẩm tưới rửa lá."),
    "Two-spotted spider mite": ("Bệnh Nhện Đỏ Cắn Phá", "Mặt trên lá lấm chấm đốm vàng/trắng nhỏ, có tơ mỏng ở mặt dưới.", "Phun thuốc trừ nhện (Abamectin, Fenpyroximate), tăng độ ẩm tưới rửa lá."),
    "Tomato___Spider_mites Two-spotted_spider_mite": ("Bệnh Nhện Đỏ Cắn Phá", "Mặt trên lá lấm chấm đốm vàng/trắng nhỏ, có tơ mỏng ở mặt dưới.", "Phun thuốc trừ nhện (Abamectin, Fenpyroximate), tăng độ ẩm tưới rửa lá."),
    
    # 7. Đốm mục tiêu / Đốm phấn
    "Target Spot": ("Bệnh Đốm Mục Tiêu", "Đốm nâu đốm hoại tử tròn có tâm màu xám sáng giống hình bia bắn.", "Tăng khoảng cách trồng, phun Azoxystrobin hoặc Pyraclostrobin."),
    "Tomato___Target_Spot": ("Bệnh Đốm Mục Tiêu", "Đốm nâu đốm hoại tử tròn có tâm màu xám sáng giống hình bia bắn.", "Tăng khoảng cách trồng, phun Azoxystrobin hoặc Pyraclostrobin."),
    
    # 8. Xoăn vàng lá virus
    "Yellow Leaf Curl Virus": ("Bệnh Xoăn Vàng Lá Virus", "Lá xoăn ngửa, thu nhỏ lại, rìa lá biến màu vàng.", "Phun diệt bọ phấn trắng (Imidacloprid) và nhổ bỏ triệt để cây bệnh."),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("Bệnh Xoăn Vàng Lá Virus", "Lá xoăn ngửa, thu nhỏ lại, rìa lá biến màu vàng.", "Phun diệt bọ phấn trắng (Imidacloprid) và nhổ bỏ triệt để cây bệnh."),
    
    # 9. Khảm virus
    "Mosaic Virus": ("Bệnh Khảm Virus (Tomato Mosaic)", "Lá biến dạng khảm loang nổ xanh nhạt - xanh đậm, lá nhăn nheo.", "Tiêu hủy cây bệnh, nhặt cỏ dại, vệ sinh dụng cụ tỉa lá."),
    "Tomato___Tomato_mosaic_virus": ("Bệnh Khảm Virus (Tomato Mosaic)", "Lá biến dạng khảm loang nổ xanh nhạt - xanh đậm, lá nhăn nheo.", "Tiêu hủy cây bệnh, nhặt cỏ dại, vệ sinh dụng cụ tỉa lá."),
    
    # 10. Khỏe mạnh
    "Healthy": ("Lá Khỏe Mạnh", "Lá xanh tốt, không phát hiện dấu hiệu vết bệnh.", "Tiếp tục chăm sóc, bón phân cân đối."),
    "Tomato___healthy": ("Lá Khỏe Mạnh", "Lá xanh tốt, không phát hiện dấu hiệu vết bệnh.", "Tiếp tục chăm sóc, bón phân cân đối.")
}

# 4. Nạp mô hình EfficientNetV2-S PyTorch (tự động tải từ GitHub Releases nếu chưa có)
@st.cache_resource
def load_pytorch_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Đang tải file trọng số mô hình từ GitHub Releases (~85MB)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    class_names = checkpoint['class_names']
    
    model = models.efficientnet_v2_s(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, len(class_names))
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, class_names

model, class_names = load_pytorch_model()

# 5. Pipeline xử lý ảnh đầu vào
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 6. Hàm lấy thông tin thời tiết Open-Meteo API
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

# 7. Giao diện ứng dụng Web Streamlit
st.title("🍅 Trợ Lý AI Chẩn Đoán Bệnh Cà Chua (EfficientNetV2)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Vị trí nông trại & Tải ảnh lá")
    
    selected_province = st.selectbox("Chọn Tỉnh / Thành phố:", list(PROVINCES_GPS.keys()), index=34) # Mặc định Lâm Đồng
    lat, lon = PROVINCES_GPS[selected_province]
    st.caption(f"Tọa độ GPS {selected_province}: Vĩ độ {lat:.4f}, Kinh độ {lon:.4f}")

    uploaded_file = st.file_uploader("Tải ảnh lá cà chua cần kiểm tra:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Ảnh gốc tải lên", use_container_width=True)

with col2:
    st.subheader("2. Kết quả phân tích Deep Learning & Thời tiết")
    if uploaded_file:
        temp, humidity, rain = get_weather(lat, lon)
        
        st.markdown(f"**Thời tiết hiện tại tại {selected_province}:**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Nhiệt độ", f"{temp} °C")
        m2.metric("Độ ẩm", f"{humidity} %")
        m3.metric("Lượng mưa", f"{rain} mm")
        st.markdown("---")
        
        # Nhận diện bệnh bằng mô hình EfficientNetV2-S
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probs, dim=1)
            
        raw_class = class_names[predicted_idx.item()]
        conf_score = confidence.item() * 100
        is_healthy = ("Healthy" in raw_class or "healthy" in raw_class)

        # Lấy thông tin chi tiết về bệnh
        vi_name, symptoms, treatment = DISEASE_DETAILS.get(
            raw_class, 
            (raw_class, "Chưa có thông tin triệu chứng.", "Tham khảo ý kiến cán bộ bảo vệ thực vật.")
        )

        # Trực quan hóa ảnh với Tiêu đề Xanh/Đỏ
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img)
        title_color = "green" if is_healthy else "red"
        ax.set_title(f"Predicted: {vi_name} ({conf_score:.1f}%)", color=title_color, fontsize=12, fontweight="bold", pad=12)
        ax.axis("off")
        
        st.pyplot(fig)
        plt.close(fig)

        # Hiển thị thông tin chẩn đoán & phác đồ
        st.markdown("### Kết quả chẩn đoán chi tiết:")
        if is_healthy:
            st.success(f"**Trạng thái:** {vi_name} (Độ tin cậy: {conf_score:.1f}%)")
            st.write(f"**Hướng dẫn chăm sóc:** {treatment}")
        else:
            st.error(f"**Phát hiện bệnh:** {vi_name} (Độ tin cậy: {conf_score:.1f}%)")
            st.write(f"**Triệu chứng:** {symptoms}")
            st.write(f"**Phác đồ điều trị:** {treatment}")

        st.markdown("---")
        st.markdown("### Đánh giá nguy cơ bùng phát & lây lan dịch bệnh:")
        if not is_healthy:
            if humidity > 80 and 18 <= temp <= 25:
                st.error(f"🔴 **CẤP BÁO LÂY LAN:** Nhiệt độ ({temp}°C) và độ ẩm ({humidity}%) cực kỳ thuận lợi cho vết bệnh này bào tử hóa và bùng phát lây lan nhanh ra toàn bộ vườn!")
            elif humidity > 70:
                st.warning(f"🟠 **CẢNH BÁO:** Độ ẩm cao ({humidity}%). Bệnh có nguy cơ lây sang các cây lân cận, cần phun thuốc kiểm soát và tỉa bớt lá.")
            else:
                st.info("🟢 **THỜI TIẾT KHÔ RÁO:** Tốc độ lây lan của vết bệnh sẽ chậm lại.")
        else:
            if humidity > 80 and 18 <= temp <= 25:
                st.warning(f"🟠 **CẢNH BÁO MÔI TRƯỜNG:** Thời tiết hiện tại ({temp}°C, độ ẩm {humidity}%) rất dễ phát sinh nấm bệnh. Cần chú ý quan sát vườn thường xuyên.")
            else:
                st.success("🟢 **MÔI TRƯỜNG AN TOÀN:** Điều kiện thời tiết hiện tại ít nguy cơ phát sinh dịch bệnh.")
    else:
        st.info("Vui lòng chọn Tỉnh/Thành phố và tải ảnh lá cà chua ở cột bên trái để ứng dụng bắt đầu chẩn đoán.")
