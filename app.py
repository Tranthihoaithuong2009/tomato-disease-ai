import os
import requests
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. CẤU HÌNH TRANG STREAMLIT
st.set_page_config(
    page_title="Chẩn Đoán Bệnh Lá Cà Chua AI",
    page_icon=None,
    layout="wide"
)

# 2. DANH SÁCH 34 TỈNH THÀNH SAU SÁP NHẬP (TỪ 01/07/2025)
PROVINCES_34 = [
    "TP Hà Nội", "TP Hồ Chí Minh", "TP Đà Nẵng", "TP Hải Phòng", "TP Cần Thơ", "TP Huế",
    "Tỉnh An Giang", "Tỉnh Bắc Ninh", "Tỉnh Cao Bằng", "Tỉnh Cà Mau", "Tỉnh Đắk Lắk",
    "Tỉnh Điện Biên", "Tỉnh Đồng Nai", "Tỉnh Đồng Tháp", "Tỉnh Gia Lai", "Tỉnh Hà Tĩnh",
    "Tỉnh Hưng Yên", "Tỉnh Khánh Hòa", "Tỉnh Lai Châu", "Tỉnh Lâm Đồng", "Tỉnh Lạng Sơn",
    "Tỉnh Lào Cai", "Tỉnh Nghệ An", "Tỉnh Ninh Bình", "Tỉnh Phú Thọ", "Tỉnh Quảng Ngãi",
    "Tỉnh Quảng Ninh", "Tỉnh Quảng Trị", "Tỉnh Sơn La", "Tỉnh Tây Ninh", "Tỉnh Thái Nguyên",
    "Tỉnh Thanh Hóa", "Tỉnh Tuyên Quang", "Tỉnh Vĩnh Long"
]

# 3. CƠ SỞ DỮ LIỆU 11 LỚP BỆNH
DISEASE_DATABASE = {
    "Bacterial_spot": {
        "vn_name": "Bệnh Đốm Vi Khuẩn (Bacterial Spot)",
        "type": "Vi khuẩn",
        "symptoms": "Xuất hiện các đốm nhỏ màu nâu đen, mọng nước trên bề mặt lá. Lá vàng và rụng dưới gốc.",
        "remedy": "Phun thuốc gốc đồng (Copper Hydroxide hoặc Kasugamycin). Tỉa bớt lá già và giữ vườn thông thoáng."
    },
    "Early_blight": {
        "vn_name": "Bệnh Đốm Vòng / Cháy Lá Sớm (Early Blight)",
        "type": "Nấm",
        "symptoms": "Vết bệnh hình tròn có các vòng đồng tâm màu nâu đen, lá bị vàng xung quanh đốm.",
        "remedy": "Phun thuốc gốc Mancozeb, Chlorothalonil hoặc Difenoconazole. Luân canh cây trồng."
    },
    "Late_blight": {
        "vn_name": "Bệnh Sương Mai / Cháy Lá Muộn (Late Blight)",
        "type": "Nấm",
        "symptoms": "Vết đốm màu xám xanh mọng nước, phát triển nhanh làm cháy khô toàn bộ lá và cành.",
        "remedy": "Phun ngay Metalaxyl, Dimethomorph hoặc Ridomil Gold. Ngừng tưới nước lên lá."
    },
    "Leaf_mold": {
        "vn_name": "Bệnh Mốc Lá (Leaf Mold)",
        "type": "Nấm",
        "symptoms": "Mặt trên lá có đốm vàng nhạt, mặt dưới xuất hiện lớp mốc màu xám xỉn hoặc nâu nhạt.",
        "remedy": "Phun thuốc gốc Đồng hoặc Carbendazim. Giảm độ ẩm và tăng cường thông gió."
    },
    "Septoria_leaf_spot": {
        "vn_name": "Bệnh Đốm Lá Septoria (Septoria Leaf Spot)",
        "type": "Nấm",
        "symptoms": "Các đốm tròn nhỏ màu xám nhạt ở giữa, viền nâu đen xuất hiện nhiều ở các lá gốc.",
        "remedy": "Phun Azoxystrobin hoặc Difenoconazole. Thu gom và tiêu hủy lá bệnh rụng dưới gốc."
    },
    "Spider_mites": {
        "vn_name": "Nhện Đỏ Hại Lá (Two-Spotted Spider Mites)",
        "type": "Côn trùng hại",
        "symptoms": "Lá xuất hiện các chấm nhỏ lốm đốm màu vàng, mặt dưới lá có màng tơ mỏng.",
        "remedy": "Phun thuốc đặc trị nhện như Abamectin, Hexythiazox hoặc Propargite. Tăng độ ẩm vườn."
    },
    "Target_Spot": {
        "vn_name": "Bệnh Đốm Mục Tiêu (Target Spot)",
        "type": "Nấm",
        "symptoms": "Vết bệnh hình tròn màu nâu với tâm sáng hơn, tạo hình dạng giống bia bắn.",
        "remedy": "Sử dụng thuốc trừ nấm chứa Chlorothalonil hoặc Pyraclostrobin."
    },
    "Yellow_Leaf_Curl_Virus": {
        "vn_name": "Bệnh Xoăn Lá Vàng Do Virus (Yellow Leaf Curl Virus)",
        "type": "Virus",
        "symptoms": "Lá bị xoăn ngửa lên trên, phiến lá nhỏ lại, màu vàng chanh, cây lùn còi cọc và không ra quả.",
        "remedy": "Không thể chữa khỏi bằng thuốc. Cần diệt bọ phấn trắng (côn trùng truyền bệnh) và nhổ bỏ cây bệnh."
    },
    "Mosaic_virus": {
        "vn_name": "Bệnh Khảm Lá Do Virus (Mosaic Virus)",
        "type": "Virus",
        "symptoms": "Lá loang lổ các vệt màu xanh đậm và xanh nhạt xen kẽ, lá bị biến dạng và nhăn nheo.",
        "remedy": "Nhổ bỏ cây bệnh để tránh lây lan. Khử trùng dụng cụ cắt tỉa và diệt rệp muỗi truyền bệnh."
    },
    "Powdery_mildew": {
        "vn_name": "Bệnh Phấn Trắng (Powdery Mildew)",
        "type": "Nấm",
        "symptoms": "Lớp bột trắng như phấn bao phủ trên bề mặt lá, làm lá khô xơ và rụng.",
        "remedy": "Phun Sulfur (Lưu huỳnh), Hexaconazole hoặc Myclobutanil."
    },
    "Healthy": {
        "vn_name": "Lá Khỏe Mạnh (Healthy)",
        "type": "Không có bệnh",
        "symptoms": "Lá xanh tươi, không có dấu hiệu bị nấm, vi khuẩn hay sâu bệnh tấn công.",
        "remedy": "Tiếp tục chăm sóc, tưới nước và bón phân định kỳ cân đối N-P-K."
    }
}

NORMALIZED_DB = {k.lower().replace("_", "").replace(" ", "").replace("-", ""): v for k, v in DISEASE_DATABASE.items()}

# 4. HÀM ĐÁNH GIÁ TÌNH TRẠNG VÀ PHỤC HỒI
def generate_tailored_advice(pred_key, temp, humidity, rainfall, leaf_status):
    info = NORMALIZED_DB.get(pred_key, {
        "vn_name": "Không xác định",
        "type": "Không rõ",
        "symptoms": "Chưa có thông tin mô tả chi tiết.",
        "remedy": "Tham khảo ý kiến chuyên gia nông nghiệp địa phương."
    })

    disease_type = info["type"]
    status_report = []
    advice_list = []

    env_summary = f"Điều kiện môi trường: Nhiệt độ {temp}°C, Độ ẩm {humidity}%, Lượng mưa {rainfall} mm."
    status_report.append(env_summary)

    if pred_key == "healthy":
        if "Héo" in leaf_status and temp <= 28 and humidity >= 50:
            status_report.append("Tình trạng: Lá bị héo nhẹ do thiếu nước hoặc suy kiệt tạm thời.")
            status_report.append("Khả năng phục hồi: CAO (100%). Thời tiết mát mẻ và độ ẩm tốt sẽ giúp cây phục hồi nhanh chóng sau khi được cấp nước.")
            advice_list.append("Tưới bổ sung nước vào gốc vào buổi sáng sớm hoặc chiều mát.")
        elif "Héo" in leaf_status and temp > 32:
            status_report.append("Tình trạng: Lá bị héo rủ do nhiệt độ cao làm thoát hơi nước cấp tính.")
            status_report.append("Khả năng phục hồi: TRUNG BÌNH. Cần che mát kịp thời.")
            advice_list.append("Che lưới giảm nắng và tưới giữ ẩm gốc, tránh tưới trực tiếp lên lá khi trời đang nắng gắt.")
        else:
            status_report.append("Tình trạng: Lá cà chua sinh trưởng tốt, không phát hiện mầm bệnh.")
            status_report.append("Khả năng phục hồi: Hoàn hảo.")
            advice_list.append(info["remedy"])
        return status_report, advice_list

    if disease_type == "Virus":
        status_report.append(f"Tình trạng: Cây đã nhiễm {info['vn_name']}.")
        status_report.append("Khả năng phục hồi: KHÔNG THỂ PHỤC HỒI bằng thời tiết hay tưới nước.")
        advice_list.append(info["remedy"])
        advice_list.append("Nhổ bỏ cây nhiễm bệnh để tránh lây lan toàn vườn.")
        return status_report, advice_list

    if "Héo" in leaf_status or "Cháy" in leaf_status:
        if temp <= 26 and rainfall < 20:
            status_report.append(f"Tình trạng: Phát hiện {info['vn_name']}.")
            status_report.append("Khả năng phục hồi: KHẢ NĂNG PHỤC HỒI CAO. Thời tiết mát mẻ giúp cây không bị mất sức, khi phun thuốc trị bệnh cây sẽ nhanh hồi phục.")
            advice_list.append("Tỉa bỏ các lá bệnh nặng và phun thuốc đặc trị theo hướng dẫn.")
            advice_list.append("Bón bổ sung phân bón lá để kích thích chồi mới.")
        elif humidity > 80 or rainfall >= 20:
            status_report.append(f"Tình trạng: {info['vn_name']} có nguy cơ bùng phát mạnh do độ ẩm cao và mưa nhiều ({rainfall} mm).")
            status_report.append("Khả năng phục hồi: TRUNG BÌNH. Cần xử lý ngay để tránh lây lan.")
            advice_list.append("Ngừng tưới nước lên lá. Phun thuốc phòng ngừa ngay sau khi tạnh mưa.")
            advice_list.append("Khơi thông rãnh thoát nước trong vườn.")
        else:
            status_report.append(f"Tình trạng: {info['vn_name']} phát triển do nhiệt độ cao ({temp}°C).")
            status_report.append("Khả năng phục hồi: TRUNG BÌNH.")
            advice_list.append("Phun thuốc trị bệnh vào chiều mát và kết hợp che nắng nhẹ.")
    else:
        status_report.append(f"Tình trạng: Phát hiện {info['vn_name']}.")
        status_report.append("Khả năng phục hồi: TỐT nếu xử lý kịp thời.")
        advice_list.append(info["remedy"])

    return status_report, advice_list

# 5. HÀM NẠP CHECKPOINT DẠNG CHUẨN
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
def load_model_from_path(model_path):
    return load_checkpoint_file(model_path)

# 6. GIAO DIỆN CHÍNH
def main():
    st.title("Chẩn Đoán Bệnh Lá Cà Chua Bằng AI")
    st.write("Ứng dụng phân tích hình ảnh lá cà chua, kết hợp các thông số thời tiết (nhiệt độ, độ ẩm, lượng mưa) để đưa ra chẩn đoán và hướng khắc phục.")

    st.sidebar.title("Vị Trí Và Thời Tiết")
    selected_province = st.sidebar.selectbox("Chọn Tỉnh/Thành phố (34 tỉnh thành):", PROVINCES_34)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Thông Số Môi Trường Vườn Trồng")
    
    temp = st.sidebar.slider("Nhiệt độ (°C):", min_value=10, max_value=45, value=25)
    humidity = st.sidebar.slider("Độ ẩm (%):", min_value=20, max_value=100, value=75)
    rainfall = st.sidebar.number_input("Lượng mưa (mm):", min_value=0.0, max_value=300.0, value=0.0, step=5.0)
    
    leaf_status = st.sidebar.selectbox(
        "Biểu hiện ngoại quan của lá:",
        ["Lá bình thường", "Lá bị héo / Rủ ngọn", "Lá bị cháy xám / Vàng đốm", "Lá bị xoăn / Biến dạng"]
    )

    st.sidebar.markdown("---")
    
    model = None
    class_names = []

    # 1. Tự động kiểm tra file local .pth
    pth_files = [f for f in os.listdir(".") if f.endswith(".pth") and os.path.getsize(f) > 5000000]
    
    if pth_files:
        target_file = pth_files.pop(0)
        try:
            model, class_names = load_model_from_path(target_file)
        except Exception:
            model = None

    # 2. Thử tải tự động từ GitHub Release v1.0
    if model is None:
        local_default = "tomato_model_best.pth"
        try:
            res = requests.get(MODEL_URL, allow_redirects=True, timeout=10)
            if res.status_code == 200 and len(res.content) > 5000000:
                with open(local_default, "wb") as f:
                    f.write(res.content)
                model, class_names = load_model_from_path(local_default)
        except Exception:
            model = None

    # 3. Hiển thị khung tải file ngay MÀN HÌNH CHÍNH nếu chưa nạp được tự động
    if model is None:
        st.info("Bước 1: Nạp file mô hình AI (Hỗ trợ file tới 200MB từ máy tính)")
        uploaded_model_file = st.file_uploader("Chọn file tomato_model_best (3).pth trên máy tính của bạn:", type=["pth"])
        
        if uploaded_model_file is not None:
            try:
                model, class_names = load_checkpoint_file(uploaded_model_file)
                st.success("Đã nạp mô hình thành công!")
            except Exception as e:
                st.error(f"Lỗi đọc file mô hình: {e}")
                return
        else:
            return

    # Tải ảnh lá cà chua lên kiểm tra
    st.markdown("---")
    st.subheader("Bước 2: Tải ảnh lá cà chua lên để chẩn đoán")
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

        status_reports, tailored_advices = generate_tailored_advice(pred_key, temp, humidity, rainfall, leaf_status)

        with col2:
            st.subheader("Kết Quả Chẩn Đoán AI")
            if pred_key == "healthy":
                st.success(f"Kết quả: {info['vn_name']}")
            else:
                st.error(f"Kết quả: {info['vn_name']}")
            
            st.metric(label="Độ tin cậy của AI", value=f"{conf_percent:.2f}%")
            st.write(f"Tác nhân gây bệnh: {info['type']}")
            st.write(f"Khu vực ghi nhận: {selected_province}")

            st.markdown("---")
            st.markdown("### Thông Số Thời Tiết Đã Ghi Nhận:")
            st.write(f"- Nhiệt độ: {temp}°C")
            st.write(f"- Độ ẩm: {humidity}%")
            st.write(f"- Lượng mưa: {rainfall} mm")

            st.markdown("---")
            st.markdown("### Triệu Chứng Đặc Trưng:")
            st.write(info["symptoms"])

            st.markdown("---")
            st.markdown("### Đánh Giá Tình Trạng & Khả Năng Phục Hồi:")
            for report in status_reports:
                st.write(f"- {report}")

            st.markdown("### Hướng Dẫn Khắc Phục Tương Ứng:")
            for adv in tailored_advices:
                st.info(adv)

if __name__ == "__main__":
    main()
