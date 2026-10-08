import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
import os

# ---------------------------------------------------------
# 1. CẤU HÌNH TRANG STREAMLIT (KHÔNG DÙNG ICON/EMOJI)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Chẩn Đoán Bệnh Lá Cà Chua AI",
    page_icon=None,
    layout="wide"
)

# ---------------------------------------------------------
# 2. DANH SÁCH 34 TỈNH THÀNH SAU SÁP NHẬP (TỪ 01/07/2025)
# ---------------------------------------------------------
PROVINCES_34 = [
    "TP Hà Nội",
    "TP Hồ Chí Minh",
    "TP Đà Nẵng",
    "TP Hải Phòng",
    "TP Cần Thơ",
    "TP Huế",
    "Tỉnh An Giang",
    "Tỉnh Bắc Ninh",
    "Tỉnh Cao Bằng",
    "Tỉnh Cà Mau",
    "Tỉnh Đắk Lắk",
    "Tỉnh Điện Biên",
    "Tỉnh Đồng Nai",
    "Tỉnh Đồng Tháp",
    "Tỉnh Gia Lai",
    "Tỉnh Hà Tĩnh",
    "Tỉnh Hưng Yên",
    "Tỉnh Khánh Hòa",
    "Tỉnh Lai Châu",
    "Tỉnh Lâm Đồng",
    "Tỉnh Lạng Sơn",
    "Tỉnh Lào Cai",
    "Tỉnh Nghệ An",
    "Tỉnh Ninh Bình",
    "Tỉnh Phú Thọ",
    "Tỉnh Quảng Ngãi",
    "Tỉnh Quảng Ninh",
    "Tỉnh Quảng Trị",
    "Tỉnh Sơn La",
    "Tỉnh Tây Ninh",
    "Tỉnh Thái Nguyên",
    "Tỉnh Thanh Hóa",
    "Tỉnh Tuyên Quang",
    "Tỉnh Vĩnh Long"
]

# ---------------------------------------------------------
# 3. CƠ SỞ DỮ LIỆU 11 LỚP BỆNH VÀ BẢN CHẤT BỆNH HỌC
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 4. HÀM ĐÁNH GIÁ TÌNH TRẠNG VÀ ĐƯA RA LỜI KHUYÊN TÙY BIẾN
# ---------------------------------------------------------
def generate_tailored_advice(pred_key, weather_condition, leaf_status, temperature):
    info = NORMALIZED_DB.get(pred_key, {
        "vn_name": "Không xác định",
        "type": "Không rõ",
        "symptoms": "Chưa có thông tin mô tả chi tiết.",
        "remedy": "Tham khảo ý kiến chuyên gia nông nghiệp địa phương."
    })

    disease_type = info["type"]
    status_report = []
    advice_list = []

    if pred_key == "healthy":
        if "Héo" in leaf_status and ("Mát mẻ" in weather_condition or temperature <= 28):
            status_report.append("Tình trạng: Lá bị héo nhẹ do thiếu nước hoặc thiếu ẩm tạm thời.")
            status_report.append("Khả năng phục hồi: CAO (100%). Do thời tiết đang mát mẻ, cây sẽ phục hồi nhanh chóng sau khi được cấp nước.")
            advice_list.append("Tưới gốc bổ sung vào buổi sáng sớm hoặc chiều mát. Không cần phun thuốc hóa học.")
        elif "Héo" in leaf_status and ("Nắng" in weather_condition or temperature > 32):
            status_report.append("Tình trạng: Lá bị mất nước cấp tính do nhiệt độ cao gây ra.")
            status_report.append("Khả năng phục hồi: TRUNG BÌNH. Cần che mát kịp thời để tránh cháy chồi.")
            advice_list.append("Che lưới giảm nắng và tưới giữ ẩm gốc, tránh tưới nước lên lá khi đang nắng gắt.")
        else:
            status_report.append("Tình trạng: Lá cà chua sinh trưởng tốt, không phát hiện mầm bệnh nguy hiểm.")
            status_report.append("Khả năng phục hồi: Hoàn hảo.")
            advice_list.append(info["remedy"])
        return status_report, advice_list

    if disease_type == "Virus":
        status_report.append(f"Tình trạng: Cây đã bị nhiễm {info['vn_name']}.")
        status_report.append("Khả năng phục hồi: KHÔNG THỂ PHỤC HỒI bằng điều chỉnh thời tiết hay tưới nước.")
        advice_list.append(info["remedy"])
        advice_list.append("Nhổ bỏ và tiêu hủy cây bệnh nặng để ngăn chặn virus lây lan sang các cây khác trong vườn.")
        return status_report, advice_list

    if "Héo" in leaf_status or "Cháy" in leaf_status:
        if "Mát mẻ" in weather_condition or temperature <= 26:
            status_report.append(f"Tình trạng: Phát hiện {info['vn_name']} đang thuyên giảm hoặc ở giai đoạn đầu.")
            status_report.append("Khả năng phục hồi: KHẢ NĂNG PHỤC HỒI CAO. Thời tiết mát mẻ giúp cây giảm mất sức, khi phun thuốc trị nấm/vi khuẩn cây sẽ nhanh ra mầm mới.")
            advice_list.append("Cắt bỏ phần lá bị bệnh nặng và phun thuốc đặc trị theo hướng dẫn.")
            advice_list.append("Bón thêm vi lượng hoặc phân bón lá để kích thích cây ra chồi mới.")
        elif "Mưa" in weather_condition or "Độ ẩm" in weather_condition:
            status_report.append(f"Tình trạng: {info['vn_name']} đang có nguy cơ bùng phát mạnh do độ ẩm cao.")
            status_report.append("Khả năng phục hồi: TRUNG BÌNH. Nguy cơ lây lan nhanh nếu không xử lý ngay.")
            advice_list.append("Ngừng ngay việc tưới nước lên lá. Phun thuốc phòng và trị ngay khi tạnh mưa.")
            advice_list.append("Khơi thông rãnh thoát nước trong vườn để tránh ngập úng gốc.")
        else:
            status_report.append(f"Tình trạng: {info['vn_name']} kết hợp với nhiệt độ cao làm lá khô nhanh hơn.")
            status_report.append("Khả năng phục hồi: TRUNG BÌNH.")
            advice_list.append("Phun thuốc trị bệnh vào chiều mát và kết hợp che nắng nhẹ cho vườn.")
    else:
        status_report.append(f"Tình trạng: Phát hiện {info['vn_name']}.")
        status_report.append("Khả năng phục hồi: TỐT nếu can thiệp kịp thời.")
        advice_list.append(info["remedy"])

    return status_report, advice_list

# ---------------------------------------------------------
# 5. LOAD MODEL TỪ GITHUB RELEASES (KHẮC PHỤC LỖI PYTORCH 2.6)
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
    
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

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

# ---------------------------------------------------------
# 6. GIAO DIỆN CHÍNH CỦA ỨNG DỤNG
# ---------------------------------------------------------
def main():
    st.title("Chẩn Đoán Bệnh Lá Cà Chua Bằng AI")
    st.write("Ứng dụng phân tích hình ảnh lá cà chua, kết hợp với tình trạng thời tiết và biểu hiện của lá để đưa ra chẩn đoán và lời khuyên phục hồi chính xác.")

    st.sidebar.title("Vị Trí Và Môi Trường")
    
    selected_province = st.sidebar.selectbox("Chọn Tỉnh/Thành phố (34 tỉnh thành):", PROVINCES_34)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Thông Tin Môi Trường Và Biểu Hiện Lá")
    
    weather_condition = st.sidebar.selectbox(
        "Thời tiết hiện tại:",
        ["Mát mẻ / Ôn hòa", "Nắng ói / Nắng gắt", "Mưa nhiều / Độ ẩm cao"]
    )
    
    temperature = st.sidebar.slider("Nhiệt độ môi trường (°C):", min_value=15, max_value=42, value=25)
    
    leaf_status = st.sidebar.selectbox(
        "Biểu hiện ngoại quan của lá:",
        ["Lá bình thường", "Lá bị héo / Rủ ngọn", "Lá bị cháy xám / Vàng đốm", "Lá bị xoăn / Biến dạng"]
    )

    st.sidebar.markdown("---")
    st.sidebar.write(f"Khu vực: {selected_province}")
    st.sidebar.write(f"Thời tiết: {weather_condition} ({temperature}°C)")
    st.sidebar.write(f"Biểu hiện lá: {leaf_status}")

    try:
        model, class_names = load_model()
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình AI: {e}")
        return

    uploaded_file = st.file_uploader("Chọn ảnh lá cà chua để kiểm tra (JPG, PNG, JPEG)...", type=["jpg", "png", "jpeg"])

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

        status_reports, tailored_advices = generate_tailored_advice(pred_key, weather_condition, leaf_status, temperature)

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
            st.markdown("### Triệu Chứng Đặc Trưng:")
            st.write(info["symptoms"])

            st.markdown("---")
            st.markdown("### Biểu Hiện & Khả Năng Phục Hồi:")
            for report in status_reports:
                st.write(f"- {report}")

            st.markdown("### Hướng Dẫn Khắc Phục Tương Ứng:")
            for adv in tailored_advices:
                st.info(adv)

if __name__ == "__main__":
    main()
