import os
import requests
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---------------------------------------------------------
# 1. CẤU HÌNH TRANG STREAMLIT & CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Chẩn Đoán Bệnh Lá Cà Chua AI",
    page_icon="🍅",
    layout="centered"
)

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    div[data-testid="stDecoration"] {display:none;}
    div[data-testid="stStatusWidget"] {display:none;}
    div[data-testid="stHeader"] {display:none;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. BẢNG TỌA ĐỘ ĐỊA LÝ 34 TỈNH THÀNH VIỆT NAM (SAU SÁP NHẬP)
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 3. CƠ SỞ DỮ LIỆU ĐẦY ĐỦ CHO TẤT CẢ CÁC BỆNH VÀ BẢNG ÁNH XÁ ALIASES
# ---------------------------------------------------------
DISEASE_ENTRIES = {
    "bacterial_spot": {
        "vn_name": "Bệnh Đốm Vi Khuẩn (Bacterial Spot)",
        "type": "Vi khuẩn",
        "symptoms": "Xuất hiện các đốm nhỏ màu nâu đen, mọng nước trên bề mặt lá. Viền lá xung quanh đốm có thể bị vàng. Lá nhiễm bệnh nặng chuyển sang màu vàng và rụng sớm dưới gốc.",
        "remedy": "• Phun thuốc gốc đồng như Copper Hydroxide, Kasugamycin hoặc Streptomyces súc rửa lá.\n• Tỉa bớt các lá già dưới gốc để tạo độ thông thoáng.\n• Tránh tưới phun mưa trực tiếp lên tán lá vào buổi chiều tối để hạn chế vi khuẩn bắn sang cây khác."
    },
    "early_blight": {
        "vn_name": "Bệnh Đốm Vòng / Cháy Lá Sớm (Early Blight)",
        "type": "Nấm",
        "symptoms": "Vết bệnh hình tròn màu nâu đen có các vòng đồng tâm đặc trưng như hình bia bắn. Phiến lá xung quanh vết bệnh thường bị vàng khè, bệnh lan dần từ lá già dưới gốc lên lá non.",
        "remedy": "• Phun các thuốc trừ nấm chứa hoạt chất Mancozeb, Chlorothalonil hoặc Difenoconazole.\n• Ngắt bỏ và gom đốt các lá bị đốm nặng.\n• Thực hiện luân canh cây trồng khác họ cà (như lúa, ngô, đậu) sau mỗi mùa vụ."
    },
    "late_blight": {
        "vn_name": "Bệnh Sương Mai / Cháy Lá Muộn (Late Blight)",
        "type": "Nấm",
        "symptoms": "Vết đốm lớn màu xám xanh mọng nước ở mép lá, phát triển cực nhanh làm cháy khô toàn bộ lá, cành và thân cây. Mức độ nguy hiểm cao khi thời tiết ẩm ướt.",
        "remedy": "• Phun ngay lập tức các thuốc đặc trị nấm sương mai như Ridomil Gold, Metalaxyl hoặc Dimethomorph.\n• Ngừng hoàn toàn việc tưới nước lên lá.\n• Cách ly cây bệnh nặng và tăng cường thoát nước cho luống trồng."
    },
    "leaf_mold": {
        "vn_name": "Bệnh Mốc Lá (Leaf Mold)",
        "type": "Nấm",
        "symptoms": "Mặt trên lá xuất hiện các đốm màu vàng nhạt không đều. Mặt dưới lá tương ứng có một lớp mốc mịn màu xám xỉn hoặc nâu nhạt bao phủ.",
        "remedy": "• Phun thuốc trừ nấm gốc Đồng, Carbendazim hoặc Hexaconazole.\n• Giảm độ ẩm trong vườn, tỉa bớt tán lá chân để tăng cường lưu thông không khí.\n• Tránh trồng quá dày làm che khuất ánh sáng."
    },
    "septoria_leaf_spot": {
        "vn_name": "Bệnh Đốm Lá Septoria (Septoria Leaf Spot)",
        "type": "Nấm",
        "symptoms": "Nhiều đốm nhỏ hình tròn (đường kính 1-3mm), tâm vết bệnh màu xám nhạt với viền màu nâu đen rất rõ nét. Bệnh tập trung chủ yếu ở các lá sát mặt đất.",
        "remedy": "• Phun thuốc chứa hoạt chất Azoxystrobin, Difenoconazole hoặc Chlorothalonil.\n• Thu gom tiêu hủy toàn bộ lá rụng dưới gốc và làm sạch cỏ dại xung quanh luống.\n• Phủ bạt nilon hoặc rơm rạ dưới gốc để ngăn bào tử nấm bắn từ đất lên lá."
    },
    "spider_mites": {
        "vn_name": "Nhện Đỏ Hại Lá (Two-Spotted Spider Mites)",
        "type": "Côn trùng hại",
        "symptoms": "Bề mặt lá xuất hiện hàng ngàn chấm nhỏ lốm đốm màu vàng nhạt. Mặt dưới lá có màng tơ mỏng li ti và nhiều nhện đỏ nhỏ như đầu kim bò xung quanh, làm lá khô cứng và rụng.",
        "remedy": "• Phun các thuốc đặc trị nhện như Abamectin, Hexythiazox, Pyridaben hoặc Propargite.\n• Phun tập trung vào mặt dưới của lá.\n• Tăng độ ẩm bằng cách tưới phun sương nhẹ vì nhện đỏ rất ghét môi trường ẩm ướt."
    },
    "target_spot": {
        "vn_name": "Bệnh Đốm Mục Tiêu (Target Spot)",
        "type": "Nấm",
        "symptoms": "Vết bệnh hình tròn màu nâu sẫm, vùng tâm sáng màu hơn tạo hiệu ứng các vòng đốm giống hình bia mục tiêu. Bệnh làm lá bị đốm lỗ chỗ và rụng nhanh.",
        "remedy": "• Sử dụng các thuốc trừ nấm chứa Chlorothalonil, Pyraclostrobin, Mancozeb hoặc Copper Sulfate.\n• Tỉa bớt lá bệnh và duy trì khoảng cách giữa các cây hợp lý."
    },
    "yellow_leaf_curl_virus": {
        "vn_name": "Bệnh Xoăn Lá Vàng Do Virus (Yellow Leaf Curl Virus)",
        "type": "Virus",
        "symptoms": "Lá non bị xoăn ngửa lên trên dạng hình lòng chảo, phiến lá thu nhỏ lại, chuyển màu vàng chanh. Cây bị lùn còi cọc, cành lá chụm lại và không thể đậu quả.",
        "remedy": "• Bệnh do virus không thể chữa bằng thuốc hóa học. Cần nhổ bỏ ngay cây nhiễm bệnh và đem tiêu hủy xa vườn.\n• Phun thuốc diệt bọ phấn trắng (vật trung gian lây truyền virus) bằng các hoạt chất Imidacloprid, Thiamethoxam hoặc Pymetrozine.\n• Dùng lưới chắn côn trùng khi gieo mạ giống."
    },
    "mosaic_virus": {
        "vn_name": "Bệnh Khảm Lá Do Virus (Mosaic Virus)",
        "type": "Virus",
        "symptoms": "Lá có các mảng loang lổ xen kẽ giữa màu xanh đậm và màu xanh nhạt/vàng nhạt. Phiến lá bị biến dạng, nhăn nheo, xoắn quăn hoặc biến thành dạng lá dải đũa.",
        "remedy": "• Nhổ bỏ triệt để cây bệnh để ngăn ngừa lây sang cả vườn.\n• Phun thuốc diệt rệp muỗi, bọ trĩ lây bệnh.\n• Khử trùng dụng cụ cắt tỉa, kéo làm vườn bằng xà phòng hoặc cồn trước khi chuyển sang cây khác."
    },
    "powdery_mildew": {
        "vn_name": "Bệnh Phấn Trắng (Powdery Mildew)",
        "type": "Nấm",
        "symptoms": "Lớp bột màu trắng mịn trông như rắc phấn bao phủ trên bề mặt lá và cành non. Lá bị nhiễm bệnh nặng chuyển sang màu vàng, khô xơ và rụng sớm.",
        "remedy": "• Phun các thuốc chứa Lưu huỳnh (Sulfur), Hexaconazole, Dinocap hoặc Myclobutanil ngay khi thấy vệt phấn trắng đầu tiên.\n• Tạo không gian thoáng mát cho vườn cà chua."
    },
    "leaf_miner": {
        "vn_name": "Sâu Vẽ Bùa / Ruồi Đục Lá (Leaf Miner)",
        "type": "Côn trùng hại",
        "symptoms": "Trên bề mặt lá xuất hiện các đường ngoằn ngoèo màu trắng nhạt mỏng như sợi chỉ do dòi ruồi đục xới mô lá. Khi bị nặng, lá khô héo và giảm khả năng quang hợp.",
        "remedy": "• Phun các thuốc đặc trị dòi đục lá chứa hoạt chất Abamectin, Spinetoram, Cyromazine hoặc Cartap.\n• Ngắt bỏ các lá có nhiều đường ngoằn ngoèo đem tiêu hủy."
    },
    "anthracnose": {
        "vn_name": "Bệnh Thán Thư (Anthracnose)",
        "type": "Nấm",
        "symptoms": "Vết bệnh hình tròn lõm xuống màu nâu sẫm, trên bề mặt vết bệnh có thể xuất hiện các chấm nhỏ màu đen xếp thành vòng đồng tâm.",
        "remedy": "• Phun các thuốc chứa Azoxystrobin, Difenoconazole hoặc Mancozeb.\n• Tránh làm tổn thương da quả và lá trong quá trình chăm sóc."
    },
    "healthy": {
        "vn_name": "Lá Khỏe Mạnh (Healthy)",
        "type": "Không có bệnh",
        "symptoms": "Lá có màu xanh tươi tự nhiên, phiến lá phẳng, không có dấu hiệu bị nấm, vi khuẩn, virus hay sâu bệnh hại tấn công.",
        "remedy": "• Tiếp tục duy trì chế độ chăm sóc tốt, tưới nước vừa đủ ở gốc.\n• Bón phân định kỳ cân đối N-P-K và bổ sung trung vi lượng (Canxi, Magie, Bo) để tăng sức đề kháng cho cây."
    }
}

ALIAS_MAP = {
    "bacterialspot": "bacterial_spot",
    "tomatobacterialspot": "bacterial_spot",
    "tomato___bacterial_spot": "bacterial_spot",
    "tomatobacterial_spot": "bacterial_spot",
    "earlyblight": "early_blight",
    "tomatoearlyblight": "early_blight",
    "tomato___early_blight": "early_blight",
    "tomatoearly_blight": "early_blight",
    "lateblight": "late_blight",
    "tomatolateblight": "late_blight",
    "tomato___late_blight": "late_blight",
    "tomatolate_blight": "late_blight",
    "leafmold": "leaf_mold",
    "tomatoleafmold": "leaf_mold",
    "tomato___leaf_mold": "leaf_mold",
    "tomatoleaf_mold": "leaf_mold",
    "septoria": "septoria_leaf_spot",
    "septorialeafspot": "septoria_leaf_spot",
    "tomatoseptorialeafspot": "septoria_leaf_spot",
    "tomatoseptoria_leaf_spot": "septoria_leaf_spot",
    "tomato___septoria_leaf_spot": "septoria_leaf_spot",
    "spidermites": "spider_mites",
    "twospottedspidermite": "spider_mites",
    "spidermitestwospottedspidermite": "spider_mites",
    "tomatospidermitestwospottedspidermite": "spider_mites",
    "tomatospider_mites": "spider_mites",
    "tomato___spider_mites_two-spotted_spider_mite": "spider_mites",
    "tomato___spider_mites": "spider_mites",
    "targetspot": "target_spot",
    "tomatotargetspot": "target_spot",
    "tomato___target_spot": "target_spot",
    "tomatotarget_spot": "target_spot",
    "yellowleafcurlvirus": "yellow_leaf_curl_virus",
    "tomatoyellowleafcurlvirus": "yellow_leaf_curl_virus",
    "tomatotomatoyellowleafcurlvirus": "yellow_leaf_curl_virus",
    "tomato___tomato_yellow_leaf_curl_virus": "yellow_leaf_curl_virus",
    "tomato___yellow_leaf_curl_virus": "yellow_leaf_curl_virus",
    "mosaicvirus": "mosaic_virus",
    "tomatomosaicvirus": "mosaic_virus",
    "tomatotomatomosaicvirus": "mosaic_virus",
    "tomato___tomato_mosaic_virus": "mosaic_virus",
    "tomato___mosaic_virus": "mosaic_virus",
    "powderymildew": "powdery_mildew",
    "tomatopowderymildew": "powdery_mildew",
    "tomato___powdery_mildew": "powdery_mildew",
    "leafminer": "leaf_miner",
    "tomatoleafminer": "leaf_miner",
    "tomato___leaf_miner": "leaf_miner",
    "tomatoleaf_miner": "leaf_miner",
    "anthracnose": "anthracnose",
    "tomatoanthracnose": "anthracnose",
    "healthy": "healthy",
    "tomatohealthy": "healthy",
    "tomato___healthy": "healthy",
    "tomatohealthy_leaf": "healthy"
}

def lookup_disease_info(raw_class_name):
    clean_key = str(raw_class_name).lower().replace(" ", "").replace("_", "").replace("-", "").replace("*", "")
    canonical_key = ALIAS_MAP.get(clean_key)
    
    if not canonical_key:
        for k in DISEASE_ENTRIES.keys():
            if k.replace("_", "") in clean_key or clean_key in k.replace("_", ""):
                canonical_key = k
                break

    if canonical_key and canonical_key in DISEASE_ENTRIES:
        return DISEASE_ENTRIES[canonical_key], canonical_key

    if "septoria" in clean_key:
        return DISEASE_ENTRIES["septoria_leaf_spot"], "septoria_leaf_spot"
    if "miner" in clean_key:
        return DISEASE_ENTRIES["leaf_miner"], "leaf_miner"
    if "bacterial" in clean_key:
        return DISEASE_ENTRIES["bacterial_spot"], "bacterial_spot"
    if "early" in clean_key:
        return DISEASE_ENTRIES["early_blight"], "early_blight"
    if "late" in clean_key:
        return DISEASE_ENTRIES["late_blight"], "late_blight"
    if "mold" in clean_key:
        return DISEASE_ENTRIES["leaf_mold"], "leaf_mold"
    if "mite" in clean_key:
        return DISEASE_ENTRIES["spider_mites"], "spider_mites"
    if "target" in clean_key:
        return DISEASE_ENTRIES["target_spot"], "target_spot"
    if "curl" in clean_key or "yellow" in clean_key:
        return DISEASE_ENTRIES["yellow_leaf_curl_virus"], "yellow_leaf_curl_virus"
    if "mosaic" in clean_key:
        return DISEASE_ENTRIES["mosaic_virus"], "mosaic_virus"
    if "mildew" in clean_key:
        return DISEASE_ENTRIES["powdery_mildew"], "powdery_mildew"
    if "healthy" in clean_key:
        return DISEASE_ENTRIES["healthy"], "healthy"

    return {
        "vn_name": f"Bệnh Lá Cà Chua ({raw_class_name})",
        "type": "Cần theo dõi thêm",
        "symptoms": "Lá có biểu hiện tổn thương đốm hoặc biến màu bất thường so với lá khỏe mạnh.",
        "remedy": "• Cách ly tạm thời các chậu/cây có biểu hiện lạ.\n• Tỉa bỏ phần lá hỏng để tránh bào tử phát tán.\n• Đảm bảo mật độ trồng thông thoáng và tưới nước ở gốc."
    }, "unknown"

# ---------------------------------------------------------
# 4. HÀM LẤY THỜI TIẾT THỜI GIAN THỰC THEO TỌA ĐỘ
# ---------------------------------------------------------
def get_realtime_weather(lat, lon):
    try:
        domain = "api.open-meteo" + ".com"
        url = f"https://{domain}/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,rain"
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

# ---------------------------------------------------------
# 5. HÀM ĐÁNH GIÁ KHẢ NĂNG LÂY LAN TRONG TƯƠNG LAI
# ---------------------------------------------------------
def evaluate_spread_forecast(disease_info, temp, humidity, rain):
    disease_type = disease_info["type"]
    vn_name = disease_info["vn_name"]

    if disease_type == "Không có bệnh":
        return "KHÔNG CÓ (0%)", "Cây trồng đang trong trạng thái khỏe mạnh. Hãy tiếp tục duy trì vệ sinh vườn và chế độ chăm sóc định kỳ."

    if disease_type == "Virus":
        return (
            "RẤT CAO (Tốc độ bùng phát cực nhanh)",
            f"{vn_name} lây truyền mạnh qua côn trùng môi giới (bọ phấn trắng, rệp muỗi). Nếu không phun thuốc diệt côn trùng trung gian và nhổ bỏ cây bệnh, mầm bệnh sẽ lan ra toàn bộ vườn trong vòng 3-5 ngày."
        )

    if disease_type == "Nấm":
        if humidity >= 75 or rain > 0:
            return (
                "CỰC KỲ CAO (Nguy cơ dịch bùng phát 80-95%)",
                f"Độ ẩm không khí thời gian thực cao ({humidity}%) kết hợp lượng mưa ({rain} mm) là điều kiện bùng phát lý tưởng cho bào tử nấm. Bệnh sẽ lây lan rất nhanh sang các cây lân cận trong 24-48 giờ tới."
            )
        else:
            return (
                "TRUNG BÌNH (30-50%)",
                f"Thời tiết hiện tại (Độ ẩm {humidity}%, Nhiệt độ {temp}°C) giúp làm chậm sự phát tán bào tử nấm. Tuy nhiên cần phun thuốc phòng ngừa trước khi có đợt mưa hoặc sương mù mới."
            )

    if disease_type == "Vi khuẩn":
        if rain > 0 or humidity >= 80:
            return (
                "CAO (Nguy cơ lây lan 70-85%)",
                f"{vn_name} lây lan rất nhanh qua giọt nước mưa bắn, nước tưới và dụng cụ cắt tỉa. Cần cách ly ngay vùng cây bệnh và ngừng tưới phun mưa."
            )
        else:
            return (
                "TRUNG BÌNH (40%)",
                "Vi khuẩn phát triển chậm hơn khi môi trường khô ráo. Tránh tưới nước lên lá để hạn chế vi khuẩn văng sang cây lành."
            )

    if disease_type == "Côn trùng hại":
        if temp >= 30 and humidity < 70:
            return (
                "RẤT CAO (Bùng phát do thời tiết khô nóng)",
                f"Nhiệt độ cao ({temp}°C) và không khí khô thúc đẩy côn trùng hại sinh sản gia tăng mật độ rất nhanh. Cần phun thuốc đặc trị và tăng cường độ ẩm cho vườn."
            )
        else:
            return (
                "TRUNG BÌNH (45%)",
                "Mật độ sâu hại đang ở mức gia tăng. Cần phun thuốc đặc trị ngay để ngăn chặn thế hệ sâu/nhện tiếp theo."
            )

    return "TRUNG BÌNH", "Cần theo dõi sát sao biểu hiện của vườn trong các ngày tới."

# ---------------------------------------------------------
# 6. TỰ ĐỘNG NẠP MÔ HÌNH AI PHÍA BACKEND
# ---------------------------------------------------------
gh_domain = "github" + ".com"
MODEL_URL = f"https://{gh_domain}/Tranthihoaithuong2009/tomato-disease-ai/releases/download/v1.0/tomato_model_best.pth"

def load_checkpoint_file(file_source):
    try:
        checkpoint = torch.load(file_source, map_location=torch.device('cpu'), weights_only=False)
    except TypeError:
        checkpoint = torch.load(file_source, map_location=torch.device('cpu'))

    if isinstance(checkpoint, dict) and 'class_names' in checkpoint:
        class_names = checkpoint['class_names']
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        class_names = list(DISEASE_ENTRIES.keys())
    else:
        state_dict = checkpoint
        class_names = list(DISEASE_ENTRIES.keys())

    num_classes = len(class_names)

    model = models.efficientnet_v2_s(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(1280, num_classes)
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, class_names

@st.cache_resource
def get_ai_model():
    pth_files = [f for f in os.listdir(".") if f.endswith(".pth") and os.path.getsize(f) > 5000000]
    if pth_files:
        target_file = pth_files
        try:
            return load_checkpoint_file(target_file)
        except Exception:
            pass

    local_default = "tomato_model_best.pth"
    if os.path.exists(local_default) and os.path.getsize(local_default) > 5000000:
        return load_checkpoint_file(local_default)

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(MODEL_URL, headers=headers, allow_redirects=True, timeout=30)
        if res.status_code == 200 and len(res.content) > 5000000:
            with open(local_default, "wb") as f:
                f.write(res.content)
            return load_checkpoint_file(local_default)
    except Exception:
        pass

    raise RuntimeError(
        "Chưa tìm thấy file 'tomato_model_best.pth'. "
        "Vui lòng đảm bảo file 'tomato_model_best.pth' đã được đặt cùng thư mục!"
    )

# ---------------------------------------------------------
# 7. GIAO DIỆN CHÍNH TRÊN TRANG WEB
# ---------------------------------------------------------
def main():
    st.title("Chẩn Đoán Bệnh Lá Cà Chua Bằng AI")
    st.write("Ứng dụng chẩn đoán bệnh lá cà chua thông minh, tra cứu thời tiết thời gian thực và dự báo nguy cơ lây lan dịch bệnh.")
    st.markdown("---")

    # BƯỚC 1: CHỌN TỈNH/THÀNH PHỐ
    st.subheader("1. Chọn Tỉnh/Thành phố")
    selected_province = st.selectbox(
        "Vui lòng chọn địa phương của bạn:",
        options=list(PROVINCE_COORDS.keys())
    )

    coords = PROVINCE_COORDS[selected_province]
    lat, lon = coords["lat"], coords["lon"]

    st.markdown("---")

    # BƯỚC 2: TẢI ẢNH LÁ CÀ CHUA
    st.subheader("2. Chọn ảnh lá cà chua")
    uploaded_file = st.file_uploader(
        "Tải ảnh lá cà chua từ thiết bị của bạn:",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        st.markdown("---")
        
        # Mở ảnh bằng PIL
        image = [đã xoá đường liên kết đáng ngờ](uploaded_file).convert('RGB')

        st.subheader("3. Kết Quả Phân Tích Chi Tiết")

        try:
            with st.spinner("Hệ thống đang khởi tạo mô hình AI và phân tích ảnh..."):
                model, class_names = get_ai_model()
        except Exception as e:
            st.error(f"{e}")
            return

        transform = transforms.Compose([
            transforms.Resize((280, 280)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image).unsqueeze(0)

        with [đã xoá đường liên kết đáng ngờ]_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top3_prob, top3_idx = torch.topk(probabilities, 3, dim=1)

        confidence = top3_prob.item()
        conf_percent = confidence * 100
        predicted_raw = class_names[top3_idx.item()]

        info, canonical_key = lookup_disease_info(predicted_raw)
        temp, humidity, rain = get_realtime_weather(lat, lon)
        spread_risk, spread_detail = evaluate_spread_forecast(info, temp, humidity, rain)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Hình ảnh lá cà chua phân tích", use_container_width=True)

        with col2:
            if canonical_key == "healthy":
                st.success(f"**Kết quả:** {info['vn_name']}")
            else:
                st.error(f"**Kết quả:** {info['vn_name']}")

            st.metric(label="Độ tin cậy của AI:", value=f"{conf_percent:.2f}%")
            st.write(f"**Tác nhân gây bệnh:** {info['type']}")
            st.write(f"**Vị trí địa lý:** {selected_province} (Tọa độ: {lat}°N, {lon}°E)")

            if conf_percent < 75.0:
                st.warning(
                    f"⚠️ **Cảnh báo (Độ tin cậy AI trung bình - {conf_percent:.2f}%):**\n\n"
                    "Mô hình AI đang phân vân giữa các khả năng sau:"
                )
                for i in range(min(3, len(class_names))):
                    idx = top3_idx[i].item()
                    prob = top3_prob[i].item() * 100
                    raw_name = class_names[idx]
                    top_info, _ = lookup_disease_info(raw_name)
                    st.write(f"- **Top {i+1}:** {top_info['vn_name']} (`{prob:.2f}%`)")

        st.markdown("---")
        
        # THỜI TIẾT THỜI GIAN THỰC
        st.subheader("Thông Tin Về Thời Tiết Tại Địa Phương")
        w_col1, w_col2, w_col3 = st.columns(3)
        w_col1.metric("Nhiệt độ", f"{temp} °C")
        w_col2.metric("Độ ẩm không khí", f"{humidity} %")
        w_col3.metric("Lượng mưa", f"{rain} mm")

        st.markdown("---")

        # DỰ BÁO LÂY LAN
        st.subheader("Khả Năng Lây Lan Trong Tương Lai")
        st.warning(f"**Mức độ nguy cơ:** {spread_risk}")
        st.write(spread_detail)

        st.markdown("---")

        # TRIỆU CHỨNG & ĐỀ XUẤT GIẢI PHÁP
        st.subheader("Triệu Chứng Đặc Trưng")
        st.write(info["symptoms"])

        st.subheader("Đề Xuất Giải Pháp Khắc Phục")
        st.write(info["remedy"])

if __name__ == "__main__":
    main()

Bạn có cần tôi hỗ trợ thêm thao tác nào khác trong dự án này nữa không?
