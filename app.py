import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
import os

# ---------------------------------------------------------
# 1. CAU HINH TRANG STREAMLIT (KHONG DUNG ICON/EMOJI)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Chan Doan Benh La Ca Chua AI",
    page_icon=None,
    layout="wide"
)

# ---------------------------------------------------------
# 2. DANH SACH 34 TINH THANH SAU SAP NHAP (TU 01/07/2025)
# ---------------------------------------------------------
PROVINCES_34 = [
    "TP Ha Noi",
    "TP Ho Chi Minh",
    "TP Da Nang",
    "TP Hai Phong",
    "TP Can Tho",
    "TP Hue",
    "Tinh An Giang",
    "Tinh Bac Ninh",
    "Tinh Cao Bang",
    "Tinh Ca Mau",
    "Tinh Dak Lak",
    "Tinh Dien Bien",
    "Tinh Dong Nai",
    "Tinh Dong Thap",
    "Tinh Gia Lai",
    "Tinh Ha Tinh",
    "Tinh Hung Yen",
    "Tinh Khanh Hoa",
    "Tinh Lai Chau",
    "Tinh Lam Dong",
    "Tinh Lang Son",
    "Tinh Lao Cai",
    "Tinh Nghe An",
    "Tinh Ninh Binh",
    "Tinh Phu Tho",
    "Tinh Quang Ngai",
    "Tinh Quang Ninh",
    "Tinh Quang Tri",
    "Tinh Son La",
    "Tinh Tay Ninh",
    "Tinh Thai Nguyen",
    "Tinh Thanh Hoa",
    "Tinh Tuyen Quang",
    "Tinh Vinh Long"
]

# ---------------------------------------------------------
# 3. CO SO DU LIEU 11 LOP BENH VA BANCHAT BENH HOC
# ---------------------------------------------------------
DISEASE_DATABASE = {
    "Bacterial_spot": {
        "vn_name": "Benh Dom Vi Khuan (Bacterial Spot)",
        "type": "Vi khuan",
        "symptoms": "Xuat hien cac dom nho mau nau den, mong nuoc tren be mat la. La vang va rung duoi goc.",
        "remedy": "Phun thuoc goc dong (Copper Hydroxide hoac Kasugamycin). Tia bot la gia va giu vuon thong thoang."
    },
    "Early_blight": {
        "vn_name": "Benh Dom Vong / Chay La Som (Early Blight)",
        "type": "Nam",
        "symptoms": "Vet benh hinh tron co cac vong dong tam mau nau den, la bi vang xung quanh dom.",
        "remedy": "Phun thuoc goc Mancozeb, Chlorothalonil hoac Difenoconazole. Luan canh cay trong."
    },
    "Late_blight": {
        "vn_name": "Benh Suong Mai / Chay La Muon (Late Blight)",
        "type": "Nam",
        "symptoms": "Vet dom mau xam xanh mong nuoc, phat trien nhanh lam chay kho toan bo la va canu.",
        "remedy": "Phun ngay Metalaxyl, Dimethomorph hoac Ridomil Gold. Ngung tuoi nuoc len la."
    },
    "Leaf_mold": {
        "vn_name": "Benh Moc La (Leaf Mold)",
        "type": "Nam",
        "symptoms": "Mat tren la co dom vang nhat, mat duoi xuat hien lop moc mau xam xin hoac nau nhat.",
        "remedy": "Phun thuoc goc Dong hoac Carbendazim. Giam do am va tang cuong thong gio."
    },
    "Septoria_leaf_spot": {
        "vn_name": "Benh Dom La Septoria (Septoria Leaf Spot)",
        "type": "Nam",
        "symptoms": "Cac dom tron nho mau xam nhat o giua, vien nau den xuat hien nhieu o cac la goc.",
        "remedy": "Phun Azoxystrobin hoac Difenoconazole. Thu gom va tieu huy la benh rung duoi goc."
    },
    "Spider_mites": {
        "vn_name": "Nhen Do Hai La (Two-Spotted Spider Mites)",
        "type": "Con trung hai",
        "symptoms": "La xuat hien cac cham nho lom dom mau vang, mat duoi la co mang to mong.",
        "remedy": "Phun thuoc dac tri nhen nhu Abamectin, Hexythiazox hoac Propargite. Tang do am vuon."
    },
    "Target_Spot": {
        "vn_name": "Benh Dom Muc Tieu (Target Spot)",
        "type": "Nam",
        "symptoms": "Vet benh hinh tron mau nau voi tam sang hon, tao hinh dang giong bia ban.",
        "remedy": "Su dung thuoc tru nam chua Chlorothalonil hoac Pyraclostrobin."
    },
    "Yellow_Leaf_Curl_Virus": {
        "vn_name": "Benh Xoan La Vang Do Virus (Yellow Leaf Curl Virus)",
        "type": "Virus",
        "symptoms": "La bi xoan ngua len tren, phien la nho lai, mau vang chanh, cay lun coi coc va khong ra qua.",
        "remedy": "Khong the chua khoi bang thuoc. Can diet bo phan trang (con trung truyen benh) va nhieu bo cay benh."
    },
    "Mosaic_virus": {
        "vn_name": "Benh Kham La Do Virus (Mosaic Virus)",
        "symptoms": "La loang lo cac vet mau xanh dam va xanh nhat xen ke, la bi bien dang va nhan nheo.",
        "remedy": "Nho bo cay benh de tranh lây lan. Khu trung dung cu cat tia va diet rep muoi truyen benh."
    },
    "Powdery_mildew": {
        "vn_name": "Benh Phan Trang (Powdery Mildew)",
        "type": "Nam",
        "symptoms": "Lop bot trang nhu phan bao phu tren be mat la, lam la kho xo va rung.",
        "remedy": "Phun Sulfur (Luu huynh), Hexaconazole hoac Myclobutanil."
    },
    "Healthy": {
        "vn_name": "La Khoe Manh (Healthy)",
        "symptoms": "La xanh tuoi, khong co dau hieu bi nam, vi khuan hay sau benh tan cong.",
        "remedy": "Tiep tuc cham soc, tuoi nuoc va bon phan dinh ky can doi N-P-K."
    }
}

# Chuand hoa bang tra cuu
NORMALIZED_DB = {k.lower().replace("_", "").replace(" ", "").replace("-", ""): v for k, v in DISEASE_DATABASE.items()}

# ---------------------------------------------------------
# 4. HAM DANH GIA TINH TRANG VA DU DUS LOI KHUYEN TUY BIEN
# ---------------------------------------------------------
def generate_tailored_advice(pred_key, weather_condition, leaf_status, temperature):
    """
    Tong hop phan tich giua:
    - Loai benh AI nhan dien
    - Thoi tiet local (Mat me / Nong oi / Mua do am cao)
    - Tinh trang bieu hien cua la (Binh thuong / Heo / Chay / Xoan)
    - Nhiet do moi truong
    """
    info = NORMALIZED_DB.get(pred_key, {
        "vn_name": "Khong xac dinh",
        "type": "Khong ro",
        "symptoms": "Chua co thong tin mo ta chi tiet.",
        "remedy": "Tham khao y kien chuyen gia nong nghiep dia phuong."
    })

    disease_type = info["type"]
    status_report = []
    advice_list = []

    # Truong hop 1: La khoe manh
    if pred_key == "healthy":
        if "Heo" in leaf_status and ("Mat me" in weather_condition or temperature <= 28):
            status_report.append("Tình trang: La bi heo nhe do thieu nuoc hoac thieu am tam thoi.")
            status_report.append("Kha nang phuc hoi: CAO (100%). Do thoi tiet dang mat me, cay se phuc hoi nhanh chong sau khi duoc cap nuoc.")
            advice_list.append("Tuoi goc bo sung vao buoi sang som hoac chieu mat. Khong can phun thuoc hoa hoc.")
        elif "Heo" in leaf_status and ("Nong" in weather_condition or temperature > 32):
            status_report.append("Tình trang: La bi mat nuoc cap tinh do nhiet do cao gieo giac.")
            status_report.append("Kha nang phuc hoi: TRUNG BINH. Can che mat kip thoi de tranh chay chot.")
            advice_list.append("Che luoi giam nang va tuoi giu am goc, tranh tuoi nuoc len la khi dang nang gat.")
        else:
            status_report.append("Tình trang: La ca chua sinh truong tot, khong phat hien mam benh nguy hien.")
            status_report.append("Kha nang phuc hoi: Hoan hao.")
            advice_list.append(info["remedy"])
        return status_report, advice_list

    # Truong hop 2: Benh do Virus (Yellow Leaf Curl, Mosaic)
    if disease_type == "Virus":
        status_report.append(f"Tình trang: Cay da bi nhiem {info['vn_name']}.")
        status_report.append("Kha nang phuc hoi: KHONG THE PHUC HOI bang dieu chinh thoi tiet hay tuoi nuoc.")
        advice_list.append(info["remedy"])
        advice_list.append("Nhổ bỏ va tieu huy cay benh nang de ngan chan virus lây lan sang cac cay khac trong vuon.")
        return status_report, advice_list

    # Truong hop 3: Benh do Nam hoac Vi khuan
    if "Heo" in leaf_status or "Chay" in leaf_status:
        if "Mat me" in weather_condition or temperature <= 26:
            status_report.append(f"Tình trang: Phat hien {info['vn_name']} dang thien giam hoac o giai doan dau.")
            status_report.append("Kha nang phuc hoi: KHA NANG PHUC HOI CAO. Thoi tiet mat me giup cay giam mat suc, khi phun thuoc tri nam/vi khuan cay se nhanh ra mầm moi.")
            advice_list.append("Cat bo phan la bi benh nang va phun thuoc dac tri theo huong dan.")
            advice_list.append("Bon them vi luong hoac phan bón la de kich thich cay ra chot moi.")
        elif "Mua" in weather_condition or "Do am" in weather_condition:
            status_report.append(f"Tình trang: {info['vn_name']} dang co nguy co bung phat manh do do am cao.")
            status_report.append("Kha nang phuc hoi: TRUNG BINH. Nguy co lay lan nhanh neu khong xu ly ngay.")
            advice_list.append("Ngung ngay viec tuoi nuoc len la. Phun thuoc phong va tri ngay khi tạnh mưa.")
            advice_list.append("Thoi thong ranh thoat nuoc trong vuon de tranh ngap ung goc.")
        else:
            status_report.append(f"Tình trang: {info['vn_name']} ket hop voi nhiet do cao lam la kho nhanh hon.")
            status_report.append("Kha nang phuc hoi: TRUNG BINH.")
            advice_list.append("Phun thuoc tri benh vao chieu mat va ket hop che nang nhe cho vuon.")
    else:
        status_report.append(f"Tình trang: Phat hien {info['vn_name']}.")
        status_report.append("Kha nang phuc hoi: TOT neu can thiep kip thoi.")
        advice_list.append(info["remedy"])

    return status_report, advice_list

# ---------------------------------------------------------
# 5. LOAD MODEL TU GITHUB RELEASES
# ---------------------------------------------------------
MODEL_URL = "https://github.com/Tranthihoaithuong2009/tomato-disease-ai/releases/download/v1.0/tomato_model_best.pth"

@st.cache_resource
def load_model():
    model_path = "tomato_model_best.pth"
    if not os.path.exists(model_path):
        with st.spinner("Dang tai mo hinh AI tu GitHub..."):
            res = requests.get(MODEL_URL)
            with open(model_path, "wb") as f:
                f.write(res.content)
    
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
# 6. GIAO DIEN CHINH CUA UNG DUNG
# ---------------------------------------------------------
def main():
    st.title("Chan Doan Benh La Ca Chua Bang AI")
    st.write("Ung dung phan tich hinh anh la ca chua, ket hop voi tinh trang thoi tiet va bieu hien cua la de dua ra chan doan va loi khuyen phuc hoi chinh xac.")

    # Thanh ben trai: Chi chon Tinh/Thanh pho va thong tin moi truong
    st.sidebar.title("Vi Tri Va Moi Truong")
    
    # 1. Chi chon Tinh / Thanh pho (34 Tinh Thanh sau sap nhap)
    selected_province = st.sidebar.selectbox("Chon Tinh/Thanh pho (34 tinh thanh):", PROVINCES_34)
    
    # 2. Cac thong so thoi tiet va bieu hien thuc te cua la
    st.sidebar.markdown("---")
    st.sidebar.subheader("Thong Tin Moi Truong Va Bieu Hien La")
    
    weather_condition = st.sidebar.selectbox(
        "Thoi tiet hien tai:",
        ["Mat me / On hoa", "Nong oi / Nang gat", "Mua nhieu / Do am cao"]
    )
    
    temperature = st.sidebar.slider("Nhiet do moi truong (C):", min_value=15, max_value=42, value=25)
    
    leaf_status = st.sidebar.selectbox(
        "Bieu hien ngoai quan cua la:",
        ["La binh thuong", "La bi heo / Ru ngua", "La bi chay xam / Vang dom", "La bi xoan / Bien dang"]
    )

    st.sidebar.markdown("---")
    st.sidebar.write(f"Khu vuc: {selected_province}")
    st.sidebar.write(f"Thoi tiet: {weather_condition} ({temperature}C)")
    st.sidebar.write(f"Bieu hien la: {leaf_status}")

    # Tai mo hinh AI
    try:
        model, class_names = load_model()
    except Exception as e:
        st.error(f"Loi khi tai mo hinh AI: {e}")
        return

    # Tai anh len
    uploaded_file = st.file_uploader("Chon anh la ca chua de kiem tra (JPG, PNG, JPEG)...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Hinh anh la da tai len", use_container_width=True)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image).unsqueeze(0)

        with st.spinner("AI dang phan tich hinh anh..."):
            with torch.no_grad():
                outputs = model(img_tensor)
                # SU DUNG dim=1 DE CHINH XAC Softmax VA Max
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_raw = class_names[predicted_idx.item()]
        conf_percent = confidence.item() * 100

        pred_key = predicted_raw.lower().replace("_", "").replace(" ", "").replace("-", "")
        info = NORMALIZED_DB.get(pred_key, {
            "vn_name": predicted_raw,
            "type": "Khong ro",
            "symptoms": "Chua co thong tin mo ta chi tiet.",
            "remedy": "Tham khao y kien chuyen gia nong nghiep dia phuong."
        })

        status_reports, tailored_advices = generate_tailored_advice(pred_key, weather_condition, leaf_status, temperature)

        with col2:
            st.subheader("Ket Qua Chan Doan AI")
            if pred_key == "healthy":
                st.success(f"Ket qua: {info['vn_name']}")
            else:
                st.error(f"Ket qua: {info['vn_name']}")
            
            st.metric(label="Do tin cay cua AI", value=f"{conf_percent:.2f}%")
            st.write(f"Tac nhan gay benh: {info['type']}")
            st.write(f"Khu vuc ghi nhan: {selected_province}")

            st.markdown("---")
            st.markdown("### Trieu Chung Dac Thuong:")
            st.write(info["symptoms"])

            st.markdown("---")
            st.markdown("### Bieu Hien & Kha Nang Phuc Hoi:")
            for report in status_reports:
                st.write(f"- {report}")

            st.markdown("### Huong Dan Khac Phuc Tuong Ung:")
            for adv in tailored_advices:
                st.info(adv)

if __name__ == "__main__":
    main()
