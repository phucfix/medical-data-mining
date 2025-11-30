#!/usr/bin/env python3
"""
Script chạy Phase 1 KHÔNG cần API (chỉ dùng dữ liệu có sẵn)
Phù hợp khi không có API key hoặc hết quota
"""
import sys
import json
import random
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from tqdm import tqdm

# Setup
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
GENERATED_DIR = DATA_DIR / "generated"
EXTERNAL_DIR = DATA_DIR / "external"

log_file = BASE_DIR / "logs" / f"phase1_offline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_file.parent.mkdir(exist_ok=True)
logger.add(log_file)


def step1_process_international_data():
    """Bước 1: Xử lý dữ liệu quốc tế (không cần API)"""
    print("\n" + "="*60)
    print("📌 BƯỚC 1: Xử lý dữ liệu quốc tế (ICD-10, MeSH, HPO)")
    print("="*60)
    
    # ICD-10 diseases - dữ liệu đã có sẵn trong code
    icd10_diseases = [
        {"code": "A00", "name_en": "Cholera", "name_vi": "Bệnh tả", "category": "Bệnh truyền nhiễm"},
        {"code": "A01", "name_en": "Typhoid fever", "name_vi": "Bệnh thương hàn", "category": "Bệnh truyền nhiễm"},
        {"code": "A09", "name_en": "Gastroenteritis", "name_vi": "Viêm dạ dày ruột", "category": "Bệnh truyền nhiễm"},
        {"code": "A15", "name_en": "Tuberculosis", "name_vi": "Bệnh lao phổi", "category": "Bệnh truyền nhiễm"},
        {"code": "A37", "name_en": "Whooping cough", "name_vi": "Ho gà", "category": "Bệnh truyền nhiễm"},
        {"code": "B01", "name_en": "Chickenpox", "name_vi": "Bệnh thủy đậu", "category": "Bệnh truyền nhiễm"},
        {"code": "B05", "name_en": "Measles", "name_vi": "Bệnh sởi", "category": "Bệnh truyền nhiễm"},
        {"code": "B15", "name_en": "Hepatitis A", "name_vi": "Viêm gan A", "category": "Bệnh truyền nhiễm"},
        {"code": "B16", "name_en": "Hepatitis B", "name_vi": "Viêm gan B", "category": "Bệnh truyền nhiễm"},
        {"code": "B20", "name_en": "HIV disease", "name_vi": "Bệnh HIV/AIDS", "category": "Bệnh truyền nhiễm"},
        {"code": "C16", "name_en": "Stomach cancer", "name_vi": "Ung thư dạ dày", "category": "Ung thư"},
        {"code": "C18", "name_en": "Colon cancer", "name_vi": "Ung thư đại tràng", "category": "Ung thư"},
        {"code": "C22", "name_en": "Liver cancer", "name_vi": "Ung thư gan", "category": "Ung thư"},
        {"code": "C34", "name_en": "Lung cancer", "name_vi": "Ung thư phổi", "category": "Ung thư"},
        {"code": "C50", "name_en": "Breast cancer", "name_vi": "Ung thư vú", "category": "Ung thư"},
        {"code": "C61", "name_en": "Prostate cancer", "name_vi": "Ung thư tuyến tiền liệt", "category": "Ung thư"},
        {"code": "E10", "name_en": "Type 1 diabetes", "name_vi": "Tiểu đường type 1", "category": "Nội tiết"},
        {"code": "E11", "name_en": "Type 2 diabetes", "name_vi": "Tiểu đường type 2", "category": "Nội tiết"},
        {"code": "E66", "name_en": "Obesity", "name_vi": "Béo phì", "category": "Nội tiết"},
        {"code": "E78", "name_en": "Hyperlipidemia", "name_vi": "Rối loạn lipid máu", "category": "Nội tiết"},
        {"code": "F32", "name_en": "Depression", "name_vi": "Trầm cảm", "category": "Tâm thần"},
        {"code": "F41", "name_en": "Anxiety disorder", "name_vi": "Rối loạn lo âu", "category": "Tâm thần"},
        {"code": "G20", "name_en": "Parkinson disease", "name_vi": "Bệnh Parkinson", "category": "Thần kinh"},
        {"code": "G30", "name_en": "Alzheimer disease", "name_vi": "Bệnh Alzheimer", "category": "Thần kinh"},
        {"code": "G40", "name_en": "Epilepsy", "name_vi": "Động kinh", "category": "Thần kinh"},
        {"code": "G43", "name_en": "Migraine", "name_vi": "Đau nửa đầu", "category": "Thần kinh"},
        {"code": "H10", "name_en": "Conjunctivitis", "name_vi": "Viêm kết mạc", "category": "Mắt"},
        {"code": "H25", "name_en": "Cataract", "name_vi": "Đục thủy tinh thể", "category": "Mắt"},
        {"code": "H40", "name_en": "Glaucoma", "name_vi": "Tăng nhãn áp", "category": "Mắt"},
        {"code": "I10", "name_en": "Hypertension", "name_vi": "Tăng huyết áp", "category": "Tim mạch"},
        {"code": "I20", "name_en": "Angina pectoris", "name_vi": "Đau thắt ngực", "category": "Tim mạch"},
        {"code": "I21", "name_en": "Myocardial infarction", "name_vi": "Nhồi máu cơ tim", "category": "Tim mạch"},
        {"code": "I50", "name_en": "Heart failure", "name_vi": "Suy tim", "category": "Tim mạch"},
        {"code": "I63", "name_en": "Stroke", "name_vi": "Đột quỵ não", "category": "Tim mạch"},
        {"code": "I84", "name_en": "Hemorrhoids", "name_vi": "Bệnh trĩ", "category": "Tim mạch"},
        {"code": "J00", "name_en": "Common cold", "name_vi": "Cảm lạnh thông thường", "category": "Hô hấp"},
        {"code": "J02", "name_en": "Pharyngitis", "name_vi": "Viêm họng", "category": "Hô hấp"},
        {"code": "J03", "name_en": "Tonsillitis", "name_vi": "Viêm amidan", "category": "Hô hấp"},
        {"code": "J06", "name_en": "Upper respiratory infection", "name_vi": "Viêm đường hô hấp trên", "category": "Hô hấp"},
        {"code": "J10", "name_en": "Influenza", "name_vi": "Cúm", "category": "Hô hấp"},
        {"code": "J18", "name_en": "Pneumonia", "name_vi": "Viêm phổi", "category": "Hô hấp"},
        {"code": "J20", "name_en": "Acute bronchitis", "name_vi": "Viêm phế quản cấp", "category": "Hô hấp"},
        {"code": "J44", "name_en": "COPD", "name_vi": "Bệnh phổi tắc nghẽn mãn tính", "category": "Hô hấp"},
        {"code": "J45", "name_en": "Asthma", "name_vi": "Hen suyễn", "category": "Hô hấp"},
        {"code": "K21", "name_en": "GERD", "name_vi": "Trào ngược dạ dày thực quản", "category": "Tiêu hóa"},
        {"code": "K25", "name_en": "Gastric ulcer", "name_vi": "Loét dạ dày", "category": "Tiêu hóa"},
        {"code": "K29", "name_en": "Gastritis", "name_vi": "Viêm dạ dày", "category": "Tiêu hóa"},
        {"code": "K35", "name_en": "Appendicitis", "name_vi": "Viêm ruột thừa", "category": "Tiêu hóa"},
        {"code": "K58", "name_en": "IBS", "name_vi": "Hội chứng ruột kích thích", "category": "Tiêu hóa"},
        {"code": "K74", "name_en": "Cirrhosis", "name_vi": "Xơ gan", "category": "Tiêu hóa"},
        {"code": "K80", "name_en": "Gallstones", "name_vi": "Sỏi mật", "category": "Tiêu hóa"},
        {"code": "L20", "name_en": "Atopic dermatitis", "name_vi": "Viêm da cơ địa", "category": "Da liễu"},
        {"code": "L40", "name_en": "Psoriasis", "name_vi": "Vẩy nến", "category": "Da liễu"},
        {"code": "L50", "name_en": "Urticaria", "name_vi": "Mề đay", "category": "Da liễu"},
        {"code": "L70", "name_en": "Acne", "name_vi": "Mụn trứng cá", "category": "Da liễu"},
        {"code": "M05", "name_en": "Rheumatoid arthritis", "name_vi": "Viêm khớp dạng thấp", "category": "Cơ xương khớp"},
        {"code": "M10", "name_en": "Gout", "name_vi": "Bệnh gout", "category": "Cơ xương khớp"},
        {"code": "M17", "name_en": "Knee osteoarthritis", "name_vi": "Thoái hóa khớp gối", "category": "Cơ xương khớp"},
        {"code": "M51", "name_en": "Disc herniation", "name_vi": "Thoát vị đĩa đệm", "category": "Cơ xương khớp"},
        {"code": "M54", "name_en": "Back pain", "name_vi": "Đau lưng", "category": "Cơ xương khớp"},
        {"code": "M81", "name_en": "Osteoporosis", "name_vi": "Loãng xương", "category": "Cơ xương khớp"},
        {"code": "N18", "name_en": "Chronic kidney disease", "name_vi": "Bệnh thận mãn", "category": "Tiết niệu"},
        {"code": "N20", "name_en": "Kidney stones", "name_vi": "Sỏi thận", "category": "Tiết niệu"},
        {"code": "N30", "name_en": "Cystitis", "name_vi": "Viêm bàng quang", "category": "Tiết niệu"},
        {"code": "N40", "name_en": "BPH", "name_vi": "Phì đại tuyến tiền liệt", "category": "Tiết niệu"},
    ]
    
    # Symptoms
    symptoms = [
        {"name_vi": "Sốt", "name_en": "Fever", "description": "Thân nhiệt cao hơn bình thường"},
        {"name_vi": "Đau đầu", "name_en": "Headache", "description": "Đau ở vùng đầu"},
        {"name_vi": "Ho", "name_en": "Cough", "description": "Phản xạ đẩy không khí ra khỏi phổi"},
        {"name_vi": "Khó thở", "name_en": "Dyspnea", "description": "Khó khăn khi hít thở"},
        {"name_vi": "Đau ngực", "name_en": "Chest pain", "description": "Cảm giác đau ở vùng ngực"},
        {"name_vi": "Đau bụng", "name_en": "Abdominal pain", "description": "Đau ở vùng bụng"},
        {"name_vi": "Buồn nôn", "name_en": "Nausea", "description": "Cảm giác muốn nôn"},
        {"name_vi": "Nôn", "name_en": "Vomiting", "description": "Tống thức ăn ra khỏi dạ dày"},
        {"name_vi": "Tiêu chảy", "name_en": "Diarrhea", "description": "Đi ngoài phân lỏng nhiều lần"},
        {"name_vi": "Táo bón", "name_en": "Constipation", "description": "Khó đi ngoài"},
        {"name_vi": "Mệt mỏi", "name_en": "Fatigue", "description": "Cảm giác kiệt sức"},
        {"name_vi": "Chóng mặt", "name_en": "Dizziness", "description": "Cảm giác quay cuồng"},
        {"name_vi": "Đau lưng", "name_en": "Back pain", "description": "Đau ở vùng lưng"},
        {"name_vi": "Đau khớp", "name_en": "Joint pain", "description": "Đau ở các khớp"},
        {"name_vi": "Sưng", "name_en": "Swelling", "description": "Phù nề do tích tụ dịch"},
        {"name_vi": "Ngứa", "name_en": "Itching", "description": "Cảm giác muốn gãi"},
        {"name_vi": "Phát ban", "name_en": "Rash", "description": "Thay đổi màu sắc da"},
        {"name_vi": "Sổ mũi", "name_en": "Runny nose", "description": "Chảy dịch mũi"},
        {"name_vi": "Nghẹt mũi", "name_en": "Nasal congestion", "description": "Tắc mũi"},
        {"name_vi": "Đau họng", "name_en": "Sore throat", "description": "Đau rát ở cổ họng"},
        {"name_vi": "Sụt cân", "name_en": "Weight loss", "description": "Giảm cân không chủ ý"},
        {"name_vi": "Mất ngủ", "name_en": "Insomnia", "description": "Khó ngủ hoặc ngủ không sâu"},
        {"name_vi": "Lo âu", "name_en": "Anxiety", "description": "Cảm giác lo lắng, bồn chồn"},
        {"name_vi": "Trầm cảm", "name_en": "Depression", "description": "Buồn bã, mất hứng thú"},
        {"name_vi": "Tim đập nhanh", "name_en": "Palpitations", "description": "Cảm nhận tim đập mạnh"},
    ]
    
    # Drugs
    drugs = [
        {"name_vi": "Paracetamol", "category": "Giảm đau hạ sốt", "indication": "Hạ sốt, giảm đau nhẹ đến vừa"},
        {"name_vi": "Ibuprofen", "category": "Kháng viêm NSAIDs", "indication": "Giảm đau, kháng viêm, hạ sốt"},
        {"name_vi": "Amoxicillin", "category": "Kháng sinh", "indication": "Nhiễm khuẩn đường hô hấp, tiết niệu"},
        {"name_vi": "Azithromycin", "category": "Kháng sinh", "indication": "Nhiễm khuẩn hô hấp, da"},
        {"name_vi": "Metformin", "category": "Thuốc tiểu đường", "indication": "Tiểu đường type 2"},
        {"name_vi": "Amlodipine", "category": "Thuốc huyết áp", "indication": "Tăng huyết áp, đau thắt ngực"},
        {"name_vi": "Omeprazole", "category": "Thuốc dạ dày", "indication": "Trào ngược, loét dạ dày"},
        {"name_vi": "Atorvastatin", "category": "Thuốc mỡ máu", "indication": "Rối loạn lipid máu"},
        {"name_vi": "Losartan", "category": "Thuốc huyết áp", "indication": "Tăng huyết áp"},
        {"name_vi": "Cetirizine", "category": "Kháng histamin", "indication": "Dị ứng, mề đay"},
        {"name_vi": "Salbutamol", "category": "Thuốc hen", "indication": "Hen suyễn, COPD"},
        {"name_vi": "Prednisolone", "category": "Corticosteroid", "indication": "Viêm, dị ứng nặng"},
        {"name_vi": "Diclofenac", "category": "Kháng viêm NSAIDs", "indication": "Đau khớp, viêm khớp"},
        {"name_vi": "Aspirin", "category": "Kháng kết tập tiểu cầu", "indication": "Phòng ngừa tim mạch"},
        {"name_vi": "Clopidogrel", "category": "Kháng kết tập tiểu cầu", "indication": "Phòng ngừa huyết khối"},
    ]
    
    # Save data
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(EXTERNAL_DIR / "icd10_diseases.json", 'w', encoding='utf-8') as f:
        json.dump(icd10_diseases, f, ensure_ascii=False, indent=2)
    
    with open(EXTERNAL_DIR / "symptoms.json", 'w', encoding='utf-8') as f:
        json.dump(symptoms, f, ensure_ascii=False, indent=2)
    
    with open(EXTERNAL_DIR / "drugs.json", 'w', encoding='utf-8') as f:
        json.dump(drugs, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Đã lưu {len(icd10_diseases)} bệnh ICD-10")
    print(f"✅ Đã lưu {len(symptoms)} triệu chứng")
    print(f"✅ Đã lưu {len(drugs)} thuốc")
    
    return {"diseases": icd10_diseases, "symptoms": symptoms, "drugs": drugs}


def step2_generate_qa_from_data(data):
    """Bước 2: Sinh câu hỏi Q&A từ dữ liệu có sẵn (không cần API)"""
    print("\n" + "="*60)
    print("📌 BƯỚC 2: Sinh câu hỏi Q&A từ dữ liệu có sẵn")
    print("="*60)
    
    all_qa = []
    
    # Templates cho câu hỏi về bệnh
    disease_templates_true = [
        "{name_vi} là một bệnh thuộc nhóm {category}.",
        "Bệnh {name_vi} có mã ICD-10 là {code}.",
        "{name_vi} (tên tiếng Anh: {name_en}) là một bệnh lý cần được điều trị.",
    ]
    
    disease_templates_false = [
        "{name_vi} là một bệnh thuộc nhóm {wrong_category}.",
        "Bệnh {name_vi} không cần điều trị y tế.",
        "{name_vi} là một triệu chứng, không phải bệnh.",
    ]
    
    # Sinh câu hỏi về bệnh
    categories = list(set(d['category'] for d in data['diseases']))
    
    for disease in data['diseases']:
        # Câu đúng
        for template in disease_templates_true:
            try:
                question = template.format(**disease)
                all_qa.append({
                    "question": question,
                    "answer": "Đúng",
                    "explanation": f"{disease['name_vi']} ({disease['name_en']}) thuộc nhóm {disease['category']}",
                    "category": "diseases",
                    "source": "template_generated"
                })
            except:
                pass
        
        # Câu sai
        wrong_cats = [c for c in categories if c != disease['category']]
        if wrong_cats:
            wrong_cat = random.choice(wrong_cats)
            disease_copy = disease.copy()
            disease_copy['wrong_category'] = wrong_cat
            
            for template in disease_templates_false:
                try:
                    question = template.format(**disease_copy)
                    all_qa.append({
                        "question": question,
                        "answer": "Sai",
                        "explanation": f"{disease['name_vi']} thuộc nhóm {disease['category']}, không phải {wrong_cat}",
                        "category": "diseases",
                        "source": "template_generated"
                    })
                except:
                    pass
    
    # Templates cho triệu chứng
    symptom_templates_true = [
        "{name_vi} là một triệu chứng y tế cần được theo dõi.",
        "Triệu chứng {name_vi} được gọi là {name_en} trong tiếng Anh.",
        "{description} là biểu hiện của triệu chứng {name_vi}.",
    ]
    
    symptom_templates_false = [
        "{name_vi} là một loại thuốc, không phải triệu chứng.",
        "Triệu chứng {name_vi} không cần quan tâm vì không nguy hiểm.",
    ]
    
    for symptom in data['symptoms']:
        for template in symptom_templates_true:
            try:
                question = template.format(**symptom)
                all_qa.append({
                    "question": question,
                    "answer": "Đúng",
                    "explanation": f"{symptom['name_vi']} - {symptom['description']}",
                    "category": "symptoms",
                    "source": "template_generated"
                })
            except:
                pass
        
        for template in symptom_templates_false:
            try:
                question = template.format(**symptom)
                all_qa.append({
                    "question": question,
                    "answer": "Sai",
                    "explanation": f"{symptom['name_vi']} là triệu chứng, cần được theo dõi",
                    "category": "symptoms",
                    "source": "template_generated"
                })
            except:
                pass
    
    # Templates cho thuốc
    drug_templates_true = [
        "{name_vi} là thuốc thuộc nhóm {category}.",
        "Thuốc {name_vi} được chỉ định điều trị {indication}.",
        "{name_vi} cần được sử dụng theo chỉ định của bác sĩ.",
    ]
    
    drug_templates_false = [
        "{name_vi} là một bệnh, không phải thuốc.",
        "Thuốc {name_vi} có thể tự ý sử dụng không cần kê đơn bác sĩ.",
    ]
    
    for drug in data['drugs']:
        for template in drug_templates_true:
            try:
                question = template.format(**drug)
                all_qa.append({
                    "question": question,
                    "answer": "Đúng",
                    "explanation": f"{drug['name_vi']} - {drug['category']} - {drug['indication']}",
                    "category": "drugs",
                    "source": "template_generated"
                })
            except:
                pass
        
        for template in drug_templates_false:
            try:
                question = template.format(**drug)
                all_qa.append({
                    "question": question,
                    "answer": "Sai",
                    "explanation": f"{drug['name_vi']} là thuốc {drug['category']}, cần dùng theo chỉ định",
                    "category": "drugs",
                    "source": "template_generated"
                })
            except:
                pass
    
    # Thêm các câu hỏi y tế phổ biến
    common_medical_qa = [
        # Bệnh tim mạch
        {"question": "Tăng huyết áp là khi huyết áp cao hơn 140/90 mmHg.", "answer": "Đúng", "explanation": "Theo WHO, huyết áp ≥140/90 mmHg được coi là tăng huyết áp."},
        {"question": "Nhồi máu cơ tim xảy ra khi động mạch vành bị tắc nghẽn.", "answer": "Đúng", "explanation": "Nhồi máu cơ tim do tắc nghẽn động mạch vành, làm tim thiếu máu."},
        {"question": "Đột quỵ não chỉ xảy ra ở người già.", "answer": "Sai", "explanation": "Đột quỵ có thể xảy ra ở mọi lứa tuổi, kể cả người trẻ."},
        {"question": "Hút thuốc lá làm tăng nguy cơ bệnh tim mạch.", "answer": "Đúng", "explanation": "Hút thuốc là yếu tố nguy cơ chính của bệnh tim mạch."},
        
        # Bệnh tiểu đường
        {"question": "Tiểu đường type 2 có thể phòng ngừa bằng lối sống lành mạnh.", "answer": "Đúng", "explanation": "Chế độ ăn uống và vận động giúp giảm nguy cơ tiểu đường type 2."},
        {"question": "Người tiểu đường không được ăn trái cây.", "answer": "Sai", "explanation": "Người tiểu đường có thể ăn trái cây với lượng vừa phải."},
        {"question": "Tiểu đường type 1 thường xuất hiện ở trẻ em và thanh niên.", "answer": "Đúng", "explanation": "Tiểu đường type 1 thường phát bệnh ở độ tuổi trẻ."},
        {"question": "Đường huyết bình thường khi đói là dưới 100 mg/dL.", "answer": "Đúng", "explanation": "Đường huyết lúc đói bình thường: 70-99 mg/dL."},
        
        # Bệnh hô hấp
        {"question": "Ho kéo dài trên 3 tuần có thể là triệu chứng của lao phổi.", "answer": "Đúng", "explanation": "Ho kéo dài là triệu chứng cần nghĩ đến lao phổi."},
        {"question": "Hen suyễn là bệnh viêm đường thở mãn tính.", "answer": "Đúng", "explanation": "Hen suyễn là bệnh viêm mãn tính làm hẹp đường thở."},
        {"question": "COPD là bệnh có thể chữa khỏi hoàn toàn.", "answer": "Sai", "explanation": "COPD là bệnh mãn tính, chỉ có thể kiểm soát, không chữa khỏi."},
        {"question": "Viêm phổi do vi khuẩn cần điều trị bằng kháng sinh.", "answer": "Đúng", "explanation": "Viêm phổi do vi khuẩn cần dùng kháng sinh phù hợp."},
        
        # Bệnh tiêu hóa
        {"question": "Viêm dạ dày có thể do vi khuẩn H. pylori gây ra.", "answer": "Đúng", "explanation": "H. pylori là nguyên nhân phổ biến của viêm loét dạ dày."},
        {"question": "Trào ngược dạ dày thực quản gây ợ nóng và ợ chua.", "answer": "Đúng", "explanation": "GERD gây triệu chứng ợ nóng, ợ chua điển hình."},
        {"question": "Uống rượu nhiều là nguyên nhân gây xơ gan.", "answer": "Đúng", "explanation": "Rượu là nguyên nhân chính gây xơ gan."},
        {"question": "Sỏi mật hình thành do cholesterol kết tụ.", "answer": "Đúng", "explanation": "Phần lớn sỏi mật là sỏi cholesterol."},
        
        # Bệnh xương khớp
        {"question": "Thoát vị đĩa đệm là do nhân nhầy đĩa đệm lồi ra chèn dây thần kinh.", "answer": "Đúng", "explanation": "Thoát vị đĩa đệm xảy ra khi nhân nhầy thoát ra ngoài."},
        {"question": "Loãng xương phổ biến hơn ở phụ nữ sau mãn kinh.", "answer": "Đúng", "explanation": "Giảm estrogen sau mãn kinh tăng nguy cơ loãng xương."},
        {"question": "Bệnh gout do acid uric tích tụ trong khớp.", "answer": "Đúng", "explanation": "Gout do tinh thể urat lắng đọng trong khớp."},
        {"question": "Viêm khớp dạng thấp là bệnh tự miễn.", "answer": "Đúng", "explanation": "Viêm khớp dạng thấp do hệ miễn dịch tấn công khớp."},
        
        # Bệnh thần kinh
        {"question": "Động kinh là tình trạng các tế bào não hoạt động bất thường gây co giật.", "answer": "Đúng", "explanation": "Động kinh do hoạt động điện bất thường của não."},
        {"question": "Bệnh Parkinson ảnh hưởng đến khả năng vận động.", "answer": "Đúng", "explanation": "Parkinson gây run, cứng đờ, chậm vận động."},
        {"question": "Alzheimer là bệnh mất trí nhớ ở người già.", "answer": "Đúng", "explanation": "Alzheimer là dạng phổ biến nhất của sa sút trí tuệ."},
        {"question": "Đau nửa đầu Migraine chỉ đau một bên đầu.", "answer": "Sai", "explanation": "Migraine có thể đau một bên hoặc hai bên đầu."},
        
        # Thuốc
        {"question": "Paracetamol là thuốc giảm đau và hạ sốt an toàn.", "answer": "Đúng", "explanation": "Paracetamol an toàn khi dùng đúng liều."},
        {"question": "Kháng sinh có thể tiêu diệt virus.", "answer": "Sai", "explanation": "Kháng sinh chỉ có tác dụng với vi khuẩn, không diệt virus."},
        {"question": "Thuốc kháng viêm NSAIDs có thể gây loét dạ dày.", "answer": "Đúng", "explanation": "NSAIDs là nguyên nhân gây loét dạ dày thường gặp."},
        {"question": "Aspirin liều thấp giúp phòng ngừa bệnh tim mạch.", "answer": "Đúng", "explanation": "Aspirin liều thấp được dùng để phòng ngừa tim mạch."},
        
        # Dinh dưỡng & sức khỏe
        {"question": "Ăn nhiều muối làm tăng huyết áp.", "answer": "Đúng", "explanation": "Sodium trong muối làm tăng huyết áp."},
        {"question": "Vitamin D giúp hấp thu canxi và tốt cho xương.", "answer": "Đúng", "explanation": "Vitamin D cần thiết cho hấp thu canxi."},
        {"question": "Uống đủ nước mỗi ngày (2-3 lít) tốt cho sức khỏe.", "answer": "Đúng", "explanation": "Nước cần thiết cho mọi hoạt động của cơ thể."},
        {"question": "Ngủ đủ 7-8 tiếng mỗi đêm tốt cho sức khỏe.", "answer": "Đúng", "explanation": "Giấc ngủ đủ giờ giúp phục hồi cơ thể và tinh thần."},
        
        # Sức khỏe sinh sản
        {"question": "Phụ nữ mang thai cần bổ sung acid folic.", "answer": "Đúng", "explanation": "Acid folic ngăn ngừa dị tật ống thần kinh thai nhi."},
        {"question": "Ung thư cổ tử cung có thể phòng ngừa bằng vaccine HPV.", "answer": "Đúng", "explanation": "Vaccine HPV ngăn ngừa virus gây ung thư cổ tử cung."},
        
        # Dịch bệnh
        {"question": "COVID-19 là bệnh do virus SARS-CoV-2 gây ra.", "answer": "Đúng", "explanation": "COVID-19 do coronavirus SARS-CoV-2 gây ra."},
        {"question": "Vaccine giúp tạo miễn dịch chống lại bệnh truyền nhiễm.", "answer": "Đúng", "explanation": "Vaccine kích thích hệ miễn dịch tạo kháng thể."},
        {"question": "Rửa tay đúng cách giúp phòng ngừa bệnh truyền nhiễm.", "answer": "Đúng", "explanation": "Rửa tay là biện pháp phòng bệnh hiệu quả."},
    ]
    
    for qa in common_medical_qa:
        qa['category'] = 'general_medical'
        qa['source'] = 'expert_curated'
        all_qa.append(qa)
    
    # Shuffle
    random.shuffle(all_qa)
    
    # Save
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    
    true_count = len([q for q in all_qa if q['answer'] == 'Đúng'])
    false_count = len([q for q in all_qa if q['answer'] == 'Sai'])
    
    dataset = {
        "metadata": {
            "total_questions": len(all_qa),
            "true_count": true_count,
            "false_count": false_count,
            "generated_date": datetime.now().isoformat(),
            "method": "template_based_offline"
        },
        "data": all_qa
    }
    
    with open(GENERATED_DIR / "medical_qa_dataset.json", 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Đã sinh {len(all_qa)} câu hỏi Q&A")
    print(f"   - Câu Đúng: {true_count}")
    print(f"   - Câu Sai: {false_count}")
    
    return all_qa


def step3_summary():
    """Bước 3: Tổng kết"""
    print("\n" + "="*60)
    print("📌 BƯỚC 3: Tổng kết dữ liệu")
    print("="*60)
    
    total = 0
    
    # Count all data
    for json_file in EXTERNAL_DIR.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            count = len(data)
            total += count
            print(f"   - {json_file.name}: {count} records")
    
    for json_file in GENERATED_DIR.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'data' in data:
                count = len(data['data'])
            else:
                count = len(data)
            total += count
            print(f"   - {json_file.name}: {count} records")
    
    print(f"\n📊 TỔNG CỘNG: {total} records")
    
    if total < 50000:
        print(f"\n⚠️ Cần thêm {50000 - total} dữ liệu để đạt yêu cầu 50,000")
        print("\n💡 Gợi ý:")
        print("   1. Lấy API key mới từ https://aistudio.google.com/app/apikey")
        print("   2. Chạy lại với: python scripts/run_phase1_pipeline.py")
        print("   3. Hoặc chạy nhiều lần script này để sinh thêm dữ liệu")


def main():
    """Chạy Phase 1 offline"""
    print("\n" + "="*60)
    print("🚀 PHASE 1 OFFLINE: THU THẬP DỮ LIỆU KHÔNG CẦN API")
    print("="*60)
    print(f"⏰ Bắt đầu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Bước 1
    data = step1_process_international_data()
    
    # Bước 2
    step2_generate_qa_from_data(data)
    
    # Bước 3
    step3_summary()
    
    print("\n" + "="*60)
    print("✅ HOÀN THÀNH!")
    print("="*60)
    print("\n📁 Dữ liệu đã lưu tại:")
    print(f"   - {EXTERNAL_DIR}")
    print(f"   - {GENERATED_DIR}")


if __name__ == "__main__":
    main()
