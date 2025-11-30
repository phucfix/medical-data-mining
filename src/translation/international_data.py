"""
Xử lý và dịch dữ liệu từ các nguồn quốc tế: UMLS, ICD-10, MeSH, HPO
Mục đích: Lấy điểm cộng cho việc sử dụng nguồn tri thức tiếng Anh
"""
import json
import csv
import requests
import time
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from loguru import logger
from deep_translator import GoogleTranslator
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import EXTERNAL_DATA_DIR, GOOGLE_API_KEY


class InternationalDataProcessor:
    """Base class cho xử lý dữ liệu quốc tế"""
    
    def __init__(self, source_name: str):
        self.source_name = source_name
        self.translator = GoogleTranslator(source='en', target='vi')
        self.output_dir = EXTERNAL_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache translations để tránh translate lại
        self.translation_cache = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load translation cache từ file"""
        cache_file = self.output_dir / f"{self.source_name}_translation_cache.json"
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                self.translation_cache = json.load(f)
    
    def _save_cache(self):
        """Lưu translation cache"""
        cache_file = self.output_dir / f"{self.source_name}_translation_cache.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.translation_cache, f, ensure_ascii=False, indent=2)
    
    def translate_text(self, text: str) -> str:
        """Dịch text từ tiếng Anh sang tiếng Việt"""
        if not text or len(text.strip()) == 0:
            return ""
        
        # Kiểm tra cache
        if text in self.translation_cache:
            return self.translation_cache[text]
        
        try:
            # Rate limiting
            time.sleep(0.5)
            
            # Chia nhỏ text nếu quá dài
            if len(text) > 4500:
                chunks = [text[i:i+4500] for i in range(0, len(text), 4500)]
                translated_chunks = []
                for chunk in chunks:
                    translated = self.translator.translate(chunk)
                    translated_chunks.append(translated)
                    time.sleep(0.5)
                result = " ".join(translated_chunks)
            else:
                result = self.translator.translate(text)
            
            # Lưu vào cache
            self.translation_cache[text] = result
            
            return result
            
        except Exception as e:
            logger.warning(f"Translation error: {e}")
            return text  # Trả về text gốc nếu lỗi
    
    def save_data(self, data: List[Dict], filename: str):
        """Lưu dữ liệu đã xử lý"""
        output_file = self.output_dir / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(data)} items to {output_file}")
        self._save_cache()


class ICD10Processor(InternationalDataProcessor):
    """
    Xử lý dữ liệu ICD-10 (International Classification of Diseases)
    Download từ: https://www.cms.gov/medicare/coding/icd10
    """
    
    def __init__(self):
        super().__init__("icd10")
        
        # ICD-10 categories và mã bệnh phổ biến
        self.icd10_data = self._get_icd10_sample_data()
    
    def _get_icd10_sample_data(self) -> List[Dict]:
        """
        Dữ liệu mẫu ICD-10 - Trong thực tế cần download từ CMS
        Đây là các mã bệnh phổ biến
        """
        return [
            # Bệnh truyền nhiễm (A00-B99)
            {"code": "A00", "name_en": "Cholera", "category": "Infectious diseases"},
            {"code": "A01", "name_en": "Typhoid and paratyphoid fevers", "category": "Infectious diseases"},
            {"code": "A09", "name_en": "Infectious gastroenteritis and colitis", "category": "Infectious diseases"},
            {"code": "A15", "name_en": "Respiratory tuberculosis", "category": "Infectious diseases"},
            {"code": "A37", "name_en": "Whooping cough", "category": "Infectious diseases"},
            {"code": "A38", "name_en": "Scarlet fever", "category": "Infectious diseases"},
            {"code": "A39", "name_en": "Meningococcal infection", "category": "Infectious diseases"},
            {"code": "A40", "name_en": "Streptococcal sepsis", "category": "Infectious diseases"},
            {"code": "B01", "name_en": "Varicella (chickenpox)", "category": "Infectious diseases"},
            {"code": "B02", "name_en": "Zoster (herpes zoster)", "category": "Infectious diseases"},
            {"code": "B05", "name_en": "Measles", "category": "Infectious diseases"},
            {"code": "B06", "name_en": "Rubella (German measles)", "category": "Infectious diseases"},
            {"code": "B15", "name_en": "Acute hepatitis A", "category": "Infectious diseases"},
            {"code": "B16", "name_en": "Acute hepatitis B", "category": "Infectious diseases"},
            {"code": "B17", "name_en": "Other acute viral hepatitis", "category": "Infectious diseases"},
            {"code": "B18", "name_en": "Chronic viral hepatitis", "category": "Infectious diseases"},
            {"code": "B20", "name_en": "Human immunodeficiency virus (HIV) disease", "category": "Infectious diseases"},
            {"code": "B26", "name_en": "Mumps", "category": "Infectious diseases"},
            {"code": "B27", "name_en": "Infectious mononucleosis", "category": "Infectious diseases"},
            {"code": "B34", "name_en": "Viral infection of unspecified site", "category": "Infectious diseases"},
            {"code": "B35", "name_en": "Dermatophytosis", "category": "Infectious diseases"},
            {"code": "B37", "name_en": "Candidiasis", "category": "Infectious diseases"},
            {"code": "B50", "name_en": "Plasmodium falciparum malaria", "category": "Infectious diseases"},
            
            # Ung thư (C00-D48)
            {"code": "C00", "name_en": "Malignant neoplasm of lip", "category": "Neoplasms"},
            {"code": "C15", "name_en": "Malignant neoplasm of esophagus", "category": "Neoplasms"},
            {"code": "C16", "name_en": "Malignant neoplasm of stomach", "category": "Neoplasms"},
            {"code": "C18", "name_en": "Malignant neoplasm of colon", "category": "Neoplasms"},
            {"code": "C20", "name_en": "Malignant neoplasm of rectum", "category": "Neoplasms"},
            {"code": "C22", "name_en": "Malignant neoplasm of liver", "category": "Neoplasms"},
            {"code": "C25", "name_en": "Malignant neoplasm of pancreas", "category": "Neoplasms"},
            {"code": "C34", "name_en": "Malignant neoplasm of bronchus and lung", "category": "Neoplasms"},
            {"code": "C43", "name_en": "Malignant melanoma of skin", "category": "Neoplasms"},
            {"code": "C50", "name_en": "Malignant neoplasm of breast", "category": "Neoplasms"},
            {"code": "C53", "name_en": "Malignant neoplasm of cervix uteri", "category": "Neoplasms"},
            {"code": "C56", "name_en": "Malignant neoplasm of ovary", "category": "Neoplasms"},
            {"code": "C61", "name_en": "Malignant neoplasm of prostate", "category": "Neoplasms"},
            {"code": "C64", "name_en": "Malignant neoplasm of kidney", "category": "Neoplasms"},
            {"code": "C67", "name_en": "Malignant neoplasm of bladder", "category": "Neoplasms"},
            {"code": "C71", "name_en": "Malignant neoplasm of brain", "category": "Neoplasms"},
            {"code": "C73", "name_en": "Malignant neoplasm of thyroid gland", "category": "Neoplasms"},
            {"code": "C81", "name_en": "Hodgkin lymphoma", "category": "Neoplasms"},
            {"code": "C90", "name_en": "Multiple myeloma", "category": "Neoplasms"},
            {"code": "C91", "name_en": "Lymphoid leukemia", "category": "Neoplasms"},
            {"code": "C92", "name_en": "Myeloid leukemia", "category": "Neoplasms"},
            
            # Bệnh nội tiết, dinh dưỡng (E00-E90)
            {"code": "E03", "name_en": "Hypothyroidism", "category": "Endocrine diseases"},
            {"code": "E05", "name_en": "Hyperthyroidism (thyrotoxicosis)", "category": "Endocrine diseases"},
            {"code": "E10", "name_en": "Type 1 diabetes mellitus", "category": "Endocrine diseases"},
            {"code": "E11", "name_en": "Type 2 diabetes mellitus", "category": "Endocrine diseases"},
            {"code": "E66", "name_en": "Obesity", "category": "Endocrine diseases"},
            {"code": "E78", "name_en": "Disorders of lipoprotein metabolism (hyperlipidemia)", "category": "Endocrine diseases"},
            {"code": "E86", "name_en": "Volume depletion (dehydration)", "category": "Endocrine diseases"},
            
            # Rối loạn tâm thần (F00-F99)
            {"code": "F00", "name_en": "Dementia in Alzheimer disease", "category": "Mental disorders"},
            {"code": "F10", "name_en": "Alcohol related disorders", "category": "Mental disorders"},
            {"code": "F20", "name_en": "Schizophrenia", "category": "Mental disorders"},
            {"code": "F31", "name_en": "Bipolar disorder", "category": "Mental disorders"},
            {"code": "F32", "name_en": "Major depressive disorder, single episode", "category": "Mental disorders"},
            {"code": "F33", "name_en": "Major depressive disorder, recurrent", "category": "Mental disorders"},
            {"code": "F40", "name_en": "Phobic anxiety disorders", "category": "Mental disorders"},
            {"code": "F41", "name_en": "Other anxiety disorders", "category": "Mental disorders"},
            {"code": "F42", "name_en": "Obsessive-compulsive disorder", "category": "Mental disorders"},
            {"code": "F43", "name_en": "Reaction to severe stress and adjustment disorders", "category": "Mental disorders"},
            {"code": "F50", "name_en": "Eating disorders", "category": "Mental disorders"},
            {"code": "F84", "name_en": "Pervasive developmental disorders (autism)", "category": "Mental disorders"},
            {"code": "F90", "name_en": "Attention-deficit hyperactivity disorders", "category": "Mental disorders"},
            
            # Bệnh thần kinh (G00-G99)
            {"code": "G00", "name_en": "Bacterial meningitis", "category": "Nervous system diseases"},
            {"code": "G03", "name_en": "Meningitis due to other causes", "category": "Nervous system diseases"},
            {"code": "G20", "name_en": "Parkinson disease", "category": "Nervous system diseases"},
            {"code": "G30", "name_en": "Alzheimer disease", "category": "Nervous system diseases"},
            {"code": "G35", "name_en": "Multiple sclerosis", "category": "Nervous system diseases"},
            {"code": "G40", "name_en": "Epilepsy", "category": "Nervous system diseases"},
            {"code": "G43", "name_en": "Migraine", "category": "Nervous system diseases"},
            {"code": "G44", "name_en": "Other headache syndromes", "category": "Nervous system diseases"},
            {"code": "G47", "name_en": "Sleep disorders", "category": "Nervous system diseases"},
            {"code": "G50", "name_en": "Disorders of trigeminal nerve", "category": "Nervous system diseases"},
            {"code": "G51", "name_en": "Facial nerve disorders (Bell palsy)", "category": "Nervous system diseases"},
            {"code": "G56", "name_en": "Mononeuropathies of upper limb (carpal tunnel)", "category": "Nervous system diseases"},
            {"code": "G80", "name_en": "Cerebral palsy", "category": "Nervous system diseases"},
            
            # Bệnh mắt (H00-H59)
            {"code": "H00", "name_en": "Hordeolum and chalazion", "category": "Eye diseases"},
            {"code": "H01", "name_en": "Other inflammation of eyelid", "category": "Eye diseases"},
            {"code": "H10", "name_en": "Conjunctivitis", "category": "Eye diseases"},
            {"code": "H25", "name_en": "Age-related cataract", "category": "Eye diseases"},
            {"code": "H26", "name_en": "Other cataract", "category": "Eye diseases"},
            {"code": "H33", "name_en": "Retinal detachments and breaks", "category": "Eye diseases"},
            {"code": "H35", "name_en": "Other retinal disorders", "category": "Eye diseases"},
            {"code": "H40", "name_en": "Glaucoma", "category": "Eye diseases"},
            {"code": "H52", "name_en": "Disorders of refraction and accommodation", "category": "Eye diseases"},
            
            # Bệnh tim mạch (I00-I99)
            {"code": "I10", "name_en": "Essential (primary) hypertension", "category": "Circulatory diseases"},
            {"code": "I11", "name_en": "Hypertensive heart disease", "category": "Circulatory diseases"},
            {"code": "I20", "name_en": "Angina pectoris", "category": "Circulatory diseases"},
            {"code": "I21", "name_en": "Acute myocardial infarction", "category": "Circulatory diseases"},
            {"code": "I25", "name_en": "Chronic ischemic heart disease", "category": "Circulatory diseases"},
            {"code": "I26", "name_en": "Pulmonary embolism", "category": "Circulatory diseases"},
            {"code": "I42", "name_en": "Cardiomyopathy", "category": "Circulatory diseases"},
            {"code": "I48", "name_en": "Atrial fibrillation and flutter", "category": "Circulatory diseases"},
            {"code": "I50", "name_en": "Heart failure", "category": "Circulatory diseases"},
            {"code": "I60", "name_en": "Nontraumatic subarachnoid hemorrhage", "category": "Circulatory diseases"},
            {"code": "I61", "name_en": "Nontraumatic intracerebral hemorrhage", "category": "Circulatory diseases"},
            {"code": "I63", "name_en": "Cerebral infarction (stroke)", "category": "Circulatory diseases"},
            {"code": "I64", "name_en": "Stroke, not specified as hemorrhage or infarction", "category": "Circulatory diseases"},
            {"code": "I70", "name_en": "Atherosclerosis", "category": "Circulatory diseases"},
            {"code": "I73", "name_en": "Other peripheral vascular diseases", "category": "Circulatory diseases"},
            {"code": "I80", "name_en": "Phlebitis and thrombophlebitis", "category": "Circulatory diseases"},
            {"code": "I83", "name_en": "Varicose veins of lower extremities", "category": "Circulatory diseases"},
            {"code": "I84", "name_en": "Hemorrhoids", "category": "Circulatory diseases"},
            
            # Bệnh hô hấp (J00-J99)
            {"code": "J00", "name_en": "Acute nasopharyngitis (common cold)", "category": "Respiratory diseases"},
            {"code": "J01", "name_en": "Acute sinusitis", "category": "Respiratory diseases"},
            {"code": "J02", "name_en": "Acute pharyngitis", "category": "Respiratory diseases"},
            {"code": "J03", "name_en": "Acute tonsillitis", "category": "Respiratory diseases"},
            {"code": "J04", "name_en": "Acute laryngitis and tracheitis", "category": "Respiratory diseases"},
            {"code": "J06", "name_en": "Acute upper respiratory infections", "category": "Respiratory diseases"},
            {"code": "J09", "name_en": "Influenza due to identified influenza virus", "category": "Respiratory diseases"},
            {"code": "J10", "name_en": "Influenza due to other identified influenza virus", "category": "Respiratory diseases"},
            {"code": "J11", "name_en": "Influenza due to unidentified influenza virus", "category": "Respiratory diseases"},
            {"code": "J12", "name_en": "Viral pneumonia", "category": "Respiratory diseases"},
            {"code": "J13", "name_en": "Pneumonia due to Streptococcus pneumoniae", "category": "Respiratory diseases"},
            {"code": "J15", "name_en": "Bacterial pneumonia", "category": "Respiratory diseases"},
            {"code": "J18", "name_en": "Pneumonia, unspecified organism", "category": "Respiratory diseases"},
            {"code": "J20", "name_en": "Acute bronchitis", "category": "Respiratory diseases"},
            {"code": "J21", "name_en": "Acute bronchiolitis", "category": "Respiratory diseases"},
            {"code": "J30", "name_en": "Allergic rhinitis", "category": "Respiratory diseases"},
            {"code": "J32", "name_en": "Chronic sinusitis", "category": "Respiratory diseases"},
            {"code": "J35", "name_en": "Chronic diseases of tonsils and adenoids", "category": "Respiratory diseases"},
            {"code": "J40", "name_en": "Bronchitis, not specified as acute or chronic", "category": "Respiratory diseases"},
            {"code": "J42", "name_en": "Unspecified chronic bronchitis", "category": "Respiratory diseases"},
            {"code": "J43", "name_en": "Emphysema", "category": "Respiratory diseases"},
            {"code": "J44", "name_en": "Chronic obstructive pulmonary disease (COPD)", "category": "Respiratory diseases"},
            {"code": "J45", "name_en": "Asthma", "category": "Respiratory diseases"},
            {"code": "J47", "name_en": "Bronchiectasis", "category": "Respiratory diseases"},
            {"code": "J84", "name_en": "Other interstitial pulmonary diseases", "category": "Respiratory diseases"},
            {"code": "J96", "name_en": "Respiratory failure", "category": "Respiratory diseases"},
            
            # Bệnh tiêu hóa (K00-K93)
            {"code": "K00", "name_en": "Disorders of tooth development and eruption", "category": "Digestive diseases"},
            {"code": "K02", "name_en": "Dental caries", "category": "Digestive diseases"},
            {"code": "K04", "name_en": "Diseases of pulp and periapical tissues", "category": "Digestive diseases"},
            {"code": "K05", "name_en": "Gingivitis and periodontal diseases", "category": "Digestive diseases"},
            {"code": "K20", "name_en": "Esophagitis", "category": "Digestive diseases"},
            {"code": "K21", "name_en": "Gastro-esophageal reflux disease (GERD)", "category": "Digestive diseases"},
            {"code": "K25", "name_en": "Gastric ulcer", "category": "Digestive diseases"},
            {"code": "K26", "name_en": "Duodenal ulcer", "category": "Digestive diseases"},
            {"code": "K27", "name_en": "Peptic ulcer, site unspecified", "category": "Digestive diseases"},
            {"code": "K29", "name_en": "Gastritis and duodenitis", "category": "Digestive diseases"},
            {"code": "K35", "name_en": "Acute appendicitis", "category": "Digestive diseases"},
            {"code": "K40", "name_en": "Inguinal hernia", "category": "Digestive diseases"},
            {"code": "K50", "name_en": "Crohn disease", "category": "Digestive diseases"},
            {"code": "K51", "name_en": "Ulcerative colitis", "category": "Digestive diseases"},
            {"code": "K57", "name_en": "Diverticular disease of intestine", "category": "Digestive diseases"},
            {"code": "K58", "name_en": "Irritable bowel syndrome", "category": "Digestive diseases"},
            {"code": "K59", "name_en": "Other functional intestinal disorders", "category": "Digestive diseases"},
            {"code": "K70", "name_en": "Alcoholic liver disease", "category": "Digestive diseases"},
            {"code": "K74", "name_en": "Fibrosis and cirrhosis of liver", "category": "Digestive diseases"},
            {"code": "K76", "name_en": "Other diseases of liver (fatty liver)", "category": "Digestive diseases"},
            {"code": "K80", "name_en": "Cholelithiasis (gallstones)", "category": "Digestive diseases"},
            {"code": "K81", "name_en": "Cholecystitis", "category": "Digestive diseases"},
            {"code": "K85", "name_en": "Acute pancreatitis", "category": "Digestive diseases"},
            {"code": "K86", "name_en": "Other diseases of pancreas", "category": "Digestive diseases"},
            
            # Bệnh da (L00-L99)
            {"code": "L00", "name_en": "Staphylococcal scalded skin syndrome", "category": "Skin diseases"},
            {"code": "L01", "name_en": "Impetigo", "category": "Skin diseases"},
            {"code": "L02", "name_en": "Cutaneous abscess, furuncle and carbuncle", "category": "Skin diseases"},
            {"code": "L03", "name_en": "Cellulitis and acute lymphangitis", "category": "Skin diseases"},
            {"code": "L20", "name_en": "Atopic dermatitis", "category": "Skin diseases"},
            {"code": "L21", "name_en": "Seborrheic dermatitis", "category": "Skin diseases"},
            {"code": "L23", "name_en": "Allergic contact dermatitis", "category": "Skin diseases"},
            {"code": "L30", "name_en": "Other dermatitis", "category": "Skin diseases"},
            {"code": "L40", "name_en": "Psoriasis", "category": "Skin diseases"},
            {"code": "L50", "name_en": "Urticaria (hives)", "category": "Skin diseases"},
            {"code": "L60", "name_en": "Nail disorders", "category": "Skin diseases"},
            {"code": "L63", "name_en": "Alopecia areata", "category": "Skin diseases"},
            {"code": "L70", "name_en": "Acne", "category": "Skin diseases"},
            {"code": "L80", "name_en": "Vitiligo", "category": "Skin diseases"},
            
            # Bệnh cơ xương khớp (M00-M99)
            {"code": "M05", "name_en": "Rheumatoid arthritis", "category": "Musculoskeletal diseases"},
            {"code": "M10", "name_en": "Gout", "category": "Musculoskeletal diseases"},
            {"code": "M15", "name_en": "Polyosteoarthritis", "category": "Musculoskeletal diseases"},
            {"code": "M16", "name_en": "Osteoarthritis of hip", "category": "Musculoskeletal diseases"},
            {"code": "M17", "name_en": "Osteoarthritis of knee", "category": "Musculoskeletal diseases"},
            {"code": "M19", "name_en": "Other osteoarthritis", "category": "Musculoskeletal diseases"},
            {"code": "M32", "name_en": "Systemic lupus erythematosus", "category": "Musculoskeletal diseases"},
            {"code": "M35", "name_en": "Other systemic involvement of connective tissue", "category": "Musculoskeletal diseases"},
            {"code": "M41", "name_en": "Scoliosis", "category": "Musculoskeletal diseases"},
            {"code": "M47", "name_en": "Spondylosis", "category": "Musculoskeletal diseases"},
            {"code": "M50", "name_en": "Cervical disc disorders", "category": "Musculoskeletal diseases"},
            {"code": "M51", "name_en": "Thoracic, thoracolumbar, and lumbosacral disc disorders", "category": "Musculoskeletal diseases"},
            {"code": "M54", "name_en": "Dorsalgia (back pain)", "category": "Musculoskeletal diseases"},
            {"code": "M62", "name_en": "Other disorders of muscle", "category": "Musculoskeletal diseases"},
            {"code": "M65", "name_en": "Synovitis and tenosynovitis", "category": "Musculoskeletal diseases"},
            {"code": "M75", "name_en": "Shoulder lesions", "category": "Musculoskeletal diseases"},
            {"code": "M79", "name_en": "Other soft tissue disorders (fibromyalgia)", "category": "Musculoskeletal diseases"},
            {"code": "M80", "name_en": "Osteoporosis with pathological fracture", "category": "Musculoskeletal diseases"},
            {"code": "M81", "name_en": "Osteoporosis without pathological fracture", "category": "Musculoskeletal diseases"},
            
            # Bệnh thận và tiết niệu (N00-N99)
            {"code": "N00", "name_en": "Acute nephritic syndrome", "category": "Genitourinary diseases"},
            {"code": "N03", "name_en": "Chronic nephritic syndrome", "category": "Genitourinary diseases"},
            {"code": "N04", "name_en": "Nephrotic syndrome", "category": "Genitourinary diseases"},
            {"code": "N10", "name_en": "Acute pyelonephritis", "category": "Genitourinary diseases"},
            {"code": "N11", "name_en": "Chronic tubulo-interstitial nephritis", "category": "Genitourinary diseases"},
            {"code": "N17", "name_en": "Acute kidney failure", "category": "Genitourinary diseases"},
            {"code": "N18", "name_en": "Chronic kidney disease", "category": "Genitourinary diseases"},
            {"code": "N20", "name_en": "Calculus of kidney and ureter (kidney stones)", "category": "Genitourinary diseases"},
            {"code": "N30", "name_en": "Cystitis", "category": "Genitourinary diseases"},
            {"code": "N39", "name_en": "Other disorders of urinary system", "category": "Genitourinary diseases"},
            {"code": "N40", "name_en": "Benign prostatic hyperplasia", "category": "Genitourinary diseases"},
            {"code": "N41", "name_en": "Inflammatory diseases of prostate", "category": "Genitourinary diseases"},
            {"code": "N60", "name_en": "Benign mammary dysplasia", "category": "Genitourinary diseases"},
            {"code": "N80", "name_en": "Endometriosis", "category": "Genitourinary diseases"},
            {"code": "N83", "name_en": "Noninflammatory disorders of ovary", "category": "Genitourinary diseases"},
            {"code": "N91", "name_en": "Absent, scanty and rare menstruation", "category": "Genitourinary diseases"},
            {"code": "N92", "name_en": "Excessive, frequent and irregular menstruation", "category": "Genitourinary diseases"},
            {"code": "N94", "name_en": "Pain and other conditions associated with female genital organs", "category": "Genitourinary diseases"},
        ]
    
    def process(self) -> List[Dict]:
        """Xử lý và dịch dữ liệu ICD-10"""
        processed_data = []
        
        logger.info(f"Processing {len(self.icd10_data)} ICD-10 codes")
        
        for item in tqdm(self.icd10_data, desc="Translating ICD-10"):
            try:
                name_vi = self.translate_text(item['name_en'])
                category_vi = self.translate_text(item['category'])
                
                processed_data.append({
                    'code': item['code'],
                    'name_en': item['name_en'],
                    'name_vi': name_vi,
                    'category_en': item['category'],
                    'category_vi': category_vi,
                    'source': 'ICD-10',
                    'type': 'disease'
                })
                
            except Exception as e:
                logger.error(f"Error processing {item['code']}: {e}")
        
        self.save_data(processed_data, "icd10_diseases.json")
        return processed_data


class MeSHProcessor(InternationalDataProcessor):
    """
    Xử lý dữ liệu MeSH (Medical Subject Headings)
    Download từ: https://www.nlm.nih.gov/mesh/
    """
    
    def __init__(self):
        super().__init__("mesh")
        self.mesh_data = self._get_mesh_sample_data()
    
    def _get_mesh_sample_data(self) -> List[Dict]:
        """Dữ liệu mẫu MeSH - symptoms và diseases"""
        return [
            # Symptoms
            {"mesh_id": "D000006", "term_en": "Abdominal Pain", "category": "Symptoms", "definition": "Sensation of discomfort, distress, or agony in the abdominal region."},
            {"mesh_id": "D000855", "term_en": "Anorexia", "category": "Symptoms", "definition": "The lack or loss of appetite for food."},
            {"mesh_id": "D001247", "term_en": "Asthenia", "category": "Symptoms", "definition": "Clinical sign or symptom manifested as debility, or lack or loss of strength and energy."},
            {"mesh_id": "D001416", "term_en": "Back Pain", "category": "Symptoms", "definition": "Acute or chronic pain located in the posterior regions of the back."},
            {"mesh_id": "D002637", "term_en": "Chest Pain", "category": "Symptoms", "definition": "Pressure, burning, or numbness in the chest."},
            {"mesh_id": "D003371", "term_en": "Cough", "category": "Symptoms", "definition": "A sudden, audible expulsion of air from the lungs through a partially closed glottis."},
            {"mesh_id": "D003967", "term_en": "Diarrhea", "category": "Symptoms", "definition": "Increased liquidity or decreased consistency of feces."},
            {"mesh_id": "D004244", "term_en": "Dizziness", "category": "Symptoms", "definition": "An imprecise term which may refer to a sense of spatial disorientation."},
            {"mesh_id": "D004417", "term_en": "Dyspnea", "category": "Symptoms", "definition": "Difficult or labored breathing."},
            {"mesh_id": "D005221", "term_en": "Fatigue", "category": "Symptoms", "definition": "The state of weariness following a period of exertion."},
            {"mesh_id": "D005334", "term_en": "Fever", "category": "Symptoms", "definition": "An abnormal elevation of body temperature."},
            {"mesh_id": "D006261", "term_en": "Headache", "category": "Symptoms", "definition": "The symptom of pain in the cranial region."},
            {"mesh_id": "D007035", "term_en": "Hypothermia", "category": "Symptoms", "definition": "Lower than normal body temperature."},
            {"mesh_id": "D007565", "term_en": "Jaundice", "category": "Symptoms", "definition": "A clinical manifestation of hyperbilirubinemia."},
            {"mesh_id": "D009325", "term_en": "Nausea", "category": "Symptoms", "definition": "An unpleasant sensation in the stomach usually accompanied by the urge to vomit."},
            {"mesh_id": "D010146", "term_en": "Pain", "category": "Symptoms", "definition": "An unpleasant sensation induced by noxious stimuli."},
            {"mesh_id": "D011537", "term_en": "Pruritus", "category": "Symptoms", "definition": "An intense itching sensation that produces the urge to rub or scratch the skin."},
            {"mesh_id": "D012640", "term_en": "Seizures", "category": "Symptoms", "definition": "Clinical or subclinical disturbances of cortical function due to abnormal electrical activity."},
            {"mesh_id": "D014839", "term_en": "Vomiting", "category": "Symptoms", "definition": "The forcible expulsion of the contents of the stomach through the mouth."},
            {"mesh_id": "D014883", "term_en": "Water-Electrolyte Imbalance", "category": "Symptoms", "definition": "Disturbances in the body's fluid or electrolyte status."},
            {"mesh_id": "D000857", "term_en": "Anosmia", "category": "Symptoms", "definition": "Complete loss of the sense of smell."},
            {"mesh_id": "D000379", "term_en": "Ageusia", "category": "Symptoms", "definition": "Complete loss of the sense of taste."},
            {"mesh_id": "D006469", "term_en": "Hemoptysis", "category": "Symptoms", "definition": "Expectoration or spitting of blood."},
            {"mesh_id": "D006471", "term_en": "Hemorrhage", "category": "Symptoms", "definition": "Bleeding or escape of blood from a vessel."},
            {"mesh_id": "D004487", "term_en": "Edema", "category": "Symptoms", "definition": "Abnormal fluid accumulation in tissues or body cavities."},
            {"mesh_id": "D003248", "term_en": "Constipation", "category": "Symptoms", "definition": "Infrequent or difficult evacuation of feces."},
            {"mesh_id": "D007319", "term_en": "Insomnia", "category": "Symptoms", "definition": "Disorders characterized by impairment of the ability to initiate or maintain sleep."},
            {"mesh_id": "D015431", "term_en": "Weight Loss", "category": "Symptoms", "definition": "Decrease in existing body weight."},
            {"mesh_id": "D000334", "term_en": "Aerophagy", "category": "Symptoms", "definition": "Spasmodic swallowing of air."},
            {"mesh_id": "D014202", "term_en": "Tremor", "category": "Symptoms", "definition": "Cyclical movement of a body part that can represent either a physiological process or a manifestation of disease."},
            
            # Drugs/Treatments  
            {"mesh_id": "D000900", "term_en": "Anti-Bacterial Agents", "category": "Drugs", "definition": "Substances that prevent infectious agents or organisms from spreading."},
            {"mesh_id": "D000894", "term_en": "Anti-Inflammatory Agents, Non-Steroidal", "category": "Drugs", "definition": "Anti-inflammatory agents that are not steroids."},
            {"mesh_id": "D000700", "term_en": "Analgesics", "category": "Drugs", "definition": "Compounds capable of relieving pain without loss of consciousness."},
            {"mesh_id": "D000927", "term_en": "Anticonvulsants", "category": "Drugs", "definition": "Drugs used to prevent seizures or reduce their severity."},
            {"mesh_id": "D000928", "term_en": "Antidepressive Agents", "category": "Drugs", "definition": "Mood-stimulating drugs used primarily in the treatment of depression."},
            {"mesh_id": "D000959", "term_en": "Antihypertensive Agents", "category": "Drugs", "definition": "Drugs used in the treatment of hypertension."},
            {"mesh_id": "D000970", "term_en": "Antineoplastic Agents", "category": "Drugs", "definition": "Substances that inhibit or prevent the proliferation of neoplasms."},
            {"mesh_id": "D000998", "term_en": "Antiviral Agents", "category": "Drugs", "definition": "Agents used in the treatment of virus diseases."},
            {"mesh_id": "D007004", "term_en": "Hypoglycemic Agents", "category": "Drugs", "definition": "Substances which lower blood glucose levels."},
            {"mesh_id": "D000960", "term_en": "Hypolipidemic Agents", "category": "Drugs", "definition": "Substances that lower the levels of fats in the blood."},
        ]
    
    def process(self) -> List[Dict]:
        """Xử lý và dịch dữ liệu MeSH"""
        processed_data = []
        
        logger.info(f"Processing {len(self.mesh_data)} MeSH terms")
        
        for item in tqdm(self.mesh_data, desc="Translating MeSH"):
            try:
                term_vi = self.translate_text(item['term_en'])
                definition_vi = self.translate_text(item['definition'])
                
                # Xác định loại (bệnh/triệu chứng/thuốc)
                if item['category'] == 'Symptoms':
                    data_type = 'symptom'
                elif item['category'] == 'Drugs':
                    data_type = 'drug'
                else:
                    data_type = 'disease'
                
                processed_data.append({
                    'mesh_id': item['mesh_id'],
                    'term_en': item['term_en'],
                    'term_vi': term_vi,
                    'definition_en': item['definition'],
                    'definition_vi': definition_vi,
                    'category': item['category'],
                    'source': 'MeSH',
                    'type': data_type
                })
                
            except Exception as e:
                logger.error(f"Error processing {item['mesh_id']}: {e}")
        
        self.save_data(processed_data, "mesh_terms.json")
        return processed_data


class HPOProcessor(InternationalDataProcessor):
    """
    Xử lý dữ liệu HPO (Human Phenotype Ontology)
    Tập trung vào triệu chứng/phenotypes
    Download từ: https://hpo.jax.org/
    """
    
    def __init__(self):
        super().__init__("hpo")
        self.hpo_data = self._get_hpo_sample_data()
    
    def _get_hpo_sample_data(self) -> List[Dict]:
        """Dữ liệu mẫu HPO - phenotypes/symptoms"""
        return [
            {"hpo_id": "HP:0001945", "term_en": "Fever", "definition": "Body temperature elevated above the normal range."},
            {"hpo_id": "HP:0002315", "term_en": "Headache", "definition": "Pain in the head."},
            {"hpo_id": "HP:0002017", "term_en": "Nausea and vomiting", "definition": "Uncontrolled regurgitation of stomach contents."},
            {"hpo_id": "HP:0002014", "term_en": "Diarrhea", "definition": "Abnormally loose or watery stools."},
            {"hpo_id": "HP:0012735", "term_en": "Cough", "definition": "A sudden expulsion of air from the lungs."},
            {"hpo_id": "HP:0002094", "term_en": "Dyspnea", "definition": "Difficulty in breathing."},
            {"hpo_id": "HP:0000989", "term_en": "Pruritus", "definition": "Itch or itchy skin."},
            {"hpo_id": "HP:0000988", "term_en": "Skin rash", "definition": "A visible change in the skin."},
            {"hpo_id": "HP:0001824", "term_en": "Weight loss", "definition": "Reduction in body weight."},
            {"hpo_id": "HP:0001297", "term_en": "Stroke", "definition": "Sudden focal neurological deficit."},
            {"hpo_id": "HP:0001250", "term_en": "Seizure", "definition": "Convulsions or fits."},
            {"hpo_id": "HP:0002321", "term_en": "Vertigo", "definition": "Sensation of movement when still."},
            {"hpo_id": "HP:0000360", "term_en": "Tinnitus", "definition": "Ringing in the ears."},
            {"hpo_id": "HP:0000365", "term_en": "Hearing impairment", "definition": "Loss of hearing ability."},
            {"hpo_id": "HP:0000505", "term_en": "Visual impairment", "definition": "Reduced vision."},
            {"hpo_id": "HP:0001635", "term_en": "Congestive heart failure", "definition": "Heart unable to pump blood effectively."},
            {"hpo_id": "HP:0001658", "term_en": "Myocardial infarction", "definition": "Heart attack."},
            {"hpo_id": "HP:0002607", "term_en": "Bowel incontinence", "definition": "Loss of bowel control."},
            {"hpo_id": "HP:0000020", "term_en": "Urinary incontinence", "definition": "Loss of bladder control."},
            {"hpo_id": "HP:0003326", "term_en": "Myalgia", "definition": "Muscle pain."},
            {"hpo_id": "HP:0002829", "term_en": "Arthralgia", "definition": "Joint pain."},
            {"hpo_id": "HP:0002018", "term_en": "Nausea", "definition": "Feeling of sickness."},
            {"hpo_id": "HP:0002019", "term_en": "Constipation", "definition": "Difficulty passing stools."},
            {"hpo_id": "HP:0002027", "term_en": "Abdominal pain", "definition": "Pain in the stomach area."},
            {"hpo_id": "HP:0002039", "term_en": "Anorexia", "definition": "Loss of appetite."},
            {"hpo_id": "HP:0002240", "term_en": "Hepatomegaly", "definition": "Enlarged liver."},
            {"hpo_id": "HP:0001744", "term_en": "Splenomegaly", "definition": "Enlarged spleen."},
            {"hpo_id": "HP:0002716", "term_en": "Lymphadenopathy", "definition": "Enlarged lymph nodes."},
            {"hpo_id": "HP:0000952", "term_en": "Jaundice", "definition": "Yellow discoloration of skin."},
            {"hpo_id": "HP:0000969", "term_en": "Edema", "definition": "Swelling from fluid accumulation."},
        ]
    
    def process(self) -> List[Dict]:
        """Xử lý và dịch dữ liệu HPO"""
        processed_data = []
        
        logger.info(f"Processing {len(self.hpo_data)} HPO terms")
        
        for item in tqdm(self.hpo_data, desc="Translating HPO"):
            try:
                term_vi = self.translate_text(item['term_en'])
                definition_vi = self.translate_text(item['definition'])
                
                processed_data.append({
                    'hpo_id': item['hpo_id'],
                    'term_en': item['term_en'],
                    'term_vi': term_vi,
                    'definition_en': item['definition'],
                    'definition_vi': definition_vi,
                    'source': 'HPO',
                    'type': 'symptom'
                })
                
            except Exception as e:
                logger.error(f"Error processing {item['hpo_id']}: {e}")
        
        self.save_data(processed_data, "hpo_symptoms.json")
        return processed_data


def process_all_international_data():
    """Xử lý tất cả dữ liệu quốc tế"""
    processors = [
        ICD10Processor(),
        MeSHProcessor(),
        HPOProcessor(),
    ]
    
    all_data = {
        'diseases': [],
        'symptoms': [],
        'drugs': []
    }
    
    for processor in processors:
        logger.info(f"Processing {processor.source_name}")
        try:
            data = processor.process()
            
            # Phân loại theo type
            for item in data:
                item_type = item.get('type', 'disease')
                if item_type == 'symptom':
                    all_data['symptoms'].append(item)
                elif item_type == 'drug':
                    all_data['drugs'].append(item)
                else:
                    all_data['diseases'].append(item)
                    
        except Exception as e:
            logger.error(f"Error processing {processor.source_name}: {e}")
    
    # Lưu tổng hợp
    for category, data in all_data.items():
        output_file = EXTERNAL_DATA_DIR / f"international_{category}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(data)} {category} items")
    
    return all_data


if __name__ == "__main__":
    process_all_international_data()
