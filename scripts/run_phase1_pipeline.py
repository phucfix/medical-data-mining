#!/usr/bin/env python3
"""
Script chạy toàn bộ Phase 1: Thu thập và tiền xử lý dữ liệu
Chỉ sử dụng Google Gemini API
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from tqdm import tqdm

# Setup logging
log_file = Path(__file__).parent.parent / "logs" / f"phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_file.parent.mkdir(exist_ok=True)
logger.add(log_file, rotation="10 MB")


def step1_process_international_data():
    """Bước 1: Xử lý dữ liệu quốc tế (ICD-10, MeSH, HPO)"""
    print("\n" + "="*60)
    print("📌 BƯỚC 1: Xử lý dữ liệu quốc tế (ICD-10, MeSH, HPO)")
    print("="*60)
    
    from src.translation import process_all_international_data
    
    try:
        data = process_all_international_data()
        
        total = sum(len(v) for v in data.values())
        print(f"✅ Đã xử lý {total} records từ nguồn quốc tế")
        print(f"   - Diseases: {len(data.get('diseases', []))}")
        print(f"   - Symptoms: {len(data.get('symptoms', []))}")
        print(f"   - Drugs: {len(data.get('drugs', []))}")
        
        return data
    except Exception as e:
        logger.error(f"Error in step 1: {e}")
        print(f"❌ Lỗi: {e}")
        return None


def step2_generate_qa_with_gemini():
    """Bước 2: Sinh câu hỏi Q&A với Gemini"""
    print("\n" + "="*60)
    print("📌 BƯỚC 2: Sinh câu hỏi Q&A với Google Gemini")
    print("="*60)
    
    import os
    from dotenv import load_dotenv
    import google.generativeai as genai
    
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Không tìm thấy GOOGLE_API_KEY trong .env")
        return None
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Các chủ đề y tế để sinh câu hỏi
    medical_topics = [
        "bệnh tim mạch và huyết áp",
        "bệnh tiểu đường và nội tiết",
        "bệnh hô hấp như viêm phổi, hen suyễn, COPD",
        "bệnh tiêu hóa như viêm dạ dày, gan, ruột",
        "bệnh thần kinh như đau đầu, động kinh, Parkinson",
        "bệnh da liễu như viêm da, vẩy nến, mụn",
        "bệnh xương khớp như viêm khớp, loãng xương, gout",
        "bệnh thận và tiết niệu",
        "bệnh truyền nhiễm như cúm, COVID-19, viêm gan",
        "thuốc kháng sinh và kháng viêm",
        "thuốc giảm đau và hạ sốt",
        "thuốc tim mạch và huyết áp",
        "thuốc tiểu đường",
        "vaccine và tiêm chủng",
        "triệu chứng đau đầu và chóng mặt",
        "triệu chứng sốt và mệt mỏi",
        "triệu chứng ho và khó thở",
        "triệu chứng đau bụng và tiêu chảy",
        "triệu chứng đau ngực và khó thở",
        "dinh dưỡng và sức khỏe",
    ]
    
    all_qa = []
    output_dir = Path(__file__).parent.parent / "data" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Sẽ sinh Q&A cho {len(medical_topics)} chủ đề...")
    
    for topic in tqdm(medical_topics, desc="Generating Q&A"):
        prompt = f"""Bạn là một chuyên gia y tế Việt Nam. Hãy tạo 50 câu hỏi Đúng/Sai về chủ đề "{topic}".

YÊU CẦU QUAN TRỌNG:
1. Tạo CHÍNH XÁC 50 câu hỏi
2. 25 câu có đáp án "Đúng" và 25 câu có đáp án "Sai"
3. Câu hỏi phải CHÍNH XÁC về mặt y khoa
4. Đối với câu Sai, hãy tạo một phát biểu sai về y tế (ví dụ: thay đổi triệu chứng, nguyên nhân, hoặc cách điều trị)
5. Mỗi câu có giải thích ngắn gọn (1-2 câu)
6. Câu hỏi đa dạng về độ khó

VÍ DỤ CÂU HỎI:
- "Ho kéo dài trên 3 tuần có thể là triệu chứng của lao phổi." -> Đúng
- "Sỏi thận hình thành do khoáng chất kết tụ trong nước tiểu." -> Đúng  
- "Thoát vị đĩa đệm là do nhân nhầy đĩa đệm lồi ra chèn dây thần kinh." -> Đúng
- "Động kinh là tình trạng các tế bào não hoạt động bất thường gây co giật." -> Đúng
- "Bệnh tiểu đường type 1 có thể chữa khỏi hoàn toàn bằng chế độ ăn." -> Sai

FORMAT OUTPUT (chỉ trả về JSON, không có text khác):
[
    {{"question": "Câu hỏi 1", "answer": "Đúng", "explanation": "Giải thích"}},
    {{"question": "Câu hỏi 2", "answer": "Sai", "explanation": "Giải thích"}},
    ...
]"""
        
        try:
            response = model.generate_content(prompt)
            response_text = response.text
            
            # Parse JSON
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                qa_pairs = json.loads(json_str)
                
                # Thêm metadata
                for qa in qa_pairs:
                    qa['topic'] = topic
                    qa['source'] = 'gemini_generated'
                    # Chuẩn hóa answer
                    if qa.get('answer', '').lower() in ['đúng', 'true', 'yes']:
                        qa['answer'] = 'Đúng'
                    else:
                        qa['answer'] = 'Sai'
                
                all_qa.extend(qa_pairs)
                logger.info(f"Generated {len(qa_pairs)} Q&A for topic: {topic}")
            
            # Rate limiting - tránh bị block
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Error generating for {topic}: {e}")
            print(f"⚠️ Lỗi với topic '{topic}': {e}")
            time.sleep(5)  # Wait longer on error
            continue
    
    # Lưu kết quả
    if all_qa:
        output_file = output_dir / "medical_qa_dataset.json"
        
        # Count true/false
        true_count = len([q for q in all_qa if q['answer'] == 'Đúng'])
        false_count = len([q for q in all_qa if q['answer'] == 'Sai'])
        
        dataset = {
            "metadata": {
                "total_questions": len(all_qa),
                "true_count": true_count,
                "false_count": false_count,
                "topics": medical_topics,
                "generated_date": datetime.now().isoformat(),
                "model": "gemini-pro"
            },
            "data": all_qa
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Đã sinh {len(all_qa)} câu hỏi Q&A")
        print(f"   - Câu Đúng: {true_count}")
        print(f"   - Câu Sai: {false_count}")
        print(f"   - Lưu tại: {output_file}")
        
        return all_qa
    
    return None


def step3_generate_more_data():
    """Bước 3: Sinh thêm dữ liệu về bệnh, triệu chứng, thuốc"""
    print("\n" + "="*60)
    print("📌 BƯỚC 3: Sinh dữ liệu chi tiết về Bệnh, Triệu chứng, Thuốc")
    print("="*60)
    
    import os
    from dotenv import load_dotenv
    import google.generativeai as genai
    
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    output_dir = Path(__file__).parent.parent / "data" / "generated"
    
    # Sinh dữ liệu bệnh
    diseases_prompt = """Hãy tạo dữ liệu về 100 bệnh phổ biến ở Việt Nam với thông tin chi tiết.

FORMAT OUTPUT (JSON):
[
    {
        "name": "Tên bệnh",
        "description": "Mô tả ngắn về bệnh",
        "symptoms": ["triệu chứng 1", "triệu chứng 2", ...],
        "causes": ["nguyên nhân 1", "nguyên nhân 2", ...],
        "treatment": "Phương pháp điều trị chính",
        "prevention": "Cách phòng ngừa",
        "category": "Nhóm bệnh (tim mạch, hô hấp, tiêu hóa, ...)"
    },
    ...
]

Chỉ trả về JSON, không có text khác."""

    symptoms_prompt = """Hãy tạo dữ liệu về 100 triệu chứng y tế phổ biến với thông tin chi tiết.

FORMAT OUTPUT (JSON):
[
    {
        "name": "Tên triệu chứng",
        "description": "Mô tả chi tiết triệu chứng",
        "related_diseases": ["bệnh liên quan 1", "bệnh liên quan 2", ...],
        "severity": "Mức độ nghiêm trọng (nhẹ/trung bình/nặng)",
        "when_to_see_doctor": "Khi nào cần gặp bác sĩ",
        "category": "Nhóm triệu chứng"
    },
    ...
]

Chỉ trả về JSON, không có text khác."""

    drugs_prompt = """Hãy tạo dữ liệu về 100 loại thuốc phổ biến ở Việt Nam với thông tin chi tiết.

FORMAT OUTPUT (JSON):
[
    {
        "name": "Tên thuốc",
        "active_ingredient": "Hoạt chất chính",
        "indication": "Chỉ định điều trị",
        "dosage": "Liều dùng thông thường",
        "side_effects": ["tác dụng phụ 1", "tác dụng phụ 2", ...],
        "contraindication": "Chống chỉ định",
        "category": "Nhóm thuốc (kháng sinh, giảm đau, ...)"
    },
    ...
]

Chỉ trả về JSON, không có text khác."""

    prompts = [
        ("diseases", diseases_prompt),
        ("symptoms", symptoms_prompt),
        ("drugs", drugs_prompt)
    ]
    
    for category, prompt in prompts:
        print(f"\n🔄 Đang sinh dữ liệu {category}...")
        
        try:
            response = model.generate_content(prompt)
            response_text = response.text
            
            # Parse JSON
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                data = json.loads(json_str)
                
                # Lưu
                output_file = output_dir / category / f"generated_{category}.json"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                print(f"   ✅ Đã sinh {len(data)} {category}")
            
            time.sleep(3)
            
        except Exception as e:
            logger.error(f"Error generating {category}: {e}")
            print(f"   ❌ Lỗi: {e}")


def step4_evaluate_quality():
    """Bước 4: Đánh giá chất lượng dữ liệu"""
    print("\n" + "="*60)
    print("📌 BƯỚC 4: Đánh giá chất lượng dữ liệu")
    print("="*60)
    
    from src.evaluation import evaluate_data_quality
    
    try:
        report = evaluate_data_quality()
        return report
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        print(f"❌ Lỗi: {e}")
        return None


def main():
    """Chạy toàn bộ Phase 1"""
    print("\n" + "="*60)
    print("🚀 BẮT ĐẦU PHASE 1: THU THẬP VÀ XỬ LÝ DỮ LIỆU Y TẾ")
    print("="*60)
    print(f"⏰ Thời gian bắt đầu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Bước 1: Xử lý dữ liệu quốc tế
    step1_process_international_data()
    
    # Bước 2: Sinh Q&A với Gemini
    step2_generate_qa_with_gemini()
    
    # Bước 3: Sinh thêm dữ liệu
    step3_generate_more_data()
    
    # Bước 4: Đánh giá chất lượng
    step4_evaluate_quality()
    
    print("\n" + "="*60)
    print("✅ HOÀN THÀNH PHASE 1!")
    print("="*60)
    print(f"⏰ Thời gian kết thúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n📝 Các bước tiếp theo:")
    print("   1. Kiểm tra báo cáo trong reports/data_quality_report.md")
    print("   2. Nếu chưa đủ dữ liệu, chạy lại để sinh thêm")
    print("   3. Tiến hành Phase 2: Fine-tune mô hình")


if __name__ == "__main__":
    main()
