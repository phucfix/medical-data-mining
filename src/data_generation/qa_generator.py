"""
Sinh dữ liệu câu hỏi Đúng/Sai từ dữ liệu y tế sử dụng LLM
Hỗ trợ OpenAI GPT và Google Gemini
"""
import json
import os
import time
import random
from pathlib import Path
from typing import List, Dict, Optional, Generator
from tqdm import tqdm
from loguru import logger
from abc import ABC, abstractmethod
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import (
    OPENAI_API_KEY, GOOGLE_API_KEY, 
    RAW_DATA_DIR, PROCESSED_DATA_DIR, GENERATED_DATA_DIR
)


class BaseLLMGenerator(ABC):
    """Base class cho sinh dữ liệu với LLM"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.output_dir = GENERATED_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Sinh text từ prompt"""
        pass
    
    def generate_qa_from_medical_info(
        self, 
        medical_info: Dict,
        num_questions: int = 5
    ) -> List[Dict]:
        """Sinh câu hỏi Đúng/Sai từ thông tin y tế"""
        
        prompt = self._create_qa_prompt(medical_info, num_questions)
        
        try:
            response = self.generate(prompt)
            qa_pairs = self._parse_qa_response(response)
            return qa_pairs
        except Exception as e:
            logger.error(f"Error generating QA: {e}")
            return []
    
    def _create_qa_prompt(self, medical_info: Dict, num_questions: int) -> str:
        """Tạo prompt để sinh câu hỏi"""
        
        # Xây dựng context từ thông tin y tế
        context_parts = []
        
        if medical_info.get('name'):
            context_parts.append(f"Tên: {medical_info['name']}")
        
        if medical_info.get('description'):
            context_parts.append(f"Mô tả: {medical_info['description']}")
        
        if medical_info.get('symptoms'):
            if isinstance(medical_info['symptoms'], list):
                symptoms_text = ", ".join(medical_info['symptoms'])
            else:
                symptoms_text = medical_info['symptoms']
            context_parts.append(f"Triệu chứng: {symptoms_text}")
        
        if medical_info.get('causes'):
            if isinstance(medical_info['causes'], list):
                causes_text = ", ".join(medical_info['causes'])
            else:
                causes_text = medical_info['causes']
            context_parts.append(f"Nguyên nhân: {causes_text}")
        
        if medical_info.get('treatment'):
            context_parts.append(f"Điều trị: {medical_info['treatment']}")
        
        if medical_info.get('content'):
            # Giới hạn độ dài content
            content = medical_info['content'][:2000]
            context_parts.append(f"Nội dung chi tiết: {content}")
        
        context = "\n".join(context_parts)
        
        prompt = f"""Bạn là một chuyên gia y tế. Dựa trên thông tin y tế sau đây, hãy tạo {num_questions} câu hỏi Đúng/Sai (True/False) về y tế.

THÔNG TIN Y TẾ:
{context}

YÊU CẦU:
1. Tạo {num_questions} câu hỏi, trong đó khoảng một nửa là câu Đúng và một nửa là câu Sai
2. Câu hỏi phải rõ ràng, chính xác về mặt y học
3. Đối với câu Sai, hãy thay đổi một chi tiết quan trọng để câu trở thành sai
4. Mỗi câu hỏi phải có giải thích ngắn gọn

FORMAT OUTPUT (JSON):
[
    {{
        "question": "Câu hỏi ở đây",
        "answer": "Đúng" hoặc "Sai",
        "explanation": "Giải thích ngắn gọn"
    }},
    ...
]

Chỉ trả về JSON, không có text khác."""

        return prompt
    
    def _parse_qa_response(self, response: str) -> List[Dict]:
        """Parse response từ LLM thành list câu hỏi"""
        try:
            # Tìm JSON trong response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                logger.warning("Could not find JSON array in response")
                return []
            
            json_str = response[start_idx:end_idx]
            qa_pairs = json.loads(json_str)
            
            # Validate format
            valid_pairs = []
            for qa in qa_pairs:
                if all(k in qa for k in ['question', 'answer']):
                    # Chuẩn hóa answer
                    answer = qa['answer'].strip()
                    if answer.lower() in ['đúng', 'true', 'yes', 'có']:
                        qa['answer'] = 'Đúng'
                    else:
                        qa['answer'] = 'Sai'
                    
                    valid_pairs.append(qa)
            
            return valid_pairs
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return []


class OpenAIGenerator(BaseLLMGenerator):
    """Generator sử dụng OpenAI GPT"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        super().__init__(model_name)
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        except ImportError:
            logger.error("OpenAI package not installed. Run: pip install openai")
            raise
    
    def generate(self, prompt: str) -> str:
        """Sinh text với OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Bạn là một chuyên gia y tế Việt Nam, luôn trả lời bằng tiếng Việt."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class GeminiGenerator(BaseLLMGenerator):
    """Generator sử dụng Google Gemini"""
    
    def __init__(self, model_name: str = "gemini-pro"):
        super().__init__(model_name)
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)
            self.model = genai.GenerativeModel(model_name)
        except ImportError:
            logger.error("Google Generative AI package not installed. Run: pip install google-generativeai")
            raise
    
    def generate(self, prompt: str) -> str:
        """Sinh text với Gemini"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise


class LocalLLMGenerator(BaseLLMGenerator):
    """Generator sử dụng local LLM (Ollama hoặc HuggingFace)"""
    
    def __init__(self, model_name: str = "llama2"):
        super().__init__(model_name)
        self.api_url = "http://localhost:11434/api/generate"  # Ollama default
    
    def generate(self, prompt: str) -> str:
        """Sinh text với local LLM qua Ollama"""
        import requests
        
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return ""
                
        except Exception as e:
            logger.error(f"Local LLM error: {e}")
            raise


class MedicalQAGenerator:
    """
    Main class để sinh dữ liệu Q&A từ tất cả nguồn
    """
    
    def __init__(self, llm_type: str = "gemini"):
        """
        Args:
            llm_type: 'openai', 'gemini', hoặc 'local'
        """
        if llm_type == "openai":
            self.generator = OpenAIGenerator()
        elif llm_type == "gemini":
            self.generator = GeminiGenerator()
        elif llm_type == "local":
            self.generator = LocalLLMGenerator()
        else:
            raise ValueError(f"Unknown LLM type: {llm_type}")
        
        self.output_dir = GENERATED_DATA_DIR
        self.all_qa_pairs = []
    
    def load_raw_data(self) -> Generator[Dict, None, None]:
        """Load tất cả dữ liệu raw đã crawl"""
        
        # Load từ thư mục raw
        for category in ['diseases', 'symptoms', 'drugs']:
            category_dir = RAW_DATA_DIR / category
            
            if not category_dir.exists():
                continue
            
            for json_file in category_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list):
                        for item in data:
                            item['data_category'] = category
                            yield item
                    else:
                        data['data_category'] = category
                        yield data
                        
                except Exception as e:
                    logger.error(f"Error loading {json_file}: {e}")
    
    def generate_from_raw_data(
        self, 
        questions_per_item: int = 5,
        max_items: int = None,
        delay: float = 1.0
    ):
        """Sinh Q&A từ dữ liệu raw"""
        
        items = list(self.load_raw_data())
        
        if max_items:
            items = items[:max_items]
        
        logger.info(f"Generating QA from {len(items)} items")
        
        for item in tqdm(items, desc="Generating QA"):
            try:
                qa_pairs = self.generator.generate_qa_from_medical_info(
                    item, 
                    num_questions=questions_per_item
                )
                
                # Thêm metadata
                for qa in qa_pairs:
                    qa['source'] = item.get('source', 'unknown')
                    qa['source_url'] = item.get('source_url', '')
                    qa['category'] = item.get('data_category', '')
                    qa['related_entity'] = item.get('name', '')
                
                self.all_qa_pairs.extend(qa_pairs)
                
                # Rate limiting
                time.sleep(delay)
                
            except Exception as e:
                logger.error(f"Error generating QA for {item.get('name', 'unknown')}: {e}")
                continue
        
        logger.info(f"Generated {len(self.all_qa_pairs)} QA pairs")
    
    def generate_additional_qa(
        self,
        topics: List[str] = None,
        num_per_topic: int = 20
    ):
        """Sinh thêm câu hỏi theo chủ đề cụ thể"""
        
        if topics is None:
            topics = [
                "bệnh tim mạch",
                "bệnh tiểu đường", 
                "bệnh hô hấp",
                "bệnh tiêu hóa",
                "bệnh thần kinh",
                "thuốc kháng sinh",
                "thuốc giảm đau",
                "vaccine",
                "dinh dưỡng",
                "sức khỏe tâm thần"
            ]
        
        for topic in tqdm(topics, desc="Generating by topic"):
            prompt = f"""Hãy tạo {num_per_topic} câu hỏi Đúng/Sai về chủ đề "{topic}" trong y tế.

YÊU CẦU:
1. Câu hỏi phải chính xác về mặt y khoa
2. Khoảng một nửa câu Đúng và một nửa câu Sai
3. Câu hỏi đa dạng về độ khó
4. Mỗi câu có giải thích ngắn gọn

FORMAT OUTPUT (JSON):
[
    {{
        "question": "Câu hỏi ở đây",
        "answer": "Đúng" hoặc "Sai",
        "explanation": "Giải thích ngắn gọn"
    }},
    ...
]

Chỉ trả về JSON, không có text khác."""

            try:
                response = self.generator.generate(prompt)
                qa_pairs = self.generator._parse_qa_response(response)
                
                for qa in qa_pairs:
                    qa['source'] = 'llm_generated'
                    qa['category'] = topic
                
                self.all_qa_pairs.extend(qa_pairs)
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error generating for topic {topic}: {e}")
    
    def generate_from_templates(self, num_questions: int = 1000):
        """Sinh câu hỏi từ templates có sẵn"""
        
        templates = [
            # Templates về triệu chứng
            {
                "template": "{symptom} là triệu chứng phổ biến của bệnh {disease}.",
                "type": "symptom_disease",
                "variations": [
                    {"symptom": "Ho kéo dài trên 3 tuần", "disease": "lao phổi", "answer": "Đúng"},
                    {"symptom": "Đau đầu", "disease": "cảm cúm", "answer": "Đúng"},
                    {"symptom": "Sốt cao", "disease": "nhiễm trùng", "answer": "Đúng"},
                    {"symptom": "Khó thở", "disease": "hen suyễn", "answer": "Đúng"},
                    {"symptom": "Đau ngực", "disease": "nhồi máu cơ tim", "answer": "Đúng"},
                    {"symptom": "Vàng da", "disease": "viêm gan", "answer": "Đúng"},
                    {"symptom": "Tiêu chảy", "disease": "viêm ruột", "answer": "Đúng"},
                    {"symptom": "Mệt mỏi", "disease": "thiếu máu", "answer": "Đúng"},
                    {"symptom": "Đau bụng", "disease": "viêm dạ dày", "answer": "Đúng"},
                    {"symptom": "Ho ra máu", "disease": "ung thư phổi", "answer": "Đúng"},
                ]
            },
            # Templates về nguyên nhân
            {
                "template": "{disease} gây ra bởi {cause}.",
                "type": "cause_disease",
                "variations": [
                    {"disease": "Sỏi thận", "cause": "khoáng chất kết tụ trong nước tiểu", "answer": "Đúng"},
                    {"disease": "Tiểu đường type 2", "cause": "kháng insulin", "answer": "Đúng"},
                    {"disease": "Xơ gan", "cause": "uống rượu quá nhiều", "answer": "Đúng"},
                    {"disease": "Ung thư phổi", "cause": "hút thuốc lá", "answer": "Đúng"},
                    {"disease": "Cao huyết áp", "cause": "ăn mặn", "answer": "Đúng"},
                ]
            },
            # Templates về cơ chế
            {
                "template": "{disease} là tình trạng {mechanism}.",
                "type": "mechanism",
                "variations": [
                    {"disease": "Thoát vị đĩa đệm", "mechanism": "nhân nhầy đĩa đệm lồi ra chèn dây thần kinh", "answer": "Đúng"},
                    {"disease": "Động kinh", "mechanism": "các tế bào não hoạt động bất thường gây co giật", "answer": "Đúng"},
                    {"disease": "Đột quỵ", "mechanism": "mạch máu não bị tắc hoặc vỡ", "answer": "Đúng"},
                    {"disease": "Viêm khớp", "mechanism": "khớp bị viêm và đau", "answer": "Đúng"},
                ]
            },
            # Templates về thuốc
            {
                "template": "{drug} được sử dụng để điều trị {condition}.",
                "type": "drug_indication",
                "variations": [
                    {"drug": "Paracetamol", "condition": "sốt và đau", "answer": "Đúng"},
                    {"drug": "Amoxicillin", "condition": "nhiễm khuẩn", "answer": "Đúng"},
                    {"drug": "Metformin", "condition": "tiểu đường type 2", "answer": "Đúng"},
                    {"drug": "Omeprazole", "condition": "trào ngược dạ dày", "answer": "Đúng"},
                    {"drug": "Lisinopril", "condition": "cao huyết áp", "answer": "Đúng"},
                ]
            },
        ]
        
        generated = []
        
        for template_group in templates:
            for variation in template_group['variations']:
                # Tạo câu đúng
                question_true = template_group['template'].format(**variation)
                generated.append({
                    "question": question_true,
                    "answer": variation['answer'],
                    "explanation": f"Đây là thông tin y khoa chính xác.",
                    "source": "template_generated",
                    "category": template_group['type']
                })
                
                # Tạo câu sai bằng cách thay đổi một phần
                # (Trong thực tế cần logic phức tạp hơn)
        
        self.all_qa_pairs.extend(generated)
        logger.info(f"Generated {len(generated)} QA pairs from templates")
    
    def balance_dataset(self):
        """Cân bằng số lượng câu Đúng và Sai"""
        true_questions = [qa for qa in self.all_qa_pairs if qa['answer'] == 'Đúng']
        false_questions = [qa for qa in self.all_qa_pairs if qa['answer'] == 'Sai']
        
        logger.info(f"Before balancing: {len(true_questions)} True, {len(false_questions)} False")
        
        # Cân bằng bằng cách lấy số ít hơn
        min_count = min(len(true_questions), len(false_questions))
        
        if len(true_questions) > min_count:
            true_questions = random.sample(true_questions, min_count)
        if len(false_questions) > min_count:
            false_questions = random.sample(false_questions, min_count)
        
        self.all_qa_pairs = true_questions + false_questions
        random.shuffle(self.all_qa_pairs)
        
        logger.info(f"After balancing: {len(true_questions)} True, {len(false_questions)} False")
    
    def save_qa_data(self, filename: str = "medical_qa_dataset.json"):
        """Lưu dataset Q&A"""
        output_file = self.output_dir / filename
        
        # Thêm metadata
        dataset = {
            "metadata": {
                "total_questions": len(self.all_qa_pairs),
                "true_count": len([qa for qa in self.all_qa_pairs if qa['answer'] == 'Đúng']),
                "false_count": len([qa for qa in self.all_qa_pairs if qa['answer'] == 'Sai']),
                "generated_date": str(Path.ctime(Path.cwd())),
            },
            "data": self.all_qa_pairs
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(self.all_qa_pairs)} QA pairs to {output_file}")
        
        # Cũng lưu dạng CSV để dễ xem
        import csv
        csv_file = self.output_dir / filename.replace('.json', '.csv')
        
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['question', 'answer', 'explanation', 'source', 'category'])
            writer.writeheader()
            writer.writerows(self.all_qa_pairs)
        
        logger.info(f"Also saved to {csv_file}")


def generate_qa_data(
    llm_type: str = "gemini",
    questions_per_item: int = 5,
    max_items: int = 100,
    additional_topics: bool = True
):
    """Main function để sinh dữ liệu Q&A"""
    
    generator = MedicalQAGenerator(llm_type=llm_type)
    
    # Sinh từ templates (không cần API)
    generator.generate_from_templates()
    
    # Sinh từ dữ liệu raw đã crawl (cần API)
    try:
        generator.generate_from_raw_data(
            questions_per_item=questions_per_item,
            max_items=max_items
        )
    except Exception as e:
        logger.warning(f"Could not generate from raw data: {e}")
    
    # Sinh thêm theo topics (cần API)
    if additional_topics:
        try:
            generator.generate_additional_qa()
        except Exception as e:
            logger.warning(f"Could not generate additional topics: {e}")
    
    # Cân bằng và lưu
    generator.balance_dataset()
    generator.save_qa_data()
    
    return generator.all_qa_pairs


if __name__ == "__main__":
    # Test với số lượng nhỏ
    qa_data = generate_qa_data(
        llm_type="gemini",  # hoặc "openai", "local"
        questions_per_item=3,
        max_items=10,
        additional_topics=False
    )
    
    print(f"\nGenerated {len(qa_data)} QA pairs")
    print("\nSample:")
    for qa in qa_data[:3]:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print(f"Explanation: {qa.get('explanation', 'N/A')}")
        print("---")
