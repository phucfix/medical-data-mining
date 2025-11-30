"""
Pipeline tiền xử lý dữ liệu y tế tiếng Việt
Bao gồm: làm sạch, chuẩn hóa, trích xuất entities
"""
import re
import json
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
import pandas as pd
from tqdm import tqdm
from loguru import logger
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR


class VietnameseMedicalCleaner:
    """Làm sạch và chuẩn hóa văn bản y tế tiếng Việt"""
    
    def __init__(self):
        # Các ký tự cần loại bỏ
        self.noise_patterns = [
            r'<[^>]+>',  # HTML tags
            r'http[s]?://\S+',  # URLs
            r'\S+@\S+',  # Emails
            r'\d{10,11}',  # Số điện thoại
            r'[►▶●■□▪▫•‣⁃]',  # Bullet points
            r'\s+',  # Multiple whitespace
        ]
        
        # Stopwords tiếng Việt cho y tế
        self.medical_stopwords = {
            'và', 'hoặc', 'của', 'là', 'được', 'có', 'trong', 'với',
            'này', 'đó', 'những', 'các', 'một', 'để', 'cho', 'từ',
            'như', 'khi', 'nếu', 'thì', 'mà', 'cũng', 'đã', 'sẽ',
            'bị', 'do', 'vì', 'nên', 'hay', 'còn', 'theo', 'về'
        }
        
        # Từ viết tắt y tế phổ biến
        self.medical_abbreviations = {
            'bs': 'bác sĩ',
            'bn': 'bệnh nhân',
            'ths': 'thạc sĩ',
            'pgs': 'phó giáo sư',
            'gs': 'giáo sư',
            'ts': 'tiến sĩ',
            'bv': 'bệnh viện',
            'pk': 'phòng khám',
            'xn': 'xét nghiệm',
            'ha': 'huyết áp',
            'đh': 'đường huyết',
            'tim': 'tim mạch',
        }
    
    def normalize_unicode(self, text: str) -> str:
        """Chuẩn hóa Unicode tiếng Việt"""
        # Chuẩn hóa NFC
        text = unicodedata.normalize('NFC', text)
        
        # Thay thế các ký tự đặc biệt
        replacements = {
            'à': 'à', 'á': 'á', 'ả': 'ả', 'ã': 'ã', 'ạ': 'ạ',
            'ă': 'ă', 'ằ': 'ằ', 'ắ': 'ắ', 'ẳ': 'ẳ', 'ẵ': 'ẵ', 'ặ': 'ặ',
            'â': 'â', 'ầ': 'ầ', 'ấ': 'ấ', 'ẩ': 'ẩ', 'ẫ': 'ẫ', 'ậ': 'ậ',
            'đ': 'đ',
            'è': 'è', 'é': 'é', 'ẻ': 'ẻ', 'ẽ': 'ẽ', 'ẹ': 'ẹ',
            'ê': 'ê', 'ề': 'ề', 'ế': 'ế', 'ể': 'ể', 'ễ': 'ễ', 'ệ': 'ệ',
            'ì': 'ì', 'í': 'í', 'ỉ': 'ỉ', 'ĩ': 'ĩ', 'ị': 'ị',
            'ò': 'ò', 'ó': 'ó', 'ỏ': 'ỏ', 'õ': 'õ', 'ọ': 'ọ',
            'ô': 'ô', 'ồ': 'ồ', 'ố': 'ố', 'ổ': 'ổ', 'ỗ': 'ỗ', 'ộ': 'ộ',
            'ơ': 'ơ', 'ờ': 'ờ', 'ớ': 'ớ', 'ở': 'ở', 'ỡ': 'ỡ', 'ợ': 'ợ',
            'ù': 'ù', 'ú': 'ú', 'ủ': 'ủ', 'ũ': 'ũ', 'ụ': 'ụ',
            'ư': 'ư', 'ừ': 'ừ', 'ứ': 'ứ', 'ử': 'ử', 'ữ': 'ữ', 'ự': 'ự',
            'ỳ': 'ỳ', 'ý': 'ý', 'ỷ': 'ỷ', 'ỹ': 'ỹ', 'ỵ': 'ỵ',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def remove_noise(self, text: str) -> str:
        """Loại bỏ nhiễu từ văn bản"""
        if not text:
            return ""
        
        for pattern in self.noise_patterns:
            text = re.sub(pattern, ' ', text)
        
        # Loại bỏ khoảng trắng thừa
        text = ' '.join(text.split())
        
        return text.strip()
    
    def expand_abbreviations(self, text: str) -> str:
        """Mở rộng các từ viết tắt y tế"""
        words = text.lower().split()
        expanded = []
        
        for word in words:
            if word in self.medical_abbreviations:
                expanded.append(self.medical_abbreviations[word])
            else:
                expanded.append(word)
        
        return ' '.join(expanded)
    
    def clean_text(self, text: str) -> str:
        """Pipeline làm sạch đầy đủ"""
        if not text:
            return ""
        
        # 1. Chuẩn hóa Unicode
        text = self.normalize_unicode(text)
        
        # 2. Loại bỏ nhiễu
        text = self.remove_noise(text)
        
        # 3. Mở rộng viết tắt
        text = self.expand_abbreviations(text)
        
        # 4. Loại bỏ ký tự đặc biệt nhưng giữ dấu tiếng Việt
        text = re.sub(r'[^\w\sàáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ.,;:?!-]', '', text, flags=re.UNICODE)
        
        return text.strip()


class MedicalEntityExtractor:
    """Trích xuất thực thể y tế từ văn bản"""
    
    def __init__(self):
        # Từ khóa cho từng loại entity
        self.disease_keywords = [
            'bệnh', 'hội chứng', 'rối loạn', 'viêm', 'ung thư', 
            'nhiễm', 'suy', 'loạn', 'thoái hóa', 'u', 'nhồi máu'
        ]
        
        self.symptom_keywords = [
            'triệu chứng', 'dấu hiệu', 'biểu hiện', 'đau', 'sốt',
            'ho', 'khó thở', 'mệt mỏi', 'buồn nôn', 'chóng mặt',
            'ngứa', 'sưng', 'đỏ', 'viêm', 'chảy máu'
        ]
        
        self.drug_keywords = [
            'thuốc', 'viên', 'mg', 'ml', 'liều', 'dùng',
            'kháng sinh', 'giảm đau', 'hạ sốt', 'vitamin'
        ]
        
        # Load medical terms nếu có
        self.medical_terms = self._load_medical_terms()
    
    def _load_medical_terms(self) -> Dict[str, List[str]]:
        """Load danh sách thuật ngữ y tế"""
        terms = {
            'diseases': [],
            'symptoms': [],
            'drugs': []
        }
        
        # Load từ external data nếu có
        for category in terms.keys():
            file_path = EXTERNAL_DATA_DIR / f"international_{category}.json"
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    for item in data:
                        if 'term_vi' in item:
                            terms[category].append(item['term_vi'].lower())
                        if 'name_vi' in item:
                            terms[category].append(item['name_vi'].lower())
                            
                except Exception as e:
                    logger.warning(f"Could not load {file_path}: {e}")
        
        return terms
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Trích xuất tất cả entities từ text"""
        text_lower = text.lower()
        
        entities = {
            'diseases': self._extract_by_keywords(text_lower, self.disease_keywords, self.medical_terms.get('diseases', [])),
            'symptoms': self._extract_by_keywords(text_lower, self.symptom_keywords, self.medical_terms.get('symptoms', [])),
            'drugs': self._extract_by_keywords(text_lower, self.drug_keywords, self.medical_terms.get('drugs', []))
        }
        
        return entities
    
    def _extract_by_keywords(
        self, 
        text: str, 
        keywords: List[str],
        known_terms: List[str]
    ) -> List[str]:
        """Trích xuất entities dựa trên keywords"""
        extracted = set()
        
        # Tìm theo keywords
        for keyword in keywords:
            # Tìm cụm từ chứa keyword
            pattern = rf'\b\w*{keyword}\w*\b'
            matches = re.findall(pattern, text)
            extracted.update(matches)
        
        # Tìm theo known terms
        for term in known_terms:
            if term in text:
                extracted.add(term)
        
        return list(extracted)


class DataPreprocessor:
    """Main class để tiền xử lý toàn bộ dữ liệu"""
    
    def __init__(self):
        self.cleaner = VietnameseMedicalCleaner()
        self.extractor = MedicalEntityExtractor()
        self.processed_data = []
        self.stats = {
            'total_raw': 0,
            'total_processed': 0,
            'duplicates_removed': 0,
            'empty_removed': 0
        }
    
    def load_all_raw_data(self) -> List[Dict]:
        """Load tất cả dữ liệu raw"""
        all_data = []
        
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
                            item['original_category'] = category
                            item['source_file'] = str(json_file)
                            all_data.append(item)
                    else:
                        data['original_category'] = category
                        data['source_file'] = str(json_file)
                        all_data.append(data)
                        
                except Exception as e:
                    logger.error(f"Error loading {json_file}: {e}")
        
        # Load external data
        for json_file in EXTERNAL_DATA_DIR.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    all_data.extend(data)
                        
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
        
        self.stats['total_raw'] = len(all_data)
        logger.info(f"Loaded {len(all_data)} raw records")
        
        return all_data
    
    def preprocess_item(self, item: Dict) -> Optional[Dict]:
        """Tiền xử lý một item"""
        processed = {}
        
        # Copy metadata
        processed['source'] = item.get('source', 'unknown')
        processed['source_url'] = item.get('source_url', '')
        processed['category'] = item.get('original_category', item.get('category', ''))
        
        # Clean các trường text
        text_fields = ['name', 'description', 'content', 'symptoms', 'causes', 'treatment']
        
        for field in text_fields:
            if field in item:
                value = item[field]
                
                if isinstance(value, list):
                    cleaned = [self.cleaner.clean_text(v) for v in value if v]
                    processed[field] = [v for v in cleaned if v]  # Remove empty
                elif isinstance(value, str):
                    processed[field] = self.cleaner.clean_text(value)
                else:
                    processed[field] = value
        
        # Xử lý trường đặc biệt cho dữ liệu quốc tế
        if 'name_vi' in item:
            processed['name'] = self.cleaner.clean_text(item['name_vi'])
        if 'term_vi' in item:
            processed['name'] = self.cleaner.clean_text(item['term_vi'])
        if 'definition_vi' in item:
            processed['description'] = self.cleaner.clean_text(item['definition_vi'])
        
        # Tạo combined text cho search/embedding
        text_parts = []
        for field in ['name', 'description', 'content']:
            if field in processed and processed[field]:
                if isinstance(processed[field], list):
                    text_parts.extend(processed[field])
                else:
                    text_parts.append(processed[field])
        
        processed['combined_text'] = ' '.join(text_parts)
        
        # Kiểm tra empty
        if not processed.get('name') and not processed.get('combined_text'):
            return None
        
        # Trích xuất entities
        if processed.get('combined_text'):
            processed['extracted_entities'] = self.extractor.extract_entities(
                processed['combined_text']
            )
        
        return processed
    
    def remove_duplicates(self, data: List[Dict]) -> List[Dict]:
        """Loại bỏ duplicates dựa trên name và content"""
        seen = set()
        unique_data = []
        
        for item in data:
            # Tạo key unique
            key_parts = [
                item.get('name', ''),
                item.get('description', '')[:100] if item.get('description') else ''
            ]
            key = '|'.join(str(p).lower() for p in key_parts)
            
            if key and key not in seen:
                seen.add(key)
                unique_data.append(item)
        
        duplicates_removed = len(data) - len(unique_data)
        self.stats['duplicates_removed'] = duplicates_removed
        logger.info(f"Removed {duplicates_removed} duplicates")
        
        return unique_data
    
    def process_all(self) -> List[Dict]:
        """Chạy toàn bộ pipeline tiền xử lý"""
        # Load raw data
        raw_data = self.load_all_raw_data()
        
        # Process each item
        processed = []
        empty_count = 0
        
        for item in tqdm(raw_data, desc="Preprocessing"):
            result = self.preprocess_item(item)
            if result:
                processed.append(result)
            else:
                empty_count += 1
        
        self.stats['empty_removed'] = empty_count
        logger.info(f"Removed {empty_count} empty records")
        
        # Remove duplicates
        processed = self.remove_duplicates(processed)
        
        self.stats['total_processed'] = len(processed)
        self.processed_data = processed
        
        return processed
    
    def save_processed_data(self):
        """Lưu dữ liệu đã xử lý"""
        # Lưu theo category
        categories = {}
        for item in self.processed_data:
            cat = item.get('category', 'other')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(item)
        
        for category, data in categories.items():
            output_file = PROCESSED_DATA_DIR / category / "processed_data.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {len(data)} {category} records to {output_file}")
        
        # Lưu tất cả
        all_output = PROCESSED_DATA_DIR / "all_processed_data.json"
        with open(all_output, 'w', encoding='utf-8') as f:
            json.dump({
                'stats': self.stats,
                'data': self.processed_data
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved all {len(self.processed_data)} records to {all_output}")
    
    def get_statistics(self) -> Dict:
        """Trả về thống kê về dữ liệu"""
        stats = self.stats.copy()
        
        # Thống kê theo category
        category_counts = {}
        for item in self.processed_data:
            cat = item.get('category', 'other')
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        stats['by_category'] = category_counts
        
        # Thống kê theo source
        source_counts = {}
        for item in self.processed_data:
            src = item.get('source', 'unknown')
            source_counts[src] = source_counts.get(src, 0) + 1
        
        stats['by_source'] = source_counts
        
        # Thống kê text length
        text_lengths = [len(item.get('combined_text', '')) for item in self.processed_data]
        if text_lengths:
            stats['avg_text_length'] = sum(text_lengths) / len(text_lengths)
            stats['min_text_length'] = min(text_lengths)
            stats['max_text_length'] = max(text_lengths)
        
        return stats


def preprocess_all_data():
    """Main function để chạy tiền xử lý"""
    preprocessor = DataPreprocessor()
    
    # Process
    preprocessor.process_all()
    
    # Save
    preprocessor.save_processed_data()
    
    # Print statistics
    stats = preprocessor.get_statistics()
    
    print("\n" + "="*50)
    print("PREPROCESSING STATISTICS")
    print("="*50)
    print(f"Total raw records: {stats['total_raw']}")
    print(f"Total processed: {stats['total_processed']}")
    print(f"Duplicates removed: {stats['duplicates_removed']}")
    print(f"Empty removed: {stats['empty_removed']}")
    print(f"\nBy category:")
    for cat, count in stats.get('by_category', {}).items():
        print(f"  - {cat}: {count}")
    print(f"\nBy source:")
    for src, count in stats.get('by_source', {}).items():
        print(f"  - {src}: {count}")
    print("="*50)
    
    return preprocessor.processed_data


if __name__ == "__main__":
    preprocess_all_data()
