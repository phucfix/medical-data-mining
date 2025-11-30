"""
Đánh giá chất lượng dữ liệu y tế đã thu thập
Tạo báo cáo chi tiết về chất lượng dữ liệu
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter
import pandas as pd
from loguru import logger
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, GENERATED_DATA_DIR, 
    EXTERNAL_DATA_DIR, BASE_DIR
)


class DataQualityEvaluator:
    """Đánh giá chất lượng dữ liệu y tế"""
    
    def __init__(self):
        self.quality_report = {
            'overview': {},
            'by_category': {},
            'by_source': {},
            'issues': [],
            'recommendations': []
        }
    
    def load_all_data(self) -> Dict[str, List[Dict]]:
        """Load tất cả dữ liệu từ các nguồn"""
        data = {
            'raw': [],
            'processed': [],
            'generated': [],
            'external': []
        }
        
        # Raw data
        for category_dir in RAW_DATA_DIR.glob("*/"):
            for json_file in category_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)
                    if isinstance(file_data, list):
                        data['raw'].extend(file_data)
                    else:
                        data['raw'].append(file_data)
                except Exception as e:
                    logger.warning(f"Error loading {json_file}: {e}")
        
        # Processed data
        processed_file = PROCESSED_DATA_DIR / "all_processed_data.json"
        if processed_file.exists():
            try:
                with open(processed_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                if 'data' in loaded:
                    data['processed'] = loaded['data']
                else:
                    data['processed'] = loaded if isinstance(loaded, list) else [loaded]
            except Exception as e:
                logger.warning(f"Error loading processed data: {e}")
        
        # Generated QA data
        for json_file in GENERATED_DATA_DIR.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                if 'data' in file_data:
                    data['generated'].extend(file_data['data'])
                elif isinstance(file_data, list):
                    data['generated'].extend(file_data)
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
        
        # External data
        for json_file in EXTERNAL_DATA_DIR.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                if isinstance(file_data, list):
                    data['external'].extend(file_data)
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
        
        return data
    
    def check_completeness(self, item: Dict, required_fields: List[str]) -> Dict:
        """Kiểm tra tính đầy đủ của một record"""
        result = {
            'complete': True,
            'missing_fields': [],
            'empty_fields': []
        }
        
        for field in required_fields:
            if field not in item:
                result['missing_fields'].append(field)
                result['complete'] = False
            elif not item[field]:
                result['empty_fields'].append(field)
                result['complete'] = False
        
        return result
    
    def check_text_quality(self, text: str) -> Dict:
        """Kiểm tra chất lượng của text"""
        if not text:
            return {'quality': 'empty', 'score': 0, 'issues': ['Empty text']}
        
        issues = []
        score = 100
        
        # Kiểm tra độ dài
        if len(text) < 20:
            issues.append('Too short')
            score -= 20
        
        # Kiểm tra HTML còn sót
        if re.search(r'<[^>]+>', text):
            issues.append('Contains HTML')
            score -= 15
        
        # Kiểm tra URL
        if re.search(r'http[s]?://', text):
            issues.append('Contains URLs')
            score -= 10
        
        # Kiểm tra ký tự lạ
        special_chars = len(re.findall(r'[^\w\sàáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ.,;:?!-]', text, re.UNICODE))
        if special_chars > len(text) * 0.1:
            issues.append('Many special characters')
            score -= 15
        
        # Kiểm tra lặp từ
        words = text.lower().split()
        if words:
            word_counts = Counter(words)
            most_common = word_counts.most_common(1)[0]
            if most_common[1] > len(words) * 0.3:
                issues.append('Repetitive content')
                score -= 20
        
        # Xác định quality level
        if score >= 80:
            quality = 'high'
        elif score >= 60:
            quality = 'medium'
        elif score >= 40:
            quality = 'low'
        else:
            quality = 'poor'
        
        return {
            'quality': quality,
            'score': max(0, score),
            'issues': issues,
            'length': len(text)
        }
    
    def evaluate_medical_content(self, text: str) -> Dict:
        """Đánh giá nội dung y tế"""
        if not text:
            return {'is_medical': False, 'confidence': 0}
        
        text_lower = text.lower()
        
        # Từ khóa y tế
        medical_keywords = [
            'bệnh', 'triệu chứng', 'thuốc', 'điều trị', 'chẩn đoán',
            'nguyên nhân', 'phòng ngừa', 'xét nghiệm', 'virus', 'vi khuẩn',
            'viêm', 'nhiễm', 'đau', 'sốt', 'ho', 'ung thư', 'tim mạch',
            'huyết áp', 'đường huyết', 'vaccine', 'kháng sinh', 'dị ứng',
            'phẫu thuật', 'liều dùng', 'tác dụng phụ', 'chống chỉ định'
        ]
        
        # Đếm từ khóa y tế
        keyword_count = sum(1 for kw in medical_keywords if kw in text_lower)
        
        # Tính confidence
        confidence = min(100, keyword_count * 10)
        
        return {
            'is_medical': keyword_count >= 2,
            'confidence': confidence,
            'keyword_count': keyword_count
        }
    
    def evaluate_qa_quality(self, qa_data: List[Dict]) -> Dict:
        """Đánh giá chất lượng dữ liệu Q&A"""
        if not qa_data:
            return {'total': 0, 'quality': 'N/A'}
        
        stats = {
            'total': len(qa_data),
            'true_count': 0,
            'false_count': 0,
            'with_explanation': 0,
            'valid_format': 0,
            'avg_question_length': 0,
            'quality_scores': []
        }
        
        question_lengths = []
        
        for qa in qa_data:
            # Đếm True/False
            answer = qa.get('answer', '').strip()
            if answer in ['Đúng', 'True', 'đúng', 'true']:
                stats['true_count'] += 1
            elif answer in ['Sai', 'False', 'sai', 'false']:
                stats['false_count'] += 1
            
            # Có explanation?
            if qa.get('explanation'):
                stats['with_explanation'] += 1
            
            # Format hợp lệ?
            if qa.get('question') and qa.get('answer'):
                stats['valid_format'] += 1
            
            # Độ dài câu hỏi
            question = qa.get('question', '')
            if question:
                question_lengths.append(len(question))
                
                # Đánh giá chất lượng câu hỏi
                q_quality = self.check_text_quality(question)
                stats['quality_scores'].append(q_quality['score'])
        
        if question_lengths:
            stats['avg_question_length'] = sum(question_lengths) / len(question_lengths)
        
        if stats['quality_scores']:
            stats['avg_quality_score'] = sum(stats['quality_scores']) / len(stats['quality_scores'])
        
        # Balance ratio
        if stats['true_count'] + stats['false_count'] > 0:
            balance = min(stats['true_count'], stats['false_count']) / max(stats['true_count'], stats['false_count'])
            stats['balance_ratio'] = round(balance, 2)
        
        return stats
    
    def generate_report(self) -> Dict:
        """Tạo báo cáo đầy đủ về chất lượng dữ liệu"""
        logger.info("Loading all data...")
        all_data = self.load_all_data()
        
        # Overview statistics
        self.quality_report['overview'] = {
            'total_raw': len(all_data['raw']),
            'total_processed': len(all_data['processed']),
            'total_generated_qa': len(all_data['generated']),
            'total_external': len(all_data['external']),
            'total_all': sum(len(v) for v in all_data.values()),
            'evaluation_date': datetime.now().isoformat()
        }
        
        # Evaluate raw data quality
        logger.info("Evaluating raw data quality...")
        raw_quality = {
            'high': 0, 'medium': 0, 'low': 0, 'poor': 0,
            'medical_content': 0,
            'issues': Counter()
        }
        
        for item in all_data['raw']:
            # Check main content
            content = item.get('content', '') or item.get('description', '')
            quality = self.check_text_quality(content)
            raw_quality[quality['quality']] += 1
            
            for issue in quality['issues']:
                raw_quality['issues'][issue] += 1
            
            # Check medical relevance
            medical = self.evaluate_medical_content(content)
            if medical['is_medical']:
                raw_quality['medical_content'] += 1
        
        raw_quality['issues'] = dict(raw_quality['issues'].most_common(10))
        self.quality_report['raw_data_quality'] = raw_quality
        
        # Evaluate QA data
        logger.info("Evaluating QA data quality...")
        qa_stats = self.evaluate_qa_quality(all_data['generated'])
        self.quality_report['qa_data_quality'] = qa_stats
        
        # Evaluate by source
        logger.info("Evaluating by source...")
        source_stats = {}
        for item in all_data['raw']:
            source = item.get('source', 'unknown')
            if source not in source_stats:
                source_stats[source] = {'count': 0, 'high_quality': 0}
            source_stats[source]['count'] += 1
            
            content = item.get('content', '') or item.get('description', '')
            if self.check_text_quality(content)['score'] >= 70:
                source_stats[source]['high_quality'] += 1
        
        # Calculate quality percentage per source
        for source, stats in source_stats.items():
            if stats['count'] > 0:
                stats['quality_percent'] = round(stats['high_quality'] / stats['count'] * 100, 1)
        
        self.quality_report['by_source'] = source_stats
        
        # Evaluate external data
        logger.info("Evaluating external data...")
        external_stats = {
            'total': len(all_data['external']),
            'with_vietnamese': 0,
            'sources': Counter()
        }
        
        for item in all_data['external']:
            if item.get('term_vi') or item.get('name_vi') or item.get('definition_vi'):
                external_stats['with_vietnamese'] += 1
            external_stats['sources'][item.get('source', 'unknown')] += 1
        
        external_stats['sources'] = dict(external_stats['sources'])
        self.quality_report['external_data'] = external_stats
        
        # Identify issues and recommendations
        self._identify_issues()
        self._generate_recommendations()
        
        return self.quality_report
    
    def _identify_issues(self):
        """Xác định các vấn đề về chất lượng"""
        issues = []
        
        # Check data volume
        total = self.quality_report['overview']['total_all']
        if total < 50000:
            issues.append({
                'type': 'volume',
                'severity': 'high',
                'message': f'Chưa đủ 50,000 dữ liệu (hiện có {total})',
                'suggestion': 'Cần crawl thêm dữ liệu hoặc sinh thêm từ LLM'
            })
        
        # Check quality distribution
        raw_quality = self.quality_report.get('raw_data_quality', {})
        poor_ratio = raw_quality.get('poor', 0) + raw_quality.get('low', 0)
        total_raw = self.quality_report['overview']['total_raw']
        
        if total_raw > 0 and poor_ratio / total_raw > 0.3:
            issues.append({
                'type': 'quality',
                'severity': 'medium',
                'message': f'{poor_ratio} records có chất lượng thấp ({poor_ratio/total_raw*100:.1f}%)',
                'suggestion': 'Cần cải thiện pipeline tiền xử lý'
            })
        
        # Check QA balance
        qa_stats = self.quality_report.get('qa_data_quality', {})
        if qa_stats.get('balance_ratio', 1) < 0.7:
            issues.append({
                'type': 'balance',
                'severity': 'medium',
                'message': 'Dữ liệu Q&A không cân bằng giữa Đúng/Sai',
                'suggestion': 'Cần sinh thêm câu hỏi cho class thiếu'
            })
        
        self.quality_report['issues'] = issues
    
    def _generate_recommendations(self):
        """Sinh các khuyến nghị cải thiện"""
        recommendations = []
        
        # Based on overview
        overview = self.quality_report['overview']
        
        if overview['total_external'] < 100:
            recommendations.append(
                "Nên tăng cường sử dụng nguồn dữ liệu quốc tế (UMLS, ICD-10, MeSH) để được điểm cộng"
            )
        
        if overview['total_generated_qa'] < 10000:
            recommendations.append(
                "Nên sinh thêm câu hỏi Q&A để đạt yêu cầu số lượng"
            )
        
        # Based on source quality
        source_stats = self.quality_report.get('by_source', {})
        low_quality_sources = [
            src for src, stats in source_stats.items() 
            if stats.get('quality_percent', 0) < 50
        ]
        
        if low_quality_sources:
            recommendations.append(
                f"Cần cải thiện xử lý dữ liệu từ các nguồn: {', '.join(low_quality_sources)}"
            )
        
        self.quality_report['recommendations'] = recommendations
    
    def save_report(self, filename: str = "data_quality_report.json"):
        """Lưu báo cáo"""
        output_file = BASE_DIR / "reports" / filename
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.quality_report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Report saved to {output_file}")
        
        # Cũng tạo report markdown
        md_report = self._generate_markdown_report()
        md_file = output_file.with_suffix('.md')
        
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_report)
        
        logger.info(f"Markdown report saved to {md_file}")
    
    def _generate_markdown_report(self) -> str:
        """Tạo báo cáo dạng Markdown"""
        report = []
        report.append("# 📊 Báo cáo Chất lượng Dữ liệu Y tế\n")
        report.append(f"*Ngày đánh giá: {self.quality_report['overview'].get('evaluation_date', 'N/A')}*\n")
        
        # Overview
        report.append("## 1. Tổng quan\n")
        overview = self.quality_report['overview']
        report.append("| Loại dữ liệu | Số lượng |")
        report.append("|-------------|----------|")
        report.append(f"| Dữ liệu thô (raw) | {overview.get('total_raw', 0):,} |")
        report.append(f"| Dữ liệu đã xử lý | {overview.get('total_processed', 0):,} |")
        report.append(f"| Câu hỏi Q&A | {overview.get('total_generated_qa', 0):,} |")
        report.append(f"| Dữ liệu quốc tế | {overview.get('total_external', 0):,} |")
        report.append(f"| **Tổng cộng** | **{overview.get('total_all', 0):,}** |")
        report.append("")
        
        # Target check
        total = overview.get('total_all', 0)
        if total >= 50000:
            report.append(f"✅ **Đạt yêu cầu tối thiểu 50,000 dữ liệu**\n")
        else:
            report.append(f"⚠️ **Chưa đạt yêu cầu: cần thêm {50000 - total:,} dữ liệu**\n")
        
        # Raw data quality
        report.append("## 2. Chất lượng dữ liệu thô\n")
        raw_quality = self.quality_report.get('raw_data_quality', {})
        
        report.append("| Mức chất lượng | Số lượng |")
        report.append("|---------------|----------|")
        report.append(f"| 🟢 Cao (High) | {raw_quality.get('high', 0):,} |")
        report.append(f"| 🟡 Trung bình (Medium) | {raw_quality.get('medium', 0):,} |")
        report.append(f"| 🟠 Thấp (Low) | {raw_quality.get('low', 0):,} |")
        report.append(f"| 🔴 Kém (Poor) | {raw_quality.get('poor', 0):,} |")
        report.append("")
        
        # Common issues
        if raw_quality.get('issues'):
            report.append("### Các vấn đề phổ biến:\n")
            for issue, count in raw_quality['issues'].items():
                report.append(f"- {issue}: {count} records")
            report.append("")
        
        # QA quality
        report.append("## 3. Chất lượng dữ liệu Q&A\n")
        qa_stats = self.quality_report.get('qa_data_quality', {})
        
        if qa_stats.get('total', 0) > 0:
            report.append(f"- Tổng số câu hỏi: **{qa_stats.get('total', 0):,}**")
            report.append(f"- Câu Đúng: {qa_stats.get('true_count', 0):,}")
            report.append(f"- Câu Sai: {qa_stats.get('false_count', 0):,}")
            report.append(f"- Có giải thích: {qa_stats.get('with_explanation', 0):,}")
            report.append(f"- Tỷ lệ cân bằng: {qa_stats.get('balance_ratio', 'N/A')}")
            report.append(f"- Điểm chất lượng TB: {qa_stats.get('avg_quality_score', 0):.1f}/100")
        else:
            report.append("*Chưa có dữ liệu Q&A*")
        report.append("")
        
        # By source
        report.append("## 4. Thống kê theo nguồn\n")
        source_stats = self.quality_report.get('by_source', {})
        
        if source_stats:
            report.append("| Nguồn | Số lượng | Chất lượng cao |")
            report.append("|-------|----------|----------------|")
            for source, stats in source_stats.items():
                report.append(f"| {source} | {stats['count']:,} | {stats.get('quality_percent', 0)}% |")
        report.append("")
        
        # External data
        report.append("## 5. Dữ liệu quốc tế (Điểm cộng)\n")
        external = self.quality_report.get('external_data', {})
        
        if external.get('total', 0) > 0:
            report.append(f"- Tổng số: {external.get('total', 0):,}")
            report.append(f"- Đã dịch sang tiếng Việt: {external.get('with_vietnamese', 0):,}")
            report.append("\nNguồn:")
            for src, count in external.get('sources', {}).items():
                report.append(f"- {src}: {count:,}")
        else:
            report.append("*Chưa có dữ liệu từ nguồn quốc tế*")
        report.append("")
        
        # Issues
        report.append("## 6. Các vấn đề cần giải quyết\n")
        issues = self.quality_report.get('issues', [])
        
        if issues:
            for issue in issues:
                severity_icon = "🔴" if issue['severity'] == 'high' else "🟡"
                report.append(f"{severity_icon} **{issue['message']}**")
                report.append(f"   - Gợi ý: {issue['suggestion']}\n")
        else:
            report.append("✅ Không phát hiện vấn đề nghiêm trọng\n")
        
        # Recommendations
        report.append("## 7. Khuyến nghị\n")
        recommendations = self.quality_report.get('recommendations', [])
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                report.append(f"{i}. {rec}")
        else:
            report.append("Không có khuyến nghị bổ sung.")
        
        return "\n".join(report)
    
    def print_summary(self):
        """In tóm tắt ra console"""
        print("\n" + "="*60)
        print("📊 BÁO CÁO CHẤT LƯỢNG DỮ LIỆU Y TẾ")
        print("="*60)
        
        overview = self.quality_report['overview']
        print(f"\n📈 TỔNG QUAN:")
        print(f"   - Dữ liệu thô: {overview.get('total_raw', 0):,}")
        print(f"   - Đã xử lý: {overview.get('total_processed', 0):,}")
        print(f"   - Câu hỏi Q&A: {overview.get('total_generated_qa', 0):,}")
        print(f"   - Dữ liệu quốc tế: {overview.get('total_external', 0):,}")
        print(f"   - TỔNG: {overview.get('total_all', 0):,}")
        
        total = overview.get('total_all', 0)
        if total >= 50000:
            print(f"\n   ✅ Đạt yêu cầu tối thiểu 50,000")
        else:
            print(f"\n   ⚠️ Cần thêm {50000 - total:,} dữ liệu")
        
        # Issues
        issues = self.quality_report.get('issues', [])
        if issues:
            print(f"\n⚠️ CÁC VẤN ĐỀ ({len(issues)}):")
            for issue in issues:
                print(f"   - {issue['message']}")
        
        print("\n" + "="*60)


def evaluate_data_quality():
    """Main function để đánh giá chất lượng"""
    evaluator = DataQualityEvaluator()
    
    # Generate report
    report = evaluator.generate_report()
    
    # Save report
    evaluator.save_report()
    
    # Print summary
    evaluator.print_summary()
    
    return report


if __name__ == "__main__":
    evaluate_data_quality()
