# 🏥 Vietnamese Medical Data Mining Project

## Giới thiệu
Project thu thập và khai phá dữ liệu y tế tiếng Việt, xây dựng mô hình ngôn ngữ nhỏ (SLM) để trả lời câu hỏi Đúng/Sai về y tế.

## 📁 Cấu trúc Project

```
medical-data-mining-project/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                    # Dữ liệu gốc từ crawl
│   │   ├── diseases/           # Dữ liệu bệnh
│   │   ├── symptoms/           # Dữ liệu triệu chứng
│   │   └── drugs/              # Dữ liệu thuốc
│   ├── external/               # Dữ liệu từ nguồn quốc tế (UMLS, ICD)
│   ├── processed/              # Dữ liệu đã xử lý
│   ├── generated/              # Dữ liệu sinh từ LLM
│   └── final/                  # Dữ liệu cuối cùng cho training
├── src/
│   ├── crawler/                # Code crawl dữ liệu
│   ├── preprocessing/          # Tiền xử lý
│   ├── data_generation/        # Sinh dữ liệu với LLM
│   ├── translation/            # Dịch dữ liệu quốc tế
│   └── evaluation/             # Đánh giá chất lượng
├── notebooks/                  # Jupyter notebooks phân tích
├── reports/                    # Báo cáo và slides
└── scripts/                    # Scripts chạy pipeline
```

## 🚀 Cài đặt

```bash
# Clone repository
git clone <repository-url>
cd medical-data-mining-project

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

## 📊 Nguồn dữ liệu

### Nguồn tiếng Việt
| Nguồn | URL | Loại dữ liệu | Độ tin cậy |
|-------|-----|--------------|------------|
| Vinmec | vinmec.com | Bệnh, triệu chứng | Cao |
| Medlatec | medlatec.vn | Bệnh, xét nghiệm | Cao |
| Drugbank VN | drugbank.vn | Thuốc | Cao |
| Wikipedia Y tế | vi.wikipedia.org | Tổng hợp | Trung bình |
| Bộ Y tế | moh.gov.vn | Chính sách, thuốc | Rất cao |

### Nguồn quốc tế (điểm cộng)
| Nguồn | Mô tả |
|-------|-------|
| UMLS | Hệ thống ngôn ngữ y tế thống nhất |
| ICD-10/11 | Phân loại bệnh quốc tế |
| MeSH | Medical Subject Headings |
| HPO | Human Phenotype Ontology |

## 🔧 Sử dụng

### 1. Thu thập dữ liệu
```bash
# Crawl dữ liệu từ các nguồn Việt Nam
python scripts/crawl_all.py

# Tải và xử lý dữ liệu quốc tế
python scripts/process_international_data.py
```

### 2. Tiền xử lý
```bash
python scripts/preprocess_all.py
```

### 3. Sinh dữ liệu Q&A
```bash
python scripts/generate_qa.py
```

### 4. Đánh giá chất lượng
```bash
python scripts/evaluate_quality.py
```

## 📈 Thống kê dữ liệu

| Loại | Số lượng | Nguồn |
|------|----------|-------|
| Bệnh | - | - |
| Triệu chứng | - | - |
| Thuốc | - | - |
| Câu hỏi Q&A | - | - |

## 👥 Thành viên nhóm

| STT | Họ tên | MSSV | Nhiệm vụ |
|-----|--------|------|----------|
| 1 | - | - | Thu thập dữ liệu |
| 2 | - | - | Tiền xử lý |
| 3 | - | - | Xây dựng mô hình |
| 4 | - | - | Đánh giá & Báo cáo |

## 📝 License
MIT License
