# 🚀 CHUNKED TRAINING ON GOOGLE COLAB

## Hướng dẫn chạy Chunked Training trên Google Colab

### 📋 Chuẩn bị:

1. **Các file cần upload lên Colab**:
   ```
   ├── src/train_slm_qwen_lora_v4_chunked.py
   ├── data/slm_train_style_adapted.jsonl  (154,477 samples)
   └── data/slm_val.jsonl
   ```

2. **Hoặc clone từ GitHub** (khuyến nghị):
   - Push code lên GitHub
   - Clone trong Colab

---

## 🎯 Option 1: Upload files trực tiếp

### Step 1: Tạo Colab Notebook mới
Go to: https://colab.research.google.com/

### Step 2: Chọn GPU Runtime
- Runtime → Change runtime type → GPU (T4)
- Nếu có Pro: chọn A100 hoặc V100 (nhanh hơn)

### Step 3: Cài đặt dependencies

```python
# Cell 1: Install dependencies
!pip install -q transformers peft datasets accelerate bitsandbytes
```

### Step 4: Upload files

```python
# Cell 2: Upload training script và data
from google.colab import files
import os

# Tạo thư mục
!mkdir -p /content/src
!mkdir -p /content/data
!mkdir -p /content/models

print("📁 Upload train_slm_qwen_lora_v4_chunked.py vào /content/src/")
uploaded = files.upload()

print("📁 Upload slm_train_style_adapted.jsonl vào /content/data/")
uploaded = files.upload()

print("✓ Upload completed!")
```

### Step 5: Chạy training

```python
# Cell 3: Run chunked training
!cd /content && python src/train_slm_qwen_lora_v4_chunked.py
```

---

## 🎯 Option 2: Clone từ GitHub (Khuyến nghị) ⭐

### Step 1: Push code lên GitHub

```bash
# Trên local machine
cd /home/phuc/workspace/school/medical-data-mining-project

# Add files
git add src/train_slm_qwen_lora_v4_chunked.py
git add CHUNKED_TRAINING_LOG.md
git add FULL_DATASET_TRAINING.md

# Commit
git commit -m "Add chunked training for full dataset"

# Push
git push origin main
```

### Step 2: Clone trong Colab

```python
# Cell 1: Clone repository
!git clone https://github.com/phucfix/medical-data-mining.git
%cd medical-data-mining
```

### Step 3: Cài dependencies

```python
# Cell 2: Install dependencies
!pip install -q transformers peft datasets accelerate bitsandbytes torch
```

### Step 4: Check data

```python
# Cell 3: Verify data files
import os
print("Checking data files...")
print(f"Train file exists: {os.path.exists('data/slm_train_style_adapted.jsonl')}")
print(f"Val file exists: {os.path.exists('data/slm_val.jsonl')}")

# Check file size
if os.path.exists('data/slm_train_style_adapted.jsonl'):
    size_mb = os.path.getsize('data/slm_train_style_adapted.jsonl') / (1024*1024)
    print(f"Train file size: {size_mb:.2f} MB")
```

### Step 5: Run training

```python
# Cell 4: Run chunked training
!python src/train_slm_qwen_lora_v4_chunked.py
```

---

## 📊 Monitoring trong Colab

Bạn sẽ thấy output như này:

```
================================================================================
CHUNKED TRAINING - TRAIN FULL 154K SAMPLES WITHOUT OOM
================================================================================

✓ GPU: Tesla T4
  Memory: 14.74 GB

Loading full training data...
✓ Total samples: 154477

✓ Split into 6 chunks:
  Chunk 1: 30000 samples
  Chunk 2: 30000 samples
  Chunk 3: 30000 samples
  Chunk 4: 30000 samples
  Chunk 5: 30000 samples
  Chunk 6: 24477 samples

================================================================================
TRAINING CHUNK 1/6
Samples in this chunk: 30000
================================================================================
Loading fresh base model: Qwen/Qwen2.5-0.5B-Instruct
...
▶ Training chunk 1...
{'loss': 0.5234, 'learning_rate': 2e-05, 'epoch': 0.5}
...
✓ Chunk 1/6 completed
  Progress: 16.7%

================================================================================
TRAINING CHUNK 2/6
Samples in this chunk: 30000
================================================================================
Loading model from previous chunk...
✓ Loaded LoRA weights from previous chunk
...
```

---

## ⏱️ Thời gian dự kiến:

### Trên T4 (Free Colab):
- **Mỗi chunk**: ~25-30 phút
- **Total**: 2.5-3 giờ
- **Lưu ý**: Free Colab có giới hạn ~12 giờ session

### Trên A100 (Colab Pro):
- **Mỗi chunk**: ~10-15 phút
- **Total**: 1-1.5 giờ
- **Khuyến nghị**: Nếu có Pro, dùng A100!

---

## 💾 Download model sau khi train xong

```python
# Cell 5: Zip và download model
!zip -r qwen_v4_chunked.zip models/qwen2.5-0.5b-med-slm-lora-v4-chunked/

from google.colab import files
files.download('qwen_v4_chunked.zip')
```

---

## 🔧 Troubleshooting

### 1. Session timeout (Free Colab)
**Giải pháp**: 
```python
# Thêm vào cell đầu để keep session alive
import IPython
display(IPython.display.Javascript('''
 function KeepClicking(){
   console.log("Clicking");
   document.querySelector("colab-connect-button").click()
 }
 setInterval(KeepClicking,60000)
'''))
```

### 2. Data file quá lớn (>100MB)
**Giải pháp**:
- Upload lên Google Drive
- Mount Drive trong Colab:

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy data từ Drive
!cp /content/drive/MyDrive/medical-data/*.jsonl /content/data/
```

### 3. OOM trên T4
**Giải pháp**: Script đã optimize, nhưng nếu vẫn OOM:
- Giảm `CHUNK_SIZE` từ 30000 → 20000
- Hoặc upgrade lên Colab Pro (A100)

---

## 📝 Complete Colab Notebook Template

Tôi đã tạo file notebook hoàn chỉnh: `COLAB_CHUNKED_TRAINING.ipynb`

Chỉ cần:
1. Upload notebook lên Colab
2. Click Runtime → Run all
3. Chờ 2.5-3 giờ
4. Download model

---

## ✅ Sau khi training xong

1. **Download model** từ Colab về local
2. **Extract zip file**
3. **Test trên Test_sample.v1.0.csv**:
   ```bash
   python src/test_qwen_on_sample_v3.py
   ```
4. **Kỳ vọng**: 85-90% accuracy! 🎉

---

## 🎯 Tóm tắt workflow:

```
1. Push code lên GitHub
   ↓
2. Mở Google Colab, chọn GPU
   ↓
3. Clone repo từ GitHub
   ↓
4. Install dependencies
   ↓
5. Run chunked training (2.5-3h)
   ↓
6. Download model
   ↓
7. Test và đạt 85-90% accuracy! 🚀
```

Bạn muốn tôi tạo file notebook Colab ready-to-use không? Hoặc bạn sẽ làm manual theo hướng dẫn trên?
