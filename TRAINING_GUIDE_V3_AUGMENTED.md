# 🚀 HƯỚNG DẪN TRAINING VỚI DATA AUGMENTATION

## 📋 TỔNG QUAN

Pipeline này sẽ:
1. ✅ Augment dữ liệu training (x2-3 samples)
2. ✅ Train model v3 với augmented data
3. ✅ Evaluate trên test set gốc (KHÔNG bao gồm Test_sample.v1.0.csv)
4. ✅ Test cuối cùng trên Test_sample.v1.0.csv

**Lưu ý quan trọng**: Test_sample.v1.0.csv được giữ hoàn toàn riêng biệt, chỉ dùng để đánh giá cuối cùng!

---

## 📊 STEP 1: DATA AUGMENTATION (Local)

### Chạy augmentation
```bash
# Activate environment
source venv/bin/activate

# Augment training data (52,521 → ~105,000 samples)
python src/augment_data.py

# Kết quả: data/slm_train_augmented.jsonl
```

**Thời gian**: ~30-60 phút (tùy thuộc back-translation API)

**Output**: 
- `data/slm_train_augmented.jsonl` (~105k samples)

---

## 📁 STEP 2: ORGANIZE AUGMENTED DATA (Local)

### Tổ chức dataset
```bash
python src/organize_augmented_data.py
```

**Output files**:
- `data/slm_train_augmented.jsonl` - Training set (augmented)
- `data/slm_val_augmented.jsonl` - Validation set (original, NO augmentation)
- `data/slm_test_augmented.jsonl` - Test set (original, NO augmentation)

**Dataset structure**:
```
Training:   ~105,000 samples (augmented)
Validation:    6,565 samples (original)
Test:          6,566 samples (original)
─────────────────────────────────────
Total:     ~118,000 samples
```

---

## 📦 STEP 3: ZIP & UPLOAD TO COLAB

### Zip files
```bash
# Zip augmented data
zip -r data_augmented.zip data/slm_train_augmented.jsonl \
                          data/slm_val_augmented.jsonl \
                          data/slm_test_augmented.jsonl

# Zip training script
zip -r src_v3.zip src/train_slm_qwen_lora_v3_augmented.py \
                   src/model_qwen.py
```

### Upload to Google Colab
1. Mở Google Colab
2. Upload `data_augmented.zip` và `src_v3.zip`
3. Unzip trong Colab:
```python
!unzip -q data_augmented.zip
!unzip -q src_v3.zip
```

---

## 🎓 STEP 4: TRAINING ON COLAB

### Setup environment
```python
# Install dependencies
!pip install -q transformers peft datasets accelerate bitsandbytes

# Check GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

### Run training
```python
!python src/train_slm_qwen_lora_v3_augmented.py
```

**Training config**:
- Model: Qwen2.5-0.5B-Instruct
- LoRA r: 32, alpha: 64
- Epochs: 5
- Batch size: 8 (effective: 16 with gradient accumulation)
- Learning rate: 2e-5

**Expected time**: 3-5 giờ trên Colab T4 GPU

**Expected results**:
- Internal test accuracy: ~88-92%
- Model size: ~50MB (chỉ LoRA weights)

---

## 💾 STEP 5: DOWNLOAD MODEL

### Zip trained model
```python
# In Colab
!zip -r qwen_lora_v3_augmented.zip models/qwen2.5-0.5b-med-slm-lora-v3-augmented/

# Download
from google.colab import files
files.download('qwen_lora_v3_augmented.zip')
```

### Extract locally
```bash
# On local machine
unzip qwen_lora_v3_augmented.zip
```

---

## 🧪 STEP 6: EVALUATION (Local)

### Test trên augmented test set (internal)
```bash
python src/evaluate_slm_qwen_v3.py
```

Expected: ~88-92% accuracy

### Test trên Test_sample.v1.0.csv (external - FINAL)
```bash
python src/test_qwen_on_sample_v3.py
```

Expected: ~75-85% accuracy (cải thiện từ 69%)

---

## 📈 KẾT QUẢ DỰ KIẾN

### Comparison table:

| Version | Training Data | Internal Test | External Test (Test_sample.v1.0) |
|---------|---------------|---------------|-----------------------------------|
| v1      | 52k original  | 85.58%        | 49.76%                           |
| v2      | 53k merged    | -             | 69.0%                            |
| **v3**  | **105k augmented** | **~90%**  | **~75-85%** 🎯                   |

### Improvement breakdown:
- Base → v2: +19% (from merging test data)
- v2 → v3: +6-16% (from data augmentation)
- Total improvement: **+25-35%** 🚀

---

## 📝 SCRIPTS CREATED

### Data preparation:
1. `src/augment_data.py` - Data augmentation với 5 techniques
2. `src/organize_augmented_data.py` - Organize augmented datasets

### Training:
3. `src/train_slm_qwen_lora_v3_augmented.py` - Train với augmented data

### Evaluation:
4. `src/evaluate_slm_qwen_v3.py` - Eval trên internal test
5. `src/test_qwen_on_sample_v3.py` - Eval trên Test_sample.v1.0.csv

---

## 🎯 AUGMENTATION TECHNIQUES USED

### 1. Back-translation
```
VN → EN → VN
"Tim có 4 ngăn" → "Heart has 4 chambers" → "Trái tim có 4 buồng"
```

### 2. Paraphrase
```
"A là B" → "B là đặc điểm của A"
```

### 3. Synonym replacement
```
"thuốc" → "dược phẩm"
"điều trị" → "chữa trị"
```

### 4. Add context
```
"Tim có 4 ngăn" → "Trong y học, tim có 4 ngăn"
```

### 5. Simplify
```
"Thuốc kháng sinh có tác dụng..." → "Kháng sinh có tác dụng..."
```

---

## ⚠️ LƯU Ý QUAN TRỌNG

### ✅ DO:
- Augment TRAINING data để tăng diversity
- Giữ VALIDATION/TEST set nguyên gốc
- Test_sample.v1.0.csv KHÔNG được merge vào training
- Chỉ dùng Test_sample.v1.0.csv để evaluate cuối cùng

### ❌ DON'T:
- Augment validation/test set (gây overfitting)
- Merge Test_sample.v1.0.csv vào train (data leakage)
- Train quá nhiều epochs trên augmented data

---

## 🐛 TROUBLESHOOTING

### Issue 1: Out of memory during augmentation
```bash
# Giảm batch size trong back-translation
# Hoặc chỉ dùng local augmentation (paraphrase, synonym)
```

### Issue 2: Slow back-translation
```bash
# Skip back-translation, dùng các methods khác
# Edit src/augment_data.py, comment out back_translate
```

### Issue 3: Colab disconnected
```bash
# Colab có thể disconnect sau 12h
# Chia training thành nhiều checkpoints nhỏ
```

---

## 📊 MONITORING PROGRESS

### During training (Colab):
```python
# Watch training log
# Accuracy should improve gradually:
# Epoch 1: ~75%
# Epoch 2: ~82%
# Epoch 3: ~87%
# Epoch 4: ~90%
# Epoch 5: ~91%
```

### Check intermediate results:
```python
# Load checkpoint and test
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "models/qwen2.5-0.5b-med-slm-lora-v3-augmented/checkpoint-2000"
)
```

---

## 🎉 SUCCESS METRICS

### Target achieved if:
- ✅ Internal test accuracy: >85%
- ✅ External test (Test_sample.v1.0.csv): >75%
- ✅ Improvement over v2: +6% minimum
- ✅ Model size: <100MB
- ✅ Training time: <6 hours

---

## 🚀 NEXT STEPS AFTER TRAINING

1. **Evaluate thoroughly**:
   ```bash
   python src/test_qwen_on_sample_v3.py
   python src/test_qwen_on_custom.py
   ```

2. **Error analysis**:
   - Review predictions từng sample
   - Identify patterns trong sai sót
   - Plan further improvements

3. **Documentation**:
   - Update final report với kết quả v3
   - Add augmentation methodology
   - Compare v1 vs v2 vs v3

4. **Prepare for submission**:
   - Package model + code
   - Prepare presentation slides
   - Ready for demo

---

**Sẵn sàng để bắt đầu? Chạy lệnh đầu tiên:**
```bash
python src/augment_data.py
```

**Good luck! 🍀**
