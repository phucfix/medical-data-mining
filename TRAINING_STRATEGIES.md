# 🚀 Training Strategies - Giải pháp cho vấn đề "Train lâu quá"

## ⏱️ Vấn đề hiện tại:
- **154,477 samples** với 4 epochs = ~3-5 giờ training trên Colab T4
- OOM errors khi evaluation

---

## 🎯 3 Chiến lược Training

### 1️⃣ **QUICK TRAINING** ⚡ (Khuyến nghị cho deadline)
**Thời gian**: ~30-45 phút  
**Accuracy dự kiến**: 75-80% (vs 69% hiện tại)  
**Phù hợp**: Demo, deadline gấp, test nhanh

#### Thông số:
```python
TRAIN_SAMPLES = 30,000      # ~20% of 154k
VAL_SAMPLES = 1,000          
EPOCHS = 2                   # Thay vì 4
BATCH_SIZE = 8               
LEARNING_RATE = 3e-5         # Cao hơn để học nhanh
```

#### Chạy:
```bash
python src/train_slm_qwen_lora_v4_quick.py
```

#### Ưu điểm:
- ✅ **Nhanh nhất** - chỉ 30-45 phút
- ✅ Vẫn có style adaptation từ Test_sample
- ✅ Accuracy tốt (~75-80%)
- ✅ Không OOM

#### Nhược điểm:
- ⚠️ Không tận dụng hết 154k samples
- ⚠️ Có thể chưa đạt accuracy tối đa

---

### 2️⃣ **BALANCED TRAINING** 🎯 (Khuyến nghị cho production)
**Thời gian**: ~1.5-2 giờ  
**Accuracy dự kiến**: 80-85%  
**Phù hợp**: Balance giữa thời gian và accuracy

#### Thông số:
```python
TRAIN_SAMPLES = 80,000       # ~50% of 154k
VAL_SAMPLES = 2,000          
EPOCHS = 3                   
BATCH_SIZE = 6               
LEARNING_RATE = 2.5e-5       
```

#### Chạy:
```bash
python src/train_slm_qwen_lora_v4_balanced.py
```

#### Ưu điểm:
- ✅ Balance tốt giữa thời gian và accuracy
- ✅ Dùng 50% data = đủ để học tốt
- ✅ Ít OOM hơn

#### Nhược điểm:
- ⚠️ Vẫn mất 1.5-2 giờ
- ⚠️ Chưa phải accuracy tối đa

---

### 3️⃣ **FULL TRAINING** 💪 (Best accuracy)
**Thời gian**: ~3-5 giờ  
**Accuracy dự kiến**: 85-90%  
**Phù hợp**: Khi cần accuracy cao nhất, có thời gian

#### Thông số:
```python
TRAIN_SAMPLES = 154,477      # ALL samples
VAL_SAMPLES = 1,000          # Giảm để tránh OOM
EPOCHS = 4                   
BATCH_SIZE = 4               # Nhỏ để tránh OOM
LEARNING_RATE = 2e-5         
```

#### Chạy:
```bash
python src/train_slm_qwen_lora_v4_style_adapted.py
```

#### Ưu điểm:
- ✅ **Accuracy cao nhất**
- ✅ Tận dụng 100% data
- ✅ Đã fix OOM issues

#### Nhược điểm:
- ⚠️ **Lâu nhất** - 3-5 giờ
- ⚠️ Cần GPU tốt

---

## 📊 So sánh các chiến lược:

| Chiến lược | Thời gian | Samples | Epochs | Accuracy dự kiến | Khuyến nghị |
|------------|-----------|---------|--------|------------------|-------------|
| **Quick** ⚡ | 30-45 min | 30k | 2 | 75-80% | ✅ Deadline |
| **Balanced** 🎯 | 1.5-2 giờ | 80k | 3 | 80-85% | ✅ Production |
| **Full** 💪 | 3-5 giờ | 154k | 4 | 85-90% | Best accuracy |

---

## 🎓 Khuyến nghị cho môn học:

### Nếu deadline gấp (< 2 giờ):
```bash
python src/train_slm_qwen_lora_v4_quick.py
```
- 30-45 phút training
- 75-80% accuracy là **đủ tốt** cho assignment
- Vẫn có style adaptation

### Nếu có thời gian (2-4 giờ):
```bash
# Tùy chọn: tạo balanced version
# Hoặc chạy quick 2 lần với random seeds khác nhau, chọn model tốt nhất
```

### Nếu muốn điểm cao (> 4 giờ):
```bash
python src/train_slm_qwen_lora_v4_style_adapted.py
```
- 85-90% accuracy
- Full training với all data

---

## 💡 Tips tăng tốc thêm:

### 1. Sử dụng Google Colab Pro
- GPU A100/V100 nhanh hơn T4 ~3-4x
- Giảm thời gian từ 3 giờ → 45 phút

### 2. Chạy song song nhiều configs
- Train quick trước để test
- Trong khi đó train full ở background

### 3. Early stopping
- Thêm patience=3 vào training_args
- Dừng sớm nếu accuracy không tăng

### 4. Giảm validation frequency
```python
EVAL_STEPS = 5000  # Thay vì 2000
```

---

## 🔧 Troubleshooting:

### Vẫn chạy lâu:
1. Dùng **Quick version** (30k samples)
2. Giảm epochs xuống 1-2
3. Tăng batch size nếu có memory

### Vẫn OOM:
1. Giảm BATCH_SIZE xuống 2
2. Giảm VAL_SAMPLES xuống 500
3. Tắt gradient_checkpointing (nhanh hơn nhưng tốn memory)

### Accuracy thấp:
1. Tăng số samples (quick → balanced)
2. Tăng epochs (2 → 3)
3. Chạy full training

---

## ✅ Quyết định nhanh:

**Bạn cần gì?**
- ⏰ **Nhanh nhất**: Quick (30-45 phút, 75-80%)
- ⚖️ **Cân bằng**: Balanced (1.5-2 giờ, 80-85%) - Chưa có script
- 🏆 **Tốt nhất**: Full (3-5 giờ, 85-90%)

**Khuyến nghị**: Chạy **Quick version** trước để test, nếu thỏa mãn thì xong. Nếu cần accuracy cao hơn thì chạy Full version sau.
