# V5 Optimized Training - Giới hạn ≤1B Parameters

## 🎯 Mục tiêu
- Cải thiện accuracy từ **51%** (v4-chunked) lên **60-65%** 
- Giữ model size ≤ 1B parameters (Qwen2.5-0.5B = 494M)
- Không cần upgrade model size, chỉ optimize training

## ⚙️ Cải tiến chính

### 1. **Tăng số epochs** ⏱️
- **v4**: 1 epoch per chunk (6 chunks total)
- **v5**: 3 epochs full dataset
- **Lý do**: Model chưa học đủ, cần nhiều epochs hơn

### 2. **Giảm Learning Rate** 📉
- **v4**: 2e-5 (fast but unstable)
- **v5**: 5e-6 (slow but stable)
- **Lý do**: LR thấp hơn → học chậm hơn nhưng chính xác hơn, tránh overshoot

### 3. **Cosine Scheduler + Longer Warmup** 📊
- **v4**: Warmup ratio = 0.05
- **v5**: Warmup ratio = 0.1 + cosine decay
- **Lý do**: Warmup dài hơn → stable start, cosine decay → smooth convergence

### 4. **Tăng LoRA Rank** 🚀
- **v4**: rank=32, alpha=64
- **v5**: rank=64, alpha=128
- **Lý do**: Model capacity cao hơn → học được patterns phức tạp hơn

### 5. **Gradient Clipping** 🛡️
- **v5**: max_grad_norm=1.0
- **Lý do**: Prevent gradient explosion, training stability

### 6. **Smart Evaluation** 📈
- **v4**: No evaluation during training
- **v5**: Eval every 2000 steps, save best checkpoint
- **Lý do**: Monitor training progress, prevent overfitting

## 📊 So sánh Config

| Parameter | v4-chunked | v5-optimized | Change |
|-----------|------------|--------------|--------|
| Epochs | 1 per chunk | 3 full | +200% |
| Learning Rate | 2e-5 | 5e-6 | -75% |
| LR Scheduler | Linear | Cosine | Better |
| Warmup Ratio | 0.05 | 0.1 | +100% |
| LoRA Rank | 32 | 64 | +100% |
| LoRA Alpha | 64 | 128 | +100% |
| Gradient Clip | None | 1.0 | Added |
| Eval Strategy | No | Every 2000 steps | Added |

## 🎯 Expected Results

| Version | Strategy | Accuracy | Improvement |
|---------|----------|----------|-------------|
| v2-merged | 50k samples, data leakage | 69% | ⚠️ Inflated |
| v4-chunked | 154k, 1 epoch chunks | 51% | Baseline |
| **v5-optimized** | **154k, 3 epochs, optimized** | **60-65%** | **+9-14%** |

## ⏱️ Training Time
- **v4-chunked**: ~2.5-3 hours (6 chunks × 25-35 min)
- **v5-optimized**: ~4-5 hours (3 epochs full dataset)
- **Hardware**: Google Colab T4 GPU (15GB VRAM)

## 📝 Cách sử dụng

### Option 1: Local Training (nếu có GPU)
```bash
python src/train_slm_qwen_lora_v5_optimized.py
```

### Option 2: Google Colab (Recommended)
1. Upload `V5_Optimized_Training_Colab.ipynb` lên Colab
2. Chọn Runtime > Change runtime type > T4 GPU
3. Run all cells
4. Đợi ~4-5 hours
5. Download model khi xong

## 🔬 Tại sao không upgrade lên 1.5B?

**User constraint**: Chỉ được dùng model ≤ 1B parameters

**Available options**:
- ✅ Qwen2.5-0.5B (494M) - đang dùng
- ❌ Qwen2.5-1.5B (1.54B) - vượt giới hạn
- ⚠️ TinyLlama-1.1B (1.1B) - có thể thử nhưng kém hơn Qwen

**Solution**: Optimize training thay vì upgrade model

## 📈 Analysis: Tại sao v4 chỉ 51%?

### 1. **Underfitting** (chưa học đủ)
- v4 train 1 epoch per chunk = effectively 1 epoch total
- Medical domain phức tạp → cần nhiều epochs hơn
- **Fix v5**: 3 epochs

### 2. **Learning Rate quá cao**
- LR=2e-5 có thể skip qua optimal points
- **Fix v5**: LR=5e-6 (slow but precise)

### 3. **LoRA capacity thấp**
- Rank=32 có thể không đủ cho medical domain
- **Fix v5**: Rank=64 (2x capacity)

### 4. **Chunked training issues**
- Mỗi chunk train riêng biệt → không ideal
- **Fix v5**: Train full dataset liên tục

### 5. **No monitoring**
- v4 không có eval → không biết training progress
- **Fix v5**: Eval every 2000 steps

## 🎓 Key Learnings

1. **Model size không phải everything**: v4 có 3x data (154k vs 50k) nhưng chỉ tăng 2% (49%→51%)
   - → Bottleneck không phải data, mà là **training strategy**

2. **Chunked training không tối ưu**: Chia nhỏ chunks → mất continuity
   - → Train full dataset liên tục tốt hơn

3. **1 epoch không đủ cho medical domain**: Cần ít nhất 3 epochs
   - → Medical QA phức tạp hơn general text

4. **LoRA rank matters**: Rank càng cao → capacity càng lớn
   - → Nhưng không nên quá cao (risk overfitting)

## 🚀 Next Steps

### Sau khi train v5:
1. **Test model**: `python src/test_qwen_on_sample_v4.py --version v5-optimized`
2. **Compare results**: v4 (51%) vs v5 (expected 60-65%)

### Nếu v5 vẫn chưa đủ (< 60%):
- **Option A**: Thử TinyLlama-1.1B (gần 1B limit)
- **Option B**: Cải thiện data quality:
  - Augment thêm medical examples
  - Filter low-quality samples
  - Balance TRUE/FALSE distribution
- **Option C**: Ensemble methods:
  - Train multiple v5 models với different seeds
  - Voting mechanism

### Nếu v5 đạt 60-65%:
- ✅ Success! Đã cải thiện +9-14% trong constraints
- 📊 Analyze error cases
- 🎯 Fine-tune trên specific medical subdomains

## 📚 Files Created

- `src/train_slm_qwen_lora_v5_optimized.py` - Training script
- `V5_Optimized_Training_Colab.ipynb` - Colab notebook
- `TRAINING_V5_GUIDE.md` - This guide

## 🤝 Comparison với alternatives

| Approach | Pros | Cons | Expected Gain |
|----------|------|------|---------------|
| **v5 Optimized** | ✅ Simple, stays within 1B limit | ⚠️ Still 0.5B capacity | **+9-14%** |
| Upgrade to 1.5B | Higher capacity | ❌ Violates ≤1B constraint | N/A |
| TinyLlama 1.1B | Almost at limit | ⚠️ Lower quality than Qwen | +5-10%? |
| Data augmentation | More training data | Diminishing returns (154k already large) | +2-5% |
| Ensemble | Best accuracy | Complex deployment | +3-7% |

**Recommendation**: Try v5 first! Nếu không đủ thì consider TinyLlama-1.1B.

---

**Created**: 2025-12-09  
**Author**: GitHub Copilot  
**Status**: Ready for training 🚀
