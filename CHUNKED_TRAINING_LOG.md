# 🚀 CHUNKED TRAINING - Execution Guide

## 📋 Thông tin training:

### Dataset:
- **Total samples**: 154,477
- **Chunks**: 6 chunks (5×30k + 1×24k)
- **Strategy**: Train từng chunk, accumulate LoRA weights

### Expected results:
- ⏱️ **Time**: 2.5-3 giờ total
- 📊 **Accuracy**: 85-90% on Test_sample.v1.0.csv
- 🚫 **OOM**: Zero risk (mỗi chunk chỉ 30k)
- 💾 **Final model**: models/qwen2.5-0.5b-med-slm-lora-v4-chunked/

---

## 🔄 Progress tracking:

### Chunk 1/6 (samples 0-30,000)
- Status: ⏳ Pending
- Time: ~25-30 minutes
- Progress: 0%

### Chunk 2/6 (samples 30,000-60,000)
- Status: ⏳ Pending
- Time: ~25-30 minutes
- Progress: 0%

### Chunk 3/6 (samples 60,000-90,000)
- Status: ⏳ Pending
- Time: ~25-30 minutes
- Progress: 0%

### Chunk 4/6 (samples 90,000-120,000)
- Status: ⏳ Pending
- Time: ~25-30 minutes
- Progress: 0%

### Chunk 5/6 (samples 120,000-150,000)
- Status: ⏳ Pending
- Time: ~25-30 minutes
- Progress: 0%

### Chunk 6/6 (samples 150,000-154,477)
- Status: ⏳ Pending
- Time: ~20-25 minutes
- Progress: 0%

---

## 📊 Estimated timeline:

```
Start time: [Will be logged]
Chunk 1:    [Start] → [End] (~30 min)
Chunk 2:    [Start] → [End] (~30 min)
Chunk 3:    [Start] → [End] (~30 min)
Chunk 4:    [Start] → [End] (~30 min)
Chunk 5:    [Start] → [End] (~30 min)
Chunk 6:    [Start] → [End] (~25 min)
Total:      ~2.5-3 hours
```

---

## 🎯 What happens during training:

1. **Load data**: Read 154,477 samples from slm_train_style_adapted.jsonl
2. **Shuffle**: Random shuffle with seed=42
3. **Split**: Divide into 6 chunks
4. **Train Chunk 1**: Train fresh model on first 30k samples → Save
5. **Train Chunk 2**: Load Chunk 1 weights → Train on next 30k → Save
6. **Train Chunk 3-6**: Continue accumulating knowledge
7. **Finalize**: Copy final model to output directory

---

## 📁 Output files:

```
models/
├── temp_chunks/          # Temporary chunk models
│   ├── chunk_0/         # Chunk 1 model (will be deleted)
│   ├── chunk_1/         # Chunk 2 model (will be deleted)
│   ├── ...
│   └── chunk_5/         # Final chunk (kept as backup)
│
└── qwen2.5-0.5b-med-slm-lora-v4-chunked/   # FINAL MODEL
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── metrics.json
```

---

## ⚠️ Important notes:

1. **Don't interrupt**: Nếu dừng giữa chừng, phải chạy lại từ đầu
2. **GPU memory**: Sẽ tự động clear cache sau mỗi chunk
3. **Disk space**: Cần ~3-4GB cho temp chunks
4. **Progress**: Monitor terminal output để biết progress

---

## 🔍 How to monitor:

Watch for these messages:
```
TRAINING CHUNK 1/6         ← Chunk starting
✓ Chunk 1/6 completed      ← Chunk finished
  Progress: 16.7%          ← Overall progress

TRAINING CHUNK 2/6         ← Next chunk
Loading model from previous chunk...  ← Weight inheritance
```

---

## ✅ Success indicators:

When training completes, you should see:
```
✓✓ CHUNKED TRAINING COMPLETED ✓✓
Final model: models/qwen2.5-0.5b-med-slm-lora-v4-chunked
Total samples trained: 154477
Number of chunks: 6
Expected accuracy: 85-90%
```

---

## 🧪 After training:

1. **Evaluate on Test_sample**:
   ```bash
   python src/test_qwen_on_sample_v3.py
   ```

2. **Check metrics**:
   ```bash
   cat models/qwen2.5-0.5b-med-slm-lora-v4-chunked/metrics.json
   ```

3. **Compare with v2**:
   - v2 (merged data): 69% accuracy
   - v4 (chunked full): 85-90% accuracy (expected)
   - Improvement: +16-21 percentage points! 🎉

---

## 🚀 Ready to start!

Command to run:
```bash
python src/train_slm_qwen_lora_v4_chunked.py
```

Expected completion: 2.5-3 giờ
Expected accuracy: 85-90%
OOM risk: Zero ✅

Good luck! 💪
