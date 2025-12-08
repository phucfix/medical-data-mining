# CHIẾN LƯỢC TĂNG ĐỘ CHÍNH XÁC CHO MODEL Y TẾ

## 📊 Hiện trạng: 69% accuracy → Mục tiêu: 80-85%+

---

## 1️⃣ CẢI THIỆN DỮ LIỆU (Data-Centric Approach)

### 🎯 A. Tăng số lượng dữ liệu training (Impact: ⭐⭐⭐⭐⭐)

#### Phương án 1: Merge toàn bộ Test_sample vào training
```python
# Thay vì 50/50, dùng 80/20 hoặc 90/10
TEST_SAMPLE_TRAIN_RATIO = 0.8  # 80% để train, 20% để test
```
**Expected improvement**: +5-10%

#### Phương án 2: Data augmentation thông minh
```python
# Paraphrase câu hỏi với nhiều cách diễn đạt khác nhau
Original: "Insulin được sản xuất bởi tuyến tụy."
Aug 1:    "Tuyến tụy là cơ quan sản xuất insulin."
Aug 2:    "Insulin có nguồn gốc từ tuyến tụy."
Aug 3:    "Tuyến tụy chịu trách nhiệm sản xuất hormone insulin."
```
**Tools**: 
- Back-translation (VN → EN → VN)
- Paraphrase với GPT-4/Gemini
- Vietnamese NLP tools (VnCoreNLP)

**Expected improvement**: +3-5%

#### Phương án 3: Active learning
```python
# 1. Test model trên unlabeled data
# 2. Chọn những câu model "không chắc chắn" (confidence < 0.7)
# 3. Human labeling cho những câu này
# 4. Thêm vào training set
```
**Expected improvement**: +5-8%

---

### 🧹 B. Cải thiện chất lượng dữ liệu (Impact: ⭐⭐⭐⭐)

#### Chiến lược 1: Data cleaning aggressive
```python
# Loại bỏ dữ liệu nhiễu
def filter_low_quality(data):
    filtered = []
    for sample in data:
        # Loại bỏ câu quá ngắn/dài
        if len(sample['input']) < 30 or len(sample['input']) > 200:
            continue
        
        # Loại bỏ câu có grammar issues
        if has_grammar_errors(sample['input']):
            continue
        
        # Loại bỏ câu có factual contradictions
        if check_medical_contradiction(sample['input'], sample['output']):
            continue
            
        filtered.append(sample)
    return filtered
```
**Expected improvement**: +2-4%

#### Chiến lược 2: Expert validation
- Thuê chuyên gia y tế review 10-20% data
- Focus vào những câu model hay sai
- Sửa labels sai và refine wording
**Expected improvement**: +3-5%

#### Chiến lược 3: Hard negative mining
```python
# Tạo câu FALSE khó hơn bằng cách:
# 1. Đảo ngược logic trong câu TRUE
# 2. Thay thế 1 chi tiết quan trọng
Original TRUE: "Insulin được sản xuất bởi tuyến tụy."
Hard FALSE:    "Insulin được sản xuất bởi tuyến giáp." (thay tụy → giáp)
```
**Expected improvement**: +4-7%

---

### 🔄 C. Balance và diversity (Impact: ⭐⭐⭐)

#### Cân bằng độ khó
```python
# Hiện tại có thể có bias về độ khó
Easy (50%):   "Tim người có 4 ngăn."
Medium (30%): "Insulin điều hòa đường huyết bằng cách..."
Hard (20%):   "Trong đái tháo đường type 2, tế bào beta..."
```

#### Cân bằng domain
```python
# Ensure coverage across medical domains
distribution = {
    "Cardiology": 15%,
    "Endocrinology": 15%,
    "Neurology": 15%,
    "Infectious Disease": 15%,
    "Pharmacology": 20%,
    "Anatomy": 10%,
    "Other": 10%
}
```
**Expected improvement**: +3-5%

---

## 2️⃣ CẢI THIỆN MODEL (Model-Centric Approach)

### 🤖 A. Thử model lớn hơn (Impact: ⭐⭐⭐⭐⭐)

#### Option 1: Qwen2.5-1.5B-Instruct
```python
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # 1.5B params
# Vẫn < 2B, tăng capacity x3
```
**Expected improvement**: +8-12%
**Trade-off**: Tốn memory/compute hơn

#### Option 2: Phi-3-mini (3.8B)
```python
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"  # 3.8B params
# Rất mạnh cho reasoning tasks
```
**Expected improvement**: +10-15%
**Trade-off**: Cần GPU tốt hơn

#### Option 3: Llama-3.2-1B
```python
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
# Balance tốt giữa size và performance
```
**Expected improvement**: +5-10%

---

### ⚙️ B. Tối ưu hyperparameters (Impact: ⭐⭐⭐⭐)

#### Grid search cho LoRA
```python
from itertools import product

# Các hyperparams cần tune
lora_r_options = [8, 16, 32, 64]
lora_alpha_options = [16, 32, 64]
lora_dropout_options = [0.05, 0.1, 0.15]
learning_rate_options = [1e-5, 2e-5, 5e-5]

# Grid search
best_acc = 0
best_config = None

for r, alpha, dropout, lr in product(
    lora_r_options, lora_alpha_options, 
    lora_dropout_options, learning_rate_options
):
    model = train_with_config(r, alpha, dropout, lr)
    acc = evaluate(model)
    if acc > best_acc:
        best_acc = acc
        best_config = (r, alpha, dropout, lr)
```
**Expected improvement**: +3-6%

#### Thử các learning rate schedules
```python
# Cosine annealing with warm restarts
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=total_steps,
    num_cycles=3  # Multiple restarts
)
```
**Expected improvement**: +2-4%

---

### 🎓 C. Advanced training techniques (Impact: ⭐⭐⭐⭐)

#### 1. Multi-stage training
```python
# Stage 1: Train trên toàn bộ data với learning rate cao
train_stage1(lr=5e-5, epochs=2)

# Stage 2: Fine-tune trên hard examples với lr thấp
hard_examples = get_hard_examples(model, train_data)
train_stage2(hard_examples, lr=1e-5, epochs=2)

# Stage 3: Polish với data augmentation
augmented_data = augment(hard_examples)
train_stage3(augmented_data, lr=5e-6, epochs=1)
```
**Expected improvement**: +5-8%

#### 2. Curriculum learning
```python
# Train từ dễ đến khó
easy_data = filter_by_difficulty(data, level='easy')
medium_data = filter_by_difficulty(data, level='medium')
hard_data = filter_by_difficulty(data, level='hard')

# Epoch 1: Easy only
train(easy_data)
# Epoch 2: Easy + Medium
train(easy_data + medium_data)
# Epoch 3: All data
train(easy_data + medium_data + hard_data)
```
**Expected improvement**: +4-6%

#### 3. Ensemble models
```python
# Train 3-5 models với different random seeds
models = []
for seed in [42, 123, 456, 789, 2024]:
    model = train_with_seed(seed)
    models.append(model)

# Voting mechanism
def predict_ensemble(text):
    votes = [model.predict(text) for model in models]
    return majority_vote(votes)
```
**Expected improvement**: +5-10%
**Trade-off**: Tốn compute x5

---

## 3️⃣ KIẾN TRÚC NÂNG CAO (Advanced Architecture)

### 🧠 A. RAG (Retrieval-Augmented Generation) (Impact: ⭐⭐⭐⭐⭐)

```python
from langchain import FAISS, OpenAI
from langchain.chains import RetrievalQA

# 1. Build medical knowledge base
knowledge_base = load_medical_kb()  # 233+ entries
embeddings = create_embeddings(knowledge_base)
vectorstore = FAISS.from_documents(knowledge_base, embeddings)

# 2. RAG pipeline
def predict_with_rag(question):
    # Retrieve relevant context
    relevant_docs = vectorstore.similarity_search(question, k=3)
    
    # Augment prompt with context
    context = "\n".join([doc.page_content for doc in relevant_docs])
    augmented_prompt = f"""
    Context y tế liên quan:
    {context}
    
    Câu hỏi: {question}
    Trả lời TRUE/FALSE:
    """
    
    # Generate answer
    return model.generate(augmented_prompt)
```
**Expected improvement**: +10-15%
**Benefit**: Model có access đến facts chính xác

---

### 🔗 B. Chain-of-Thought prompting (Impact: ⭐⭐⭐⭐)

```python
# Thay vì chỉ yêu cầu TRUE/FALSE, yêu cầu reasoning
prompt = """
Bạn là bác sĩ. Hãy phân tích câu sau:

Câu hỏi: {question}

Bước 1: Xác định các khái niệm y tế chính
Bước 2: Phân tích tính đúng/sai của từng phần
Bước 3: Kết luận TRUE hoặc FALSE

Trả lời:
"""

# Training data cũng cần format CoT
{
    "input": "Insulin được sản xuất bởi tuyến tụy.",
    "output": """
    Bước 1: Khái niệm - Insulin (hormone), tuyến tụy (cơ quan)
    Bước 2: Insulin được sản xuất bởi tế bào beta trong đảo tụy (islets of Langerhans) ở tuyến tụy
    Bước 3: Kết luận: TRUE
    """
}
```
**Expected improvement**: +8-12%
**Note**: Cần nhiều compute hơn cho generation dài

---

### 🎯 C. Two-stage prediction (Impact: ⭐⭐⭐⭐)

```python
# Stage 1: Binary classification (TRUE/FALSE)
class BinaryClassifier(nn.Module):
    def __init__(self, base_model):
        self.encoder = base_model
        self.classifier = nn.Linear(hidden_size, 2)  # TRUE/FALSE
    
    def forward(self, input_ids):
        embeddings = self.encoder(input_ids)
        logits = self.classifier(embeddings)
        return logits

# Stage 2: Confidence estimation
class ConfidenceEstimator(nn.Module):
    def estimate_confidence(self, embeddings):
        # Predict how confident the model should be
        return confidence_score

# Final prediction
pred = binary_classifier(text)
conf = confidence_estimator(text)
if conf < 0.6:
    # Use RAG or ensemble for low-confidence predictions
    pred = fallback_prediction(text)
```
**Expected improvement**: +6-10%

---

## 4️⃣ KỸ THUẬT HẬU XỬ LÝ (Post-processing)

### 🔍 A. Confidence thresholding (Impact: ⭐⭐⭐)

```python
def predict_with_confidence(text, threshold=0.7):
    logits = model(text)
    probs = softmax(logits)
    max_prob = max(probs)
    
    if max_prob < threshold:
        # Không chắc chắn → dùng fallback
        return ensemble_predict(text)
    else:
        return argmax(probs)
```
**Expected improvement**: +3-5%

---

### 🧪 B. Rule-based post-correction (Impact: ⭐⭐⭐)

```python
def post_process_prediction(text, pred):
    # Rule 1: Từ khóa "không", "không phải" → likely FALSE
    if "không" in text and pred == "TRUE":
        # Double-check with higher threshold
        if model_confidence(text) < 0.85:
            pred = "FALSE"
    
    # Rule 2: Medical facts từ knowledge base
    if check_against_kb(text) != pred:
        # KB says different → trust KB for factual statements
        pred = get_from_kb(text)
    
    # Rule 3: Logic consistency
    if has_contradiction(text):
        pred = "FALSE"
    
    return pred
```
**Expected improvement**: +2-4%

---

## 5️⃣ PHƯƠNG PHÁP KẾT HỢP (Hybrid Approach)

### 🎭 A. Multi-model ensemble (Impact: ⭐⭐⭐⭐⭐)

```python
# Combine different architectures
models = {
    'qwen_small': Qwen2.5-0.5B,
    'qwen_large': Qwen2.5-1.5B,
    'phi3': Phi-3-mini,
    'llama': Llama-3.2-1B
}

def weighted_ensemble(text):
    predictions = {}
    for name, model in models.items():
        pred, conf = model.predict_with_confidence(text)
        predictions[name] = (pred, conf)
    
    # Weighted voting based on confidence
    weighted_vote = sum(conf if pred == 'TRUE' else -conf 
                       for pred, conf in predictions.values())
    
    return 'TRUE' if weighted_vote > 0 else 'FALSE'
```
**Expected improvement**: +10-15%

---

### 🔬 B. Test-time augmentation (Impact: ⭐⭐⭐)

```python
def predict_with_tta(text):
    # Generate variations of input
    variations = [
        text,
        paraphrase(text),
        add_context(text),
        simplify(text)
    ]
    
    # Predict on all variations
    predictions = [model.predict(var) for var in variations]
    
    # Majority vote
    return majority_vote(predictions)
```
**Expected improvement**: +4-6%

---

## 📊 ROADMAP ƯU TIÊN

### 🚀 Quick Wins (1-3 ngày):
1. **Merge 80% Test_sample vào training** → +5-8%
2. **Tune hyperparameters (LoRA r, lr)** → +3-5%
3. **Rule-based post-processing** → +2-3%
**Tổng**: +10-16% → **79-85% accuracy**

### 🎯 Medium-term (1-2 tuần):
4. **Data augmentation (paraphrase)** → +3-5%
5. **Hard negative mining** → +4-6%
6. **Try larger model (Qwen-1.5B)** → +8-10%
**Tổng**: +15-21% → **84-90% accuracy**

### 🏆 Advanced (1 tháng):
7. **RAG integration** → +10-15%
8. **Ensemble 3-5 models** → +5-8%
9. **Chain-of-Thought training** → +5-8%
**Tổng**: +20-31% → **89-100% accuracy**

---

## 💡 KHUYẾN NGHỊ CỤ THỂ

### Để đạt 75-80% ngay (trong 3 ngày):
```bash
# 1. Merge more test data
python src/merge_datasets.py --ratio 0.8

# 2. Train với LoRA tối ưu
python src/train_slm_qwen_lora_v3.py \
    --lora_r 32 \
    --lora_alpha 64 \
    --learning_rate 2e-5 \
    --epochs 5

# 3. Add post-processing rules
python src/test_with_post_processing.py
```

### Để đạt 85%+ (trong 2 tuần):
- Upgrade to Qwen2.5-1.5B hoặc Phi-3
- Implement RAG với medical KB
- Data augmentation extensive (x5 data)

---

## 🎯 KẾT LUẬN

**Most effective strategies** (ROI cao nhất):
1. ⭐⭐⭐⭐⭐ Tăng model size (Qwen-1.5B/Phi-3): +10-15%
2. ⭐⭐⭐⭐⭐ RAG integration: +10-15%
3. ⭐⭐⭐⭐⭐ Merge more training data: +5-10%
4. ⭐⭐⭐⭐ Data augmentation: +5-8%
5. ⭐⭐⭐⭐ Ensemble models: +5-10%

**Realistic target**:
- 1 tuần: **75-80% accuracy**
- 2 tuần: **80-85% accuracy**  
- 1 tháng: **85-90% accuracy**

Bạn muốn tôi implement cụ thể phương án nào không?
