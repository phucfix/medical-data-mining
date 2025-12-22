"""
Per-group evaluation script
- Load model + tokenizer
- Run generation on a JSONL dataset (format: {"input":"...","output":"TRUE"/"FALSE"})
- Use MedicalEntityExtractor to assign each sample to a category: disease/symptom/drug/other
- Compute accuracy per category and overall metrics
- Save predictions to `outputs/per_group_predictions.jsonl`

Usage:
  python src/evaluation/per_group_evaluation.py --input data/slm_test_dev.jsonl --model models/qwen2.5-0.5b-med-slm-lora

"""
import argparse
import json
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import extractor
from src.preprocessing.data_cleaner import MedicalEntityExtractor


def load_jsonl(path: Path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def create_prompt(input_text: str) -> str:
    return f"Bạn là một trợ lý y tế. Hãy trả lời Đúng hoặc Sai.\nNhận định: {input_text}\nĐáp án:"


def predict(model, tokenizer, inputs, device='cpu', max_new_tokens=10):
    model.eval()
    preds = []
    with torch.no_grad():
        for inp in inputs:
            prompt = create_prompt(inp)
            tokens = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=256)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            out = model.generate(**tokens, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            gen = tokenizer.decode(out[0][tokens['input_ids'].shape[1]:], skip_special_tokens=True).strip().upper()
            if 'TRUE' in gen:
                preds.append('TRUE')
            elif 'FALSE' in gen:
                preds.append('FALSE')
            else:
                preds.append(gen.split()[0] if gen else 'UNKNOWN')
    return preds


def categorize_sample(extractor: MedicalEntityExtractor, text: str) -> str:
    entities = extractor.extract_entities(text)
    if entities.get('diseases'):
        return 'disease'
    if entities.get('symptoms'):
        return 'symptom'
    if entities.get('drugs'):
        return 'drug'
    return 'other'


def main(args):
    input_path = Path(args.input)
    model_dir = Path(args.model)
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)

    data = load_jsonl(input_path)
    texts = [d['input'] for d in data]
    golds = [d['output'].upper() for d in data]

    print(f"Loaded {len(data)} samples from {input_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map='auto' if device=='cuda' else None, torch_dtype=torch.float16 if device=='cuda' else torch.float32, trust_remote_code=True)

    extractor = MedicalEntityExtractor()

    preds = predict(model, tokenizer, texts, device=device)

    # Aggregate
    stats = defaultdict(lambda: {'total':0,'correct':0,'conf':defaultdict(int)})
    overall = {'total':0,'correct':0}

    out_file = output_dir / 'per_group_predictions.jsonl'
    with open(out_file, 'w', encoding='utf-8') as f:
        for txt, gold, pred in zip(texts, golds, preds):
            cat = categorize_sample(extractor, txt)
            stats[cat]['total'] += 1
            stats[cat]['conf'][f"{gold}->{pred}"] += 1
            if pred == gold:
                stats[cat]['correct'] += 1
                overall['correct'] += 1
            overall['total'] += 1
            f.write(json.dumps({'input': txt, 'gold': gold, 'pred': pred, 'category': cat}, ensure_ascii=False) + '\n')

    # Print report
    print('\nPer-group accuracy:')
    for cat, v in stats.items():
        total = v['total']
        correct = v['correct']
        acc = correct / total if total else 0
        print(f"  - {cat}: {correct}/{total} => {acc*100:.2f}%")
        print(f"    Confusions: {dict(v['conf'])}")

    overall_acc = overall['correct']/overall['total'] if overall['total'] else 0
    print(f"\nOverall: {overall['correct']}/{overall['total']} => {overall_acc*100:.2f}%")
    print(f"Predictions saved to: {out_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--model', type=str, required=True, help='Model directory (tokenizer + model)')
    args = parser.parse_args()
    main(args)
