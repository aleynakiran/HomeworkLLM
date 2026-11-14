"""Distill teacher logits into multilingual DistilBERT and save dynamic INT8 weights."""
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.models.common import ARTIFACTS, labels_from_train, read_split
from src.models.teacher import TextDataset


def main():
    torch.manual_seed(42)
    labels = labels_from_train()
    teacher_path = ARTIFACTS / 'teacher'
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_path)
    name = 'distilbert-base-multilingual-cased'
    tokenizer = AutoTokenizer.from_pretrained(name)
    student = AutoModelForSequenceClassification.from_pretrained(name, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id={v: i for i, v in enumerate(labels)})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_path).to(device).eval()
    student.to(device).train()
    rows = read_split('train')
    # Both datasets have identical order; each uses its own model's tokenizer.
    student_data = TextDataset(rows, tokenizer, labels)
    teacher_data = TextDataset(rows, teacher_tokenizer, labels)
    order = torch.randperm(len(rows)).tolist()
    optimizer = torch.optim.AdamW(student.parameters(), lr=2e-5)
    temperature = 2.0
    for _ in range(3):
        for start in range(0, len(order), 16):
            indexes = order[start:start + 16]
            def batch(dataset):
                examples = [dataset[i] for i in indexes]
                return {k: torch.stack([example[k] for example in examples]).to(device) for k in examples[0]}
            sb, tb = batch(student_data), batch(teacher_data)
            with torch.no_grad():
                teacher_logits = teacher(**tb).logits
            output = student(**sb)
            soft = nn.functional.kl_div(nn.functional.log_softmax(output.logits / temperature, dim=-1), nn.functional.softmax(teacher_logits / temperature, dim=-1), reduction='batchmean') * temperature**2
            loss = 0.5 * output.loss + 0.5 * soft
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    path = ARTIFACTS / 'student'
    student.cpu().eval().save_pretrained(path)
    tokenizer.save_pretrained(path)
    # Save a state dict; reconstruct the same architecture before loading it.
    quantized = torch.quantization.quantize_dynamic(student.cpu().eval(), {nn.Linear}, dtype=torch.qint8)
    torch.save(quantized.state_dict(), ARTIFACTS / 'student_int8.pt')

if __name__ == '__main__':
    main()
