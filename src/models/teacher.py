"""Fine-tune XLM-R on the shared training split."""
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.models.common import ARTIFACTS, labels_from_train, read_split
from src.nlp.normalize import normalize


class TextDataset(Dataset):
    def __init__(self, rows, tokenizer, labels):
        self.rows, self.tokenizer, self.labels = rows, tokenizer, labels

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        data = self.tokenizer(normalize(row['text']), truncation=True, padding='max_length', max_length=64, return_tensors='pt')
        result = {k: v.squeeze(0) for k, v in data.items()}
        result['labels'] = torch.tensor(self.labels.index(row['label']))
        return result


def main():
    torch.manual_seed(42)
    labels = labels_from_train()
    name = 'xlm-roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id={v: i for i, v in enumerate(labels)})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).train()
    loader = DataLoader(TextDataset(read_split('train'), tokenizer, labels), batch_size=16, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    for _ in range(3):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            model(**batch).loss.backward()
            optimizer.step()
    path = ARTIFACTS / 'teacher'
    model.cpu().save_pretrained(path)
    tokenizer.save_pretrained(path)

if __name__ == '__main__':
    main()
