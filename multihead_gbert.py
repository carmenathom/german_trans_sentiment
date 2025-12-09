import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
import pandas as pd
import numpy as np
from itertools import cycle

class TextDataset(Dataset):
    def __init__(self, tokenized_inputs, labels):
        self.input_ids = tokenized_inputs["input_ids"]
        self.attention_mask = tokenized_inputs["attention_mask"]
        self.labels = torch.tensor(labels, dtype = torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx]
        }

class MultiHeadGBERT(nn.Module):
    def __init__(self, pretrained_model="deepset/gbert-base",
                 num_labels_offense = 2, num_labels_toxic = 2, num_labels_hate = 3):
        super(MultiHeadGBERT, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model)
        hidden_size = self.bert.config.hidden_size

        self.classifier_offense = nn.Linear(hidden_size, num_labels_offense)
        self.classifier_toxic = nn.Linear(hidden_size, num_labels_toxic)
        self.classifier_hate = nn.Linear(hidden_size, num_labels_hate)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  

        return {
            "offense": self.classifier_offense(pooled_output),
            "toxic": self.classifier_toxic(pooled_output),
            "hate": self.classifier_hate(pooled_output)
        }

def tokenize_texts(tokenizer, texts, max_length = 128):
    return tokenizer(
        list(texts),
        padding = True,
        truncation = True,
        max_length = max_length,
        return_tensors = "pt"
    )

def create_dataloader(df, text_col, label_col, tokenizer, batch_size = 16, shuffle = True):
    tokenized = tokenize_texts(tokenizer, df[text_col])
    labels = df[label_col].tolist()
    dataset = TextDataset(tokenized, labels)
    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)

def evaluate(model, dataloader, head, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)[head]
            loss = criterion(outputs, labels)
            total_loss += loss.item() * input_ids.size(0)

            preds = torch.argmax(outputs, dim = 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def train_multitask(model, dataloaders_train, dataloaders_val, class_weights, device,
                    epochs = 5, patience = 2):
    optimizer = AdamW(model.parameters(), lr = 2e-5)

    criterion_offense = nn.CrossEntropyLoss(weight=class_weights["offense"].to(device))
    criterion_toxic   = nn.CrossEntropyLoss(weight=class_weights["toxic"].to(device))
    criterion_hate    = nn.CrossEntropyLoss(weight=class_weights["hate"].to(device))

    best_val_loss = np.inf
    patience_counter = 0

    iter_offense = cycle(dataloaders_train["offense"])
    iter_toxic   = cycle(dataloaders_train["toxic"])
    iter_hate    = cycle(dataloaders_train["hate"])

    steps_per_epoch = max(len(dataloaders_train["offense"]),
                          len(dataloaders_train["toxic"]),
                          len(dataloaders_train["hate"]))

    for epoch in range(epochs):
        model.train()
        total_loss_epoch = 0.0
        print(f"\nEpoch {epoch+1}/{epochs}")

        for step in range(steps_per_epoch):
            optimizer.zero_grad()

            batch = next(iter_offense)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids, attention_mask)
            loss_offense = criterion_offense(outputs["offense"], labels)

            batch = next(iter_toxic)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids, attention_mask)
            loss_toxic = criterion_toxic(outputs["toxic"], labels)

            batch = next(iter_hate)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids, attention_mask)
            loss_hate = criterion_hate(outputs["hate"], labels)

            total_loss = loss_offense + loss_toxic + loss_hate
            total_loss.backward()
            optimizer.step()

            total_loss_epoch += total_loss.item()
            if step % 10 == 0:
                print(f"Step {step}, Batch Loss: {total_loss.item():.4f}")

        model.eval()
        val_loss_offense, val_acc_offense = evaluate(model, dataloaders_val["offense"], "offense", device, criterion_offense)
        val_loss_toxic, val_acc_toxic = evaluate(model, dataloaders_val["toxic"], "toxic", device, criterion_toxic)
        val_loss_hate, val_acc_hate = evaluate(model, dataloaders_val["hate"], "hate", device, criterion_hate)

        avg_val_loss = val_loss_offense + val_loss_toxic + val_loss_hate
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
        print(f"Offense Acc: {val_acc_offense:.4f}, Toxic Acc: {val_acc_toxic:.4f}, Hate Acc: {val_acc_hate:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "multihead_gbert_best.pt")
            print("Saved best model.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-base")

    germeval18_train = pd.read_csv("dataset/sentiment/clean/germeval18_train.csv")
    germeval18_test  = pd.read_csv("dataset/sentiment/clean/germeval18_test.csv")
    germeval21_train = pd.read_csv("dataset/sentiment/clean/germeval21_train.csv")
    germeval21_test  = pd.read_csv("dataset/sentiment/clean/germeval21_test.csv")
    hate_train       = pd.read_csv("dataset/sentiment/clean/hate_train.csv")
    hate_test        = pd.read_csv("dataset/sentiment/clean/hate_test.csv")

    germeval18_train = germeval18_train.dropna(subset=["offensive"])
    germeval18_test  = germeval18_test.dropna(subset=["offensive"])
    germeval18_train["offensive"] = germeval18_train["offensive"].astype(int)
    germeval18_test["offensive"] = germeval18_test["offensive"].astype(int)

    germeval21_train = germeval21_train.dropna(subset=["toxic"])
    germeval21_test  = germeval21_test.dropna(subset=["toxic"])
    germeval21_train["toxic"] = germeval21_train["toxic"].astype(int)
    germeval21_test["toxic"] = germeval21_test["toxic"].astype(int)

    hate_train = hate_train.dropna(subset=["label"])
    hate_test  = hate_test.dropna(subset=["label"])
    hate_train["label"] = hate_train["label"].astype(int)
    hate_test["label"] = hate_test["label"].astype(int)

    batch_size = 16
    dl_train = {
        "offense": create_dataloader(germeval18_train, "cleaned_text", "offensive", tokenizer, batch_size),
        "toxic":   create_dataloader(germeval21_train, "cleaned_text", "toxic", tokenizer, batch_size),
        "hate":    create_dataloader(hate_train, "cleaned_text", "label", tokenizer, batch_size)
    }
    dl_val = {
        "offense": create_dataloader(germeval18_test, "cleaned_text", "offensive", tokenizer, batch_size, shuffle = False),
        "toxic":   create_dataloader(germeval21_test, "cleaned_text", "toxic", tokenizer, batch_size, shuffle = False),
        "hate":    create_dataloader(hate_test, "cleaned_text", "label", tokenizer, batch_size, shuffle = False)
    }

    class_weights = {
        "offense": torch.tensor([1.0, (len(germeval18_train) / germeval18_train["offensive"].sum()) - 1], dtype = torch.float32).to(device),
        "toxic":   torch.tensor([1.0, (len(germeval21_train) / germeval21_train["toxic"].sum()) - 1], dtype = torch.float32).to(device),
        "hate":    torch.tensor([1.0, 1.0, 1.0], dtype = torch.float32).to(device)  
    }

    model = MultiHeadGBERT()
    model.to(device)

    train_multitask(model, dl_train, dl_val, class_weights, device, epochs=5, patience=2)
    print("Training complete.")

if __name__ == "__main__":
    main()
