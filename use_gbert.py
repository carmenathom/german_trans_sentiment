import torch
from transformers import AutoTokenizer, AutoModel
from torch import nn


class MultiHeadGBERT(nn.Module):
    def __init__(self, pretrained_model="deepset/gbert-base",
                 num_labels_offense=2, num_labels_toxic=2, num_labels_hate=3):
        super().__init__()
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


def predict(sentence, model, tokenizer, device):
    encoded = tokenizer(
        sentence,
        truncation = True,
        padding = True,
        max_length = 128,
        return_tensors = "pt"
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    offense = torch.argmax(outputs["offense"], dim=1).item()
    toxic   = torch.argmax(outputs["toxic"],   dim=1).item()
    hate    = torch.argmax(outputs["hate"],    dim=1).item()

    return offense, toxic, hate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiHeadGBERT()
    model.load_state_dict(torch.load("gbert_model.pt", map_location=device))
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-base")

    sentences = [
        "Ein normal Satz.",
        "Ich hasse Trans Leute.",
    ]

    for s in sentences:
        o, t, h = predict(s, model, tokenizer, device)
        print(f"Sentence: {s}")
        print(f"  Offense = {o}")
        print(f"  Toxic   = {t}")
        print(f"  Hate    = {h}\n")


if __name__ == "__main__":
    main()
