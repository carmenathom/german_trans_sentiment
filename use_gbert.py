import torch
from transformers import AutoTokenizer, AutoModel
from torch import nn

class MultiTaskGBERT(nn.Module):
    def __init__(
        self,
        pretrained_model="deepset/gbert-base",
        num_labels_offense = 2,
        num_labels_toxic = 2,
        num_labels_hate = 3
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(pretrained_model)
        hidden_size = self.bert.config.hidden_size

        self.offense = nn.Linear(hidden_size, num_labels_offense)
        self.toxic = nn.Linear(hidden_size, num_labels_toxic)
        self.hate = nn.Linear(hidden_size, num_labels_hate)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]  

        return {
            "offense": self.offense(cls),
            "toxic": self.toxic(cls),
            "hate": self.hate(cls),
        }

def predict(sentence: str):
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

    softmax = torch.nn.functional.softmax

    offense = torch.argmax(softmax(outputs["offense"], dim = 1)).item()
    toxic   = torch.argmax(softmax(outputs["toxic"], dim = 1)).item()
    hate    = torch.argmax(softmax(outputs["hate"], dim = 1)).item()

    return offense, toxic, hate

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiTaskGBERT()
    state_dict = torch.load("gbert_model.pt", map_location=device)

    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-base")

    sentences = [
        "Ich liebe Trans Leute.",
        "Ich hasse Trans Leute.",
    ]

    for s in sentences:
        o, t, h = predict(s)
        print(f"Sentence: {s}")
        print(f"  Offense = {o}")
        print(f"  Toxic   = {t}")
        print(f"  Hate    = {h}")
        print()

if __name__ == __main__():
    main()