import re
import pandas as pd

def load_hatr_dataset(path):
    data = []
    
    current_label = None
    current_text = []

    label_pattern = re.compile(r'^(n|p|hs)\t')

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip("\n")

            if label_pattern.match(line):
                if current_label is not None:
                    data.append({"label": current_label, "text": " ".join(current_text)})

                parts = line.split("\t", 1)
                current_label = parts[0]
                current_text = [parts[1] if len(parts) > 1 else ""]

            else:
                if current_label is not None:
                    current_text.append(line)

        if current_label is not None:
            data.append({"label": current_label, "text": " ".join(current_text)})

    return pd.DataFrame(data)

def clean_text(text):
    if not isinstance(text, str):
        return text
    
    text = text.replace("|LBR|", " ")
    text = text.replace("|TAB|", " ") 

    text = text.replace("|||", " ")
    text = text.replace("||", " ")

    text = re.sub(r"<EOS>|<END>|<s>|</s>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", "@USER", text)
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def main():
    germeval18_train = pd.read_csv("dataset/sentiment/GermEval18_TrainData.txt", 
                                   quotechar = '"', on_bad_lines = "skip", 
                                   escapechar="\\", engine="python", 
                                   sep = r"\t+", header = None)
    germeval18_test = pd.read_csv("dataset/sentiment/GermEval18_TestData.txt", 
                                  quotechar = '"', on_bad_lines = "skip", 
                                  sep = r"\t+", escapechar = "\\", 
                                  engine = "python", header = None)
    germeval21_train  = pd.read_csv("dataset/sentiment/GermEval21_TrainData.csv", 
                                    quotechar = '"', on_bad_lines = "skip", 
                                    escapechar="\\", engine="python", 
                                    sep = ",", header = 0)
    germeval21_test = pd.read_csv("dataset/sentiment/GermEval21_TrainData.csv", 
                                  quotechar = '"', on_bad_lines = "skip", 
                                  escapechar="\\", engine="python", 
                                  sep = ",", header = 0)
    hasoc = pd.read_csv("dataset/sentiment/HASOC.csv", 
                                  quotechar = '"', on_bad_lines = "skip", 
                                  escapechar="\\", engine="python", 
                                  sep = "\t", header = None)
    polly = pd.read_csv("dataset/sentiment/POLLY.csv", 
                                  quotechar = '"', on_bad_lines = "skip", 
                                  escapechar="\\", engine="python", 
                                  sep = "\t", header = 0)
    hatr = load_hatr_dataset("dataset/sentiment/hatr.csv")

    germeval18_train = germeval18_train.drop(2, axis=1)
    germeval18_test = germeval18_test.drop(2, axis=1)
    germeval21_train = germeval21_train.drop(columns = ["comment_id", "Sub2_Engaging", "Sub3_FactClaiming"])
    germeval21_test = germeval21_test.drop(columns = ["comment_id", "Sub2_Engaging", "Sub3_FactClaiming"])

    germeval18_train.columns = ["text", "offensive"]
    germeval18_test.columns = ["text", "offensive"]
    germeval21_train.columns = ["text", "toxic"]
    germeval21_test.columns = ["text", "toxic"]
    hasoc.columns = ["label", "text"]
    polly.columns = ["label", "text"]

    offensive_mapping = {"OTHER": 0, "OFFENSE": 1}
    label_mapping = {"n": 0, "p": 1, "hs": 2}

    germeval18_train["offensive"] = germeval18_train["offensive"].map(offensive_mapping)
    germeval18_test["offensive"] = germeval18_test["offensive"].map(offensive_mapping)
    hasoc["label"] = hasoc["label"].map(label_mapping)
    polly["label"] = polly["label"].map(label_mapping)
    hatr["label"] = hatr["label"].map(label_mapping)
    
    germeval18_train["cleaned_text"] = germeval18_train["text"].apply(clean_text)
    germeval18_test["cleaned_text"] = germeval18_test["text"].apply(clean_text)
    germeval21_train["cleaned_text"] = germeval21_train["text"].apply(clean_text)
    germeval21_test["cleaned_text"] = germeval21_test["text"].apply(clean_text)
    hasoc["cleaned_text"] = hasoc["text"].apply(clean_text)
    polly["cleaned_text"] = polly["text"].apply(clean_text)
    hatr["cleaned_text"] = hatr["text"].apply(clean_text)

    hate_df = pd.concat([hasoc, polly, hatr], ignore_index = True)
    shuffle_hate_df = hate_df.sample(frac = 1)
    train_size = int(0.7 * len(shuffle_hate_df))

    hate_train = shuffle_hate_df[:train_size]
    hate_test = shuffle_hate_df[train_size:]

    germeval18_train.to_csv('dataset/sentiment/clean/germeval18_train.csv', index=False) 
    germeval18_test.to_csv('dataset/sentiment/clean/germeval18_test.csv', index=False) 
    germeval21_train.to_csv('dataset/sentiment/clean/germeval21_train.csv', index=False) 
    germeval21_test.to_csv('dataset/sentiment/clean/germeval21_test.csv', index=False) 
    hate_train.to_csv('dataset/sentiment/clean/hate_train.csv', index=False) 
    hate_test.to_csv('dataset/sentiment/clean/hate_test.csv', index=False) 

if __name__ == "__main__":
    main()