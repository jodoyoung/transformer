from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

emotions = load_dataset("emotion")
print(emotions)

train_ds = emotions["train"]
print(train_ds)
print(len(train_ds))
print(train_ds[0])
print(train_ds.column_names)
print(train_ds.features)
print(train_ds[:5])
print(train_ds["text"][:5])

emotions.set_format(type="pandas")
df = emotions["train"][:]
print(df.head())

def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)

df["label_name"] = df["label"].apply(label_int2str)
print(df.head())

df["label_name"].value_counts(ascending=True).plot.barh()
plt.title("Frequency of Classes")
plt.show()

df["Words Per Tweet"] = df["text"].str.split().apply(len)
df.boxplot("Words Per Tweet", by="label_name", grid=False,
           color="black", showfliers=False)
plt.suptitle("")
plt.xlabel("")
plt.show()

emotions.reset_format()

text = "Tokenizing text is a core task of NLP"
tokenized_txt = list(text)
print(tokenized_txt)

token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_txt)))}
print(token2idx)

input_ids = [token2idx[token] for token in tokenized_txt]
print(input_ids)

categorical_df = pd.DataFrame(
    {"Name": ["Bumblebee", "Optimus Prime", "Megatron"], "Label ID": [0,1,2]}
)
pd.get_dummies(categorical_df["Name"])

input_ids = torch.tensor(input_ids)
one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))
one_hot_encodings.shape

print(f"토큰: {tokenized_txt[0]}")
print(f"인덱스: {input_ids[0]}")
print(f"원-핫 인코딩: {one_hot_encodings[0]}")