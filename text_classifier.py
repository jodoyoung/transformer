from datasets import load_dataset

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