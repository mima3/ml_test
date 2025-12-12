import numpy as np
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments

MODEL_NAME = "bert-base-uncased"

# 1) データセット読み込み（SST-2）
dataset = load_dataset("glue", "sst2")

tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return tokenizer(
        batch["sentence"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean().item()
    return {"accuracy": acc}

encoded = dataset.map(tokenize_fn, batched=True)

encoded = encoded.remove_columns(["sentence", "idx"])
encoded = encoded.rename_column("label", "labels")
encoded.set_format("torch")

# 先頭数千件だけ使って軽く学習・評価
train_ds_full = encoded["train"]
eval_ds_full  = encoded["validation"]

train_ds = train_ds_full.select(range(500))   # 先頭 500 件だけ
eval_ds  = eval_ds_full.select(range(125))    # 先頭 125 件だけ

# 2) モデル読み込み
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# 3) 学習設定
args = TrainingArguments(
    output_dir="/workspace/app/data/bert-sst2",

    # 1 epoch だけ
    num_train_epochs=1.0,

    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,

    # 各 epoch 終わりで評価（eval_ds は 500 件に減ってるので軽い）
    eval_strategy="epoch",

    # checkpoint はいったん保存しない（ストレージと時間節約）
    save_strategy="no",

    # ログは 50 ステップごと
    logging_strategy="steps",
    logging_steps=50,

    # CPU-only & メモリ節約
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    compute_metrics=compute_metrics,
)

trainer.train()
metrics = trainer.evaluate()          # 明示的に呼んでもいい
print(metrics)
trainer.save_model("/workspace/app/data/bert-sst2")
tokenizer.save_pretrained("/workspace/app/data/bert-sst2")
