## python

### mnistのサンプルコード

```bash
cd app/py/mnist
# pytorchでトレーニング
python train_pytorch_mnist.py
# onnx形式で出力
python export_to_onnx.py
# onnxで推論
python eval_onnx_mnist.py
```

### bertのサンプルコード

```bash
cd app/py/bert
# pytorchでトレーニング(CPUで実行するため学習データを大幅に間引いています)
python train_bert.py
# onnx形式で出力
python export_to_onnx.py
# onnxで推論
python check_onnx.py
text: This movie is great!
  logits: [[-0.11655042  0.8542732 ]]
  pred_id: 1 -> positive
----------------------------------------
text: This movie is terrible.
  logits: [[ 0.07142931 -0.6349491 ]]
  pred_id: 0 -> negative
----------------------------------------
text: I really loved this film.
  logits: [[-0.16516581  0.46208158]]
  pred_id: 1 -> positive
----------------------------------------
text: I really hated this film.
  logits: [[ 0.04016718 -0.7027098 ]]
  pred_id: 0 -> negative
----------------------------------------
text: The plot was boring and slow.
  logits: [[ 0.01931576 -0.84986705]]
  pred_id: 0 -> negative
----------------------------------------
```

## go

### mnistのサンプルコード

```bash
cd app/go/mnist
# 実行時（正確にはビルド時に必要なファイルをダウンロードしている)
go run eval_mnist.go
```

### bertのサンプルコード

```bash
cd app/go/bert
# 関数の動作確認
go test -v
# 推論
go run eval_bert.go
```

## node.js

```
cd app/node
npm install
```

### mnistのサンプルコード

```
node eval_mnist.js
``` 

### bertのサンプルコード

```
node eval_bert.js
```
