## python

```bash
cd app/py
# pytorchでトレーニング
python train_pytorch_mnist.py
# onnx形式で出力
python export_to_onnx.py
# onnxで推論
python eval_onnx_mnist.py
```

## go

```bash
cd app/go 
# 実行時（正確にはビルド時に必要なファイルをダウンロードしている)
go run eval_mnist.go
```
