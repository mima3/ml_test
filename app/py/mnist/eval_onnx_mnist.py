import numpy as np
import onnxruntime as ort


def main():
    # ====== npz からテストデータ読み込み ======
    data = np.load("/workspace/app/data/mnist_test_normalized.npz")
    x = data["x"].astype("float32")  # [N,1,28,28] NCHW（PyTorch前処理済み）
    y = data["y"].astype("int64")    # [N]

    print("loaded test set:", x.shape, y.shape)

    # ====== ONNX モデル読み込み ======
    sess = ort.InferenceSession(
        "/workspace/app/data/mnist_cnn.onnx",
        providers=["CPUExecutionProvider"],  # 必要なら CUDAExecutionProvider など
    )

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print(f"ONNX input name: {input_name}, output name: {output_name}")

    # ====== バッチで精度評価 ======
    batch_size = 256
    correct = 0
    total = x.shape[0]

    for i in range(0, total, batch_size):
        xb = x[i:i + batch_size]      # [B,1,28,28] そのまま NCHW でOK（PyTorch→ONNX）
        yb = y[i:i + batch_size]      # [B]

        # ONNX Runtime は numpy array を受け取る
        logits = sess.run([output_name], {input_name: xb})[0]  # [B,10]
        pred = np.argmax(logits, axis=1)                       # [B]

        correct += (pred == yb).sum()

    acc = correct / total
    print(f"Test accuracy (ONNX, from npz): {acc:.4f}")


if __name__ == "__main__":
    main()
