# export_to_onnx.py
import torch
from torch.export import Dim
from train_pytorch_mnist import SimpleCNN

def main():
    device = torch.device("cpu")
    model = SimpleCNN().to(device)
    state = torch.load("/workspace/app/data/mnist_cnn.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 1, 28, 28, device=device)

    # forward(self, x) の x に対応するのでキー名は "x"
    batch = Dim("batch")
    dynamic_shapes = {
        "x": {0: batch},  # 0次元目（バッチ）を動的にする
    }

    torch.onnx.export(
        model,
        dummy,
        "/workspace/app/data/mnist_cnn.onnx",
        input_names=["input"],
        output_names=["logits"],
        dynamic_shapes=dynamic_shapes,
        # dynamo=True がデフォルトなので指定しなくてもOK
    )

if __name__ == "__main__":
    main()
