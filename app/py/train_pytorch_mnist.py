# train_pytorch_mnist.py
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from simple_cnn import SimpleCNN
import numpy as np  # ★ 追加


def main():
    # 学習用の前処理（PyTorch側の標準的なMNIST前処理）
    mean = 0.1307
    std  = 0.3081
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])

    # train / test をきちんと分ける
    train_ds = datasets.MNIST("/workspace/app/data/mnist", train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST("/workspace/app/data/mnist", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ====== 学習 ======
    model.train()
    for epoch in range(1):  # デモなので1エポックだけ
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
        print(f"epoch {epoch+1} done, loss={loss.item():.4f}")

    # ====== テストデータでPyTorch自身の評価 ======
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f"Test accuracy (PyTorch): {acc:.4f}")

    # ====== テストデータ（前処理済み）を汎用形式で保存 ======
    xs = []
    ys = []
    for i in range(len(test_ds)):
        img, label = test_ds[i]          # img: torch.Tensor [1,28,28] (Normalize 済み)
        xs.append(img.numpy())
        ys.append(label)

    xs = np.stack(xs, axis=0).astype("float32")      # [10000,1,28,28]
    ys = np.array(ys, dtype="int64")                 # [10000]

    np.savez_compressed(
        "/workspace/app/data/mnist_test_normalized.npz",
        x=xs,
        y=ys,
        mean=np.array([mean], dtype="float32"),
        std=np.array([std], dtype="float32"),
    )
    print("saved /workspace/app/data/mnist_test_normalized.npz")

    # ====== モデル保存 ======
    torch.save(model.state_dict(), "/workspace/app/data/mnist_cnn.pth")

if __name__ == "__main__":
    main()
