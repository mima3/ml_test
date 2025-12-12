// npz_extract_npy.js
import fs from "node:fs";
import path from "node:path";
import JSZip from "jszip";
import { load } from "npyjs";

async function loadNPZ(npzPath) {
  // ====== npz(zip) を読む ======
  const npzData = fs.readFileSync(npzPath);
  const zip = await JSZip.loadAsync(npzData);
  const result = {};

  // ====== zip 内のすべての .npy を読み取る ======
  for (const filename of Object.keys(zip.files)) {
    if (!filename.endsWith(".npy")) continue;

    const file = zip.files[filename];
    const npyBuffer = await file.async("uint8array");

    // npyjs で decode
    const npyArray = await load(npyBuffer);

    // filename → 配列データを格納
    // 例: "x.npy" → result["x"] = { data, shape, dtype }
    const key = path.basename(filename, ".npy");
    result[key] = npyArray;
  }

  return result;
}

// ====== 実行 ======
(async () => {
  const npzPath = "/workspace/app/data/mnist_test_normalized.npz";

  const arrays = await loadNPZ(npzPath);

  console.log("Keys in NPZ:", Object.keys(arrays));
  // => ["x", "y"]

  const x = arrays.x; // {data, shape, dtype, fortranOrder}
  const y = arrays.y;

  console.log("x.shape:", x.shape);
  console.log("y.shape:", y.shape);

  // 例: x.data は Float32Array
  console.log("x dtype:", x.dtype);
  console.log("y dtype:", y.dtype);

  // 1枚目の画像データ（28*28）
  console.log("First 10 elements of x.data:", x.data.slice(0, 10));
})();
