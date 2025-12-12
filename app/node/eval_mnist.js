// mnist_eval_onnx.mjs

import fs from "node:fs";
import path from "node:path";
import JSZip from "jszip";
import { load } from "npyjs";
import * as ort from "onnxruntime-node";

const channels = 1;
const height = 28;
const width = 28;
const numClasses = 10;

/**
 * .npz を読み込み、内部の .npy を npyjs でパースして { x, y } を返す
 * Go版: xFlat (float32), y (int64) に相当
 */
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
  if (!result.x || !result.y) {
    throw new Error("npz 内に x.npy / y.npy が見つかりませんでした");
  }

  return {
    x: result.x, // { data: TypedArray, shape: [...], ... }
    y: result.y,
  };
}

/**
 * Go版と同じロジックで ONNX Runtime による精度評価
 */
async function main() {
  const npzPath = "/workspace/app/data/mnist_test_normalized.npz";
  const onnxPath = "/workspace/app/data/mnist_cnn.onnx";

  // ====== npz から x,y を読み込み ======
  const { x, y } = await loadNPZ(npzPath);

  const xFlat = x.data; // 1次元の float32 or float64 など
  const yRaw = y.data;  // おそらく int64 → BigInt64Array

  // y は BigInt の可能性があるので Number に落とす（0〜9なので安全）
  const labels = Array.from(yRaw, (v) => Number(v));

  const numSamples = labels.length;
  if (numSamples === 0) {
    throw new Error("empty test set");
  }

  const imgSize = channels * height * width;

  if (xFlat.length !== numSamples * imgSize) {
    throw new Error(
      `unexpected x size: got ${xFlat.length}, want ${
        numSamples * imgSize
      } (= ${numSamples} * ${imgSize})`
    );
  }

  console.log(
    `loaded test set: N=${numSamples}, xFlat=${xFlat.length}, y=${labels.length}`
  );

  // ====== ONNX Runtime セッション作成 ======
  const session = await ort.InferenceSession.create(onnxPath);

  // 1枚ぶんの入力バッファ [1,1,28,28]
  const inputTensorData = new Float32Array(imgSize);

  let correct = 0;

  for (let i = 0; i < numSamples; i++) {
    // xFlat から i枚目を inputTensorData にコピー
    const start = i * imgSize;
    const end = start + imgSize;

    // TypedArray 同士なので subarray + set でコピー
    inputTensorData.set(xFlat.subarray(start, end));

    // ONNX Runtime の Tensor を作成
    const inputTensor = new ort.Tensor("float32", inputTensorData, [
      1,
      channels,
      height,
      width,
    ]);

    // 入力名 "input" / 出力名 "logits" は Go 版と同じ前提
    const feeds = { input: inputTensor };

    const results = await session.run(feeds);
    const logitsTensor = results.logits;
    if (!logitsTensor) {
      throw new Error("output 'logits' が見つかりません");
    }

    const logits = logitsTensor.data;
    if (logits.length !== numClasses) {
      throw new Error(
        `unexpected logits len: ${logits.length} (expected ${numClasses})`
      );
    }

    // ====== argmax ======
    let bestIdx = 0;
    let bestVal = -Infinity;
    for (let c = 0; c < numClasses; c++) {
      const v = logits[c];
      if (v > bestVal) {
        bestVal = v;
        bestIdx = c;
      }
    }

    if (bestIdx === labels[i]) {
      correct++;
    }

    // たとえば進捗を出したければこんな感じ（不要なら消してOK）
    // if ((i + 1) % 1000 === 0) {
    //   console.log(`processed ${i + 1}/${numSamples}`);
    // }
  }

  const acc = correct / numSamples;
  console.log(`Test accuracy (Node.js + ONNX, from npz): ${acc.toFixed(4)}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
