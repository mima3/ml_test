// bert_sst2_onnx.mjs

import * as ort from "onnxruntime-node";
import { AutoTokenizer } from "@huggingface/transformers";

const onnxModelPath = "/workspace/app/data/bert-sst2.onnx";
const maxLen = 128;
const labelNames = ["negative", "positive"];

// ---------- softmax & argmax ----------

function softmax(logits) {
  let maxVal = -Infinity;
  for (const v of logits) {
    if (v > maxVal) maxVal = v;
  }

  const exps = new Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - maxVal);
    exps[i] = e;
    sum += e;
  }

  const probs = new Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    probs[i] = exps[i] / sum;
  }
  return probs;
}

function argmax(v) {
  let idx = 0;
  let maxVal = v[0];
  for (let i = 1; i < v.length; i++) {
    if (v[i] > maxVal) {
      maxVal = v[i];
      idx = i;
    }
  }
  return idx;
}

// ---------- main ----------

async function main() {
  // 1. Tokenizer 読み込み（bert-base-uncased 相当）
  //
  // Transformers.js では:
  //   const tokenizer = await AutoTokenizer.from_pretrained('Xenova/bert-base-uncased');
  // のようにして BERT 系トークナイザをロードできます。
  //
  const tokenizer = await AutoTokenizer.from_pretrained(
    "Xenova/bert-base-uncased",
    {
      // デフォルトは BigInt ベースだが、ここでは通常の number 配列にしたいので無効化
      // （必要に応じて BigInt のままでも OK）
      useBigInt: false, // CSDN の transformers.js 記事にもあるオプション 
    }
  );

  // Python サンプルと同じテキスト
  const texts = [
    "This movie is great!",
    "This movie is terrible.",
    "I really loved this film.",
    "I really hated this film.",
    "The plot was boring and slow.",
  ];

  // 2. ONNX Runtime セッション作成
  const session = await ort.InferenceSession.create(onnxModelPath);

  for (const text of texts) {
    // 3. 1文ずつ BERT トークナイズ
    //
    // Go 版の encodeSingle では:
    //   - [CLS] tokens... [SEP]
    //   - maxLen=128 で truncate
    //   - pad を [PAD] で埋めて attention_mask=0
    //
    // Transformers.js では padding / truncation を指定することで
    // 同等の処理ができます。
    const encoded = await tokenizer(text, {
      max_length: maxLen,
      padding: "max_length",
      truncation: true,
      return_attention_mask: true,
    });

    // encoded.input_ids / encoded.attention_mask は
    // 「Tensor ライクなオブジェクト」で、.data に TypedArray が入る想定です。
    const inputIdsTyped = encoded.input_ids.data; // Uint32Array
    const attentionMaskTyped = encoded.attention_mask.data; // Uint8Array or Uint32Array

    // ONNX Runtime の int64 は BigInt64Array が必要なので変換
    const inputIds = BigInt64Array.from(
      Array.from(inputIdsTyped, (v) => BigInt(v))
    );
    const attentionMask = BigInt64Array.from(
      Array.from(attentionMaskTyped, (v) => BigInt(v))
    );

    if (inputIds.length !== maxLen || attentionMask.length !== maxLen) {
      throw new Error(
        `unexpected seq length: input_ids=${inputIds.length}, attention_mask=${attentionMask.length}`
      );
    }

    // 4. ONNX Runtime に Tensor を渡して推論
    const inputIdsTensor = new ort.Tensor("int64", inputIds, [1, maxLen]);
    const attentionMaskTensor = new ort.Tensor(
      "int64",
      attentionMask,
      [1, maxLen]
    );

    const feeds = {
      input_ids: inputIdsTensor,
      attention_mask: attentionMaskTensor,
    };

    const results = await session.run(feeds);
    const logitsTensor = results.logits;
    if (!logitsTensor) {
      throw new Error("output 'logits' is missing");
    }

    const logitsData = Array.from(logitsTensor.data); // Float32Array → 普通の配列に

    if (logitsData.length !== 2) {
      throw new Error(`unexpected logits size: ${logitsData.length}`);
    }

    const probs = softmax(logitsData);
    const pred = argmax(logitsData);

    console.log(`text: ${text}`);
    console.log(`  logits: ${logitsData}`);
    console.log(`  probs : ${probs.map((p) => p.toFixed(4))}`);
    console.log(`  pred  : ${pred} -> ${labelNames[pred]}`);
    console.log("----------------------------------------");
  }

  console.log("DONE");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
