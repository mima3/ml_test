package main

import (
	"fmt"
	"log"
	"math"

	"codeberg.org/sbinet/npyio/npz"
	ort "github.com/yalue/onnxruntime_go"
)

const (
	channels   = int64(1)
	height     = int64(28)
	width      = int64(28)
	numClasses = int64(10)
)

func main() {
	// ====== npz からテストデータ読み込み ======
	r, err := npz.Open("/workspace/app/data/mnist_test_normalized.npz")
	if err != nil {
		log.Fatalf("failed to open npz: %v", err)
	}
	defer r.Close()
	var xFlat []float32 // [N * (1*28*28)]
	if err := r.Read("x.npy", &xFlat); err != nil {
		log.Fatalf("failed to read x from npz: %v", err)
	}

	var y []int64 // [N]
	if err := r.Read("y.npy", &y); err != nil {
		log.Fatalf("failed to read y from npz: %v", err)
	}

	numSamples := len(y)
	if numSamples == 0 {
		log.Fatal("empty test set")
	}
	imgSize := int(channels * height * width)
	if len(xFlat) != numSamples*imgSize {
		log.Fatalf("unexpected x size: got %d, want %d (= %d * %d)",
			len(xFlat), numSamples*imgSize, numSamples, imgSize)
	}

	fmt.Printf("loaded test set: N=%d, xFlat=%d, y=%d\n",
		numSamples, len(xFlat), len(y))

	// ====== ONNX Runtime 初期化 ======
	// libonnxruntime.so の場所は環境に合わせて変更してください
	ort.SetSharedLibraryPath("/usr/local/lib/libonnxruntime.so")

	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatalf("failed to init onnxruntime env: %v", err)
	}
	defer func() {
		if err := ort.DestroyEnvironment(); err != nil {
			log.Printf("failed to destroy env: %v", err)
		}
	}()

	// ====== 1枚ずつ推論するシンプルな形にする ======
	// 入力: [1,1,28,28]
	inputShape := ort.NewShape(1, channels, height, width)
	inputData := make([]float32, imgSize) // 1枚ぶんのバッファ
	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		log.Fatalf("failed to create input tensor: %v", err)
	}
	defer inputTensor.Destroy()

	// 出力: [1,10]
	outputShape := ort.NewShape(1, numClasses)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		log.Fatalf("failed to create output tensor: %v", err)
	}
	defer outputTensor.Destroy()

	// ====== セッション作成 ======
	session, err := ort.NewAdvancedSession(
		"/workspace/app/data/mnist_cnn.onnx",
		[]string{"input"},  // ← sess.get_inputs()[0].name の値
		[]string{"logits"}, // ← sess.get_outputs()[0].name の値
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
		nil, // SessionOptions（デフォルトでOKなら nil）
	)
	if err != nil {
		log.Fatalf("failed to create session: %v", err)
	}
	defer func() {
		if err := session.Destroy(); err != nil {
			log.Printf("failed to destroy session: %v", err)
		}
	}()

	// ====== 全サンプルで精度評価（1枚ずつ Run） ======
	correct := 0

	for i := 0; i < numSamples; i++ {
		// xFlat から i枚目を inputData にコピー
		start := i * imgSize
		end := start + imgSize
		copy(inputData, xFlat[start:end])

		// ネットワーク1回実行（inputData の中身を読んで outputTensor を更新）
		if err := session.Run(); err != nil {
			log.Fatalf("failed to run session: %v", err)
		}

		// 出力は [1,10] → 長さ10のスライスとして取得
		logits := outputTensor.GetData()
		if len(logits) != int(numClasses) {
			log.Fatalf("unexpected logits len: %d", len(logits))
		}

		// argmax
		bestIdx := 0
		bestVal := float32(math.Inf(-1))
		for c := 0; c < int(numClasses); c++ {
			if logits[c] > bestVal {
				bestVal = logits[c]
				bestIdx = c
			}
		}

		if int64(bestIdx) == y[i] {
			correct++
		}
	}

	acc := float64(correct) / float64(numSamples)
	fmt.Printf("Test accuracy (Go + ONNX, from npz): %.4f\n", acc)
}
