package main

import (
	"fmt"
	"log"
	"math"

	ort "github.com/yalue/onnxruntime_go"

	tk "github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
)

const (
	onnxModelPath = "/workspace/app/data/bert-sst2.onnx"
	maxLen        = 128
)

var labelNames = []string{"negative", "positive"}

// ---------- softmax & argmax ----------

func softmax(logits []float32) []float32 {
	maxVal := float32(math.Inf(-1))
	for _, v := range logits {
		if v > maxVal {
			maxVal = v
		}
	}

	exps := make([]float64, len(logits))
	var sum float64
	for i, v := range logits {
		e := math.Exp(float64(v - maxVal))
		exps[i] = e
		sum += e
	}

	probs := make([]float32, len(logits))
	for i, e := range exps {
		probs[i] = float32(e / sum)
	}
	return probs
}

func argmax(v []float32) int {
	idx := 0
	maxVal := v[0]
	for i := 1; i < len(v); i++ {
		if v[i] > maxVal {
			maxVal = v[i]
			idx = i
		}
	}
	return idx
}

// ---------- BERT トークナイズ ----------

type Encoded struct {
	InputIDs      []int64
	AttentionMask []int64
}

func encodeSingle(t *tk.Tokenizer, text string, maxLen int) (*Encoded, error) {
	enc, err := t.EncodeSingle(text)
	if err != nil {
		return nil, err
	}

	// WordPiece の ID 列（CLS/SEP なし）
	ids := enc.Ids // []int

	// 特殊トークン ID（これも int）
	clsID, _ := t.TokenToId("[CLS]")
	sepID, _ := t.TokenToId("[SEP]")
	padID, _ := t.TokenToId("[PAD]")

	// [CLS] tokens... [SEP] を組み立てる（全部 int）
	full := make([]int, 0, len(ids)+2)
	full = append(full, clsID)
	full = append(full, ids...)
	full = append(full, sepID)

	// max_length=128 に合わせて truncate
	if len(full) > maxLen {
		full = full[:maxLen]
	}

	inputIDs := make([]int64, maxLen)
	attnMask := make([]int64, maxLen)

	// トークン部をコピー（int -> int64）
	copyLen := len(full)
	if copyLen > maxLen {
		copyLen = maxLen
	}
	for i := 0; i < copyLen; i++ {
		inputIDs[i] = int64(full[i])
		attnMask[i] = 1
	}

	// 残りを PAD で埋める
	for i := copyLen; i < maxLen; i++ {
		inputIDs[i] = int64(padID)
		attnMask[i] = 0
	}

	return &Encoded{
		InputIDs:      inputIDs,
		AttentionMask: attnMask,
	}, nil
}

func main() {
	// ====== ONNX Runtime 初期化 ======
	// libonnxruntime.so の場所は環境に合わせて変更してください
	ort.SetSharedLibraryPath("/usr/local/lib/libonnxruntime.so")
	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatalf("InitializeEnvironment error: %v", err)
	}
	defer ort.DestroyEnvironment()

	// ===== 1. Tokenizer (bert-base-uncased 相当) 読み込み =====
	tokenizer := pretrained.BertBaseUncased()
	if tokenizer == nil {
		log.Fatal("failed to load BertBaseUncased tokenizer")
	}

	// Python サンプルと同じテキスト
	texts := []string{
		"This movie is great!",
		"This movie is terrible.",
		"I really loved this film.",
		"I really hated this film.",
		"The plot was boring and slow.",
	}

	// ===== 2. 入力・出力 Tensor を用意（batch=1 を繰り返し使い回す） =====
	// shape [1, maxLen]
	inputShape := ort.NewShape(1, int64(maxLen))

	inputIDsTensor, err := ort.NewEmptyTensor[int64](inputShape)
	if err != nil {
		log.Fatalf("NewEmptyTensor(input_ids) error: %v", err)
	}
	defer inputIDsTensor.Destroy()

	attentionTensor, err := ort.NewEmptyTensor[int64](inputShape)
	if err != nil {
		log.Fatalf("NewEmptyTensor(attention_mask) error: %v", err)
	}
	defer attentionTensor.Destroy()

	// shape [1, 2]（2クラス: neg/pos）
	outputShape := ort.NewShape(1, 2)
	logitsTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		log.Fatalf("NewEmptyTensor(logits) error: %v", err)
	}
	defer logitsTensor.Destroy()

	// AdvancedSession を作成
	session, err := ort.NewAdvancedSession(
		onnxModelPath,
		[]string{"input_ids", "attention_mask"},
		[]string{"logits"},
		[]ort.Value{inputIDsTensor, attentionTensor},
		[]ort.Value{logitsTensor},
		nil, // SessionOptions: 必要ならここで設定
	)
	if err != nil {
		log.Fatalf("NewAdvancedSession error: %v", err)
	}
	defer session.Destroy()

	// ===== 3. 各テキストについて Python と同じように推論 =====
	inputIDsData := inputIDsTensor.GetData()
	attentionData := attentionTensor.GetData()
	logitsData := logitsTensor.GetData() // len 2

	for _, text := range texts {
		enc, err := encodeSingle(tokenizer, text, maxLen)
		if err != nil {
			log.Fatalf("encodeSingle error: %v", err)
		}

		// batch=1 なのでそのままコピー
		copy(inputIDsData, enc.InputIDs)
		copy(attentionData, enc.AttentionMask)

		if err := session.Run(); err != nil {
			log.Fatalf("session.Run error: %v", err)
		}

		// logitsData は [2] (1行分)
		row := make([]float32, len(logitsData))
		copy(row, logitsData)

		probs := softmax(row)
		pred := argmax(row)

		fmt.Printf("text: %s\n", text)
		fmt.Printf("  logits: %v\n", row)
		fmt.Printf("  probs : %v\n", probs)
		fmt.Printf("  pred  : %d -> %s\n", pred, labelNames[pred])
		fmt.Println("----------------------------------------")
	}

	fmt.Println("DONE")
}
