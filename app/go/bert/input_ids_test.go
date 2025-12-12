package main

import (
	"fmt"
	"log"
	"reflect"
	"testing"

	"github.com/sugarme/tokenizer/pretrained"
)

/*
Pythonで期待値を作成
from transformers import BertTokenizerFast
DIR = "/workspace/app/data/bert-sst2"
tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)

text = "This movie is great!"

inputs = tokenizer(

	text,
	padding="max_length",
	truncation=True,
	max_length=128,
	return_tensors="np",

)

print(inputs["input_ids"][0].tolist())

=> [101, 2023, 3185, 2003, 2307, 999, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
*/
func TestInputIDs(t *testing.T) {
	const modelDir = "/workspace/app/data/bert-sst2"

	// 1) tokenizer.json を直接読み込む
	configFile := modelDir + "/tokenizer.json"

	tokenizer, err := pretrained.FromFile(configFile)
	if err != nil {
		log.Fatalf("failed to load tokenizer.json: %v", err)
	}
	text := "This movie is great!"
	act, err := encodeSingle(tokenizer, text, 128)
	if err != nil {
		log.Fatalf("failed to encodeSingle: %v", err)
	}
	fmt.Println("raw tokens:", act.InputIDs)
	expect := []int64{
		101, 2023, 3185, 2003, 2307, 999, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	}
	if !reflect.DeepEqual(act.InputIDs, expect) {
		t.Fatalf("unexpected input_ids")
	}
}
