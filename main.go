package main

import (
	"fmt"
	"onnx-inference/inference"

	"gorgonia.org/tensor"
)

func main() {
	model := "model.onnx"
	input := tensor.New(tensor.WithShape(1, 4), tensor.WithBacking([]float32{0.2, 0.4, 0.34, 0.5}))
	logits := inference.Infere(model, input)

	fmt.Println(logits)
}
