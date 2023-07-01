package main

import (
	"fmt"
	"io/ioutil"
	"log"

	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/x/gorgonnx"
	"gorgonia.org/tensor"
)

// go mod init onnx-inference
// go mod tidy
func main() {

	backend := gorgonnx.NewGraph()

	model := onnx.NewModel(backend)

	b, _ := ioutil.ReadFile("model.onnx")

	err := model.UnmarshalBinary(b)
	if err != nil {
		log.Fatal(err)
	}

	input := tensor.New(tensor.WithShape(1, 4), tensor.WithBacking([]float32{0.2, 0.4, 0.34, 0.5}))
	model.SetInput(0, input)
	err = backend.Run()
	if err != nil {
		log.Fatal(err)
	}

	output, _ := model.GetOutputTensors()

	fmt.Println(output[0])

}
