package inference

import (
	"io/ioutil"
	"log"
	"math"

	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/x/gorgonnx"
	"gorgonia.org/tensor"
)

// Infere runs a inference for the given model and assumes batch size one.
func Infere(model_name string, input *tensor.Dense) []float32 {
	backend := gorgonnx.NewGraph()

	model := onnx.NewModel(backend)

	b, _ := ioutil.ReadFile(model_name)

	err := model.UnmarshalBinary(b)
	if err != nil {
		log.Fatal(err)
	}

	model.SetInput(0, input)
	err = backend.Run()
	if err != nil {
		log.Fatal(err)
	}

	logits_raw, _ := model.GetOutputTensors()
	logits := logits_raw[0].Data().([]float32)

	return softmax(logits)
}

func softmax(input []float32) []float32 {
	var sum float32
	for i := range input {
		input[i] = float32(math.Exp(float64(input[i])))
		sum += input[i]
	}

	for i := range input {
		input[i] /= sum
	}

	return input
}
