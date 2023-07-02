package inference

import (
	"io/ioutil"
	"log"

	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/x/gorgonnx"
	"gorgonia.org/tensor"
)

func Infere(model_name string, input *tensor.Dense) []tensor.Tensor {
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

	logits, _ := model.GetOutputTensors()

	return logits
}
