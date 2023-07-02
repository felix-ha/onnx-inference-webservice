package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"onnx-inference/inference"
	"reflect"
	"runtime"

	"gorgonia.org/tensor"
)

type Data struct {
	Info   string
	Logits []float32
}

func runInference(w http.ResponseWriter, r *http.Request) {
	model := "model.onnx"
	input := tensor.New(tensor.WithShape(1, 4), tensor.WithBacking([]float32{0.2, 0.4, 0.34, 0.5}))
	logits := inference.Infere(model, input)

	w.Header().Set("Content-Type", "application/json")
	post := &Data{
		Info:   "done",
		Logits: logits,
	}
	json, _ := json.Marshal(post)
	w.Write(json)
}

func log(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		name := runtime.FuncForPC(reflect.ValueOf(h).Pointer()).Name()
		fmt.Println("Handler function called - " + name)
		h(w, r)
	}
}

func main() {

	server := http.Server{
		Addr: "0.0.0.0:8080",
	}

	http.HandleFunc("/inference", log(runInference))

	server.ListenAndServe()

}
