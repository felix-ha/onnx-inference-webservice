package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"onnx-inference/inference"
	"reflect"
	"runtime"

	"gorgonia.org/tensor"
)

type RequestData struct {
	Model string    `json:"model"`
	Input []float32 `json:"input"`
}

type ResponseData struct {
	Info   string
	Logits []float32
}

func runInference(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read request body", http.StatusInternalServerError)
		return
	}

	var data RequestData
	err = json.Unmarshal(body, &data)
	if err != nil {
		http.Error(w, "Failed to parse JSON data", http.StatusBadRequest)
		return
	}

	model := data.Model
	input_raw := data.Input

	input := tensor.New(tensor.WithShape(1, 4), tensor.WithBacking(input_raw))
	logits := inference.Infere(model, input)

	w.Header().Set("Content-Type", "application/json")
	post := &ResponseData{
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
