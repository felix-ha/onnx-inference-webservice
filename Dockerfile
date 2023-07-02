FROM golang:1.13

ADD . /go/src/github.com/onnx-inference
WORKDIR /go/src/github.com/onnx-inference

RUN go mod tidy
RUN go install onnx-inference

ENTRYPOINT /go/bin/onnx-inference

EXPOSE 8080