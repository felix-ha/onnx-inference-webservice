.PHONY: docker_build
docker_build:
	docker build --tag onnx .

.PHONY: docker_run
docker_run:
	docker run --publish 80:8080 --name onnx --rm onnx

.PHONY: request
request:
	curl -H "Content-Type: application/json" -d '{"model":"model.onnx","x":[0.8,0.071,-0.28,0.148]}' -i http://0.0.0.0:80/inference
