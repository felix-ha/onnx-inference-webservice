.PHONY: docker_build
docker_build:
	docker build --tag onnx .

.PHONY: docker_run
docker_run:
	docker run --name onnx --rm onnx
