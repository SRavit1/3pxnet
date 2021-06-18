ifndef PYTHON_EXECUTABLE
    PYTHON_EXECUTABLE = python
endif

SIZE=Small
SIZE_OPTION=0
PERMUTE_OPTION=1

all: train convert
prereq:
	pip3 install git+https://github.com/KhronosGroup/NNEF-Tools.git
	pip3 install future typing six numpy protobuf onnx torch
	pip3 install pandas
	pip3 install bokeh
train:
ifdef MNIST
	$(PYTHON_EXECUTABLE) main_mnist.py --size $(SIZE_OPTION) --permute $(PERMUTE_OPTION) --cpu
else ifdef CIFAR
	$(PYTHON_EXECUTABLE) main_conv.py --size $(SIZE_OPTION) --permute $(PERMUTE_OPTION) --cpu
endif
convert:
ifdef MNIST
	$(PYTHON_EXECUTABLE) convert.py --input-format onnx --output-format nnef --input-mode "training_data/FC_$(SIZE).onnx" --output-mode "../FC_$(SIZE).nnef"
	$(PYTHON_EXECUTABLE) ../3pxnet-compiler/compiler.py --input="../FC_$(SIZE).nnef"
else ifdef CIFAR
	$(PYTHON_EXECUTABLE) convert.py --input-format onnx --output-format nnef --input-mode "training_data/CNN_$(SIZE).onnx" --output-mode "../CNN_$(SIZE).nnef"
	$(PYTHON_EXECUTABLE) ../3pxnet-compiler/compiler.py --input="../CNN_$(SIZE).nnef" --dataset=CIFAR10
endif
clean:
	rm -rf *.nnef *.onnx ../3pxnet-compiler/autogen/*