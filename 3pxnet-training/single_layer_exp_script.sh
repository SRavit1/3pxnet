for name in 0000 0001 0010 0011 0100 0101 0110 0111 1000 1001 1010 1011 1100 1101 1110 1110 1111;
do
python convert.py --input-format onnx --output-format nnef --input-mode "training_data/single_exp_$name.onnx" --output-mode "../single_exp_$name.nnef";
rm ../3pxnet-compiler/autogen/*;
python ../3pxnet-compiler/compiler.py --input="../single_exp_$name.nnef/" --dataset CIFAR10 --test_end_id=1;
mkdir ~/Documents/NanoCAD/esp-who/components/esp-face/3PXNet_models/simple_exp/3pxnet-model-autogen/autogen_$name;
mv ../3pxnet-compiler/autogen/* ~/Documents/NanoCAD/esp-who/components/esp-face/3PXNet_models/simple_exp/3pxnet-model-autogen/autogen_$name;
mv ~/Documents/NanoCAD/esp-who/components/esp-face/3PXNet_models/simple_exp/3pxnet-model-autogen/autogen_$name/source.c ~/Documents/NanoCAD/esp-who/components/esp-face/3PXNet_models/simple_exp/3pxnet-model-autogen/autogen_$name/source.h
done
