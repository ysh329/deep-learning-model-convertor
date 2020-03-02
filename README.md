The project will be updated continuously ......  

Pull requests are welcome!

# [Deep Learning Model Convertors](https://github.com/ysh329/deep-learning-model-convertor)

Note: **This is not one convertor for all frameworks, but a collection of different converters.** Because github is an open source platform, I hope we can help each other here, gather everyone's strength.

Because of these different frameworks, the awesome convertors of deep learning models for different frameworks occur. It should be noted that I did not test all the converters, so I could not guarantee that each was available. But I also hope this convertor collection may help you!

The sheet below is a overview of all convertors in github (not only contain official provided and more are user-self implementations). I just make a little work to collect these convertors. Also, hope everyone can support this project to help more people who're also crazy because of various frameworks.

| convertor | [mxnet](http://data.dmlc.ml/models/) | [caffe](https://github.com/BVLC/caffe/wiki/Model-Zoo) :sparkles:  | [caffe2](https://github.com/caffe2/caffe2/wiki/Model-Zoo) | [CNTK](https://www.microsoft.com/en-us/cognitive-toolkit/features/model-gallery/) | [theano](https://github.com/Theano/Theano/wiki/Related-projects)/[lasagne](https://github.com/Lasagne/Recipes) | [neon](https://github.com/NervanaSystems/ModelZoo) | [pytorch](https://github.com/pytorch/vision):sparkles: | [torch](https://github.com/torch/torch7/wiki/ModelZoo) | [keras](https://github.com/fchollet/deep-learning-models) :sparkles: | [darknet](https://pjreddie.com/darknet/imagenet/) | [tensorflow](https://github.com/tensorflow/models) :sparkles: | [chainer](http://docs.chainer.org/en/stable/reference/caffe.html) | [coreML/iOS](https://developer.apple.com/documentation/coreml) :sparkles:| [paddle](https://github.com/PaddlePaddle/models) | ONNX |
| --------- |:-----:|:-----:|:-----:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|**[mxnet](http://data.dmlc.ml/models/)**  |   -   | [MMdnn](https://github.com/Microsoft/MMdnn) [MXNet2Caffe](https://github.com/cypw/MXNet2Caffe) [Mxnet2Caffe](https://github.com/wranglerwong/Mxnet2Caffe) | [MMdnn (through ONNX)](https://github.com/Microsoft/MMdnn) | [MMdnn](https://github.com/Microsoft/MMdnn) | None | None | [MMdnn](https://github.com/Microsoft/MMdnn) [gluon2pytorch](https://github.com/nerox8664/gluon2pytorch) | None | [MMdnn](https://github.com/Microsoft/MMdnn) | None | [MMdnn](https://github.com/Microsoft/MMdnn) | None | [mxnet-to-coreml](https://github.com/apache/incubator-mxnet/tree/master/tools/coreml) [MMdnn](https://github.com/Microsoft/MMdnn) | None | None |
|**[caffe](https://github.com/BVLC/caffe/wiki/Model-Zoo)** :sparkles: | [mxnet/tools/caffe_converter](https://github.com/dmlc/mxnet/tree/master/tools/caffe_converter) [ResNet_caffe2mxnet](https://github.com/nicklhy/ResNet_caffe2mxnet) [MMdnn](https://github.com/Microsoft/MMdnn) |  - | [CaffeToCaffe2](https://caffe2.ai/docs/caffe-migration.html#caffe-to-caffe2) [MMdnn (through ONNX)](https://github.com/Microsoft/MMdnn) | [crosstalkcaffe/CaffeConverter](https://github.com/Microsoft/CNTK/tree/master/bindings/python/cntk/contrib/crosstalkcaffe) [MMdnn](https://github.com/Microsoft/MMdnn) | [caffe_theano_conversion](https://github.com/an-kumar/caffe-theano-conversion) [caffe-model-convert](https://github.com/kencoken/caffe-model-convert) [caffe-to-theano](https://github.com/piergiaj/caffe-to-theano) |[caffe2neon](https://github.com/NervanaSystems/caffe2neon) | [MMdnn](https://github.com/Microsoft/MMdnn) [pytorch-caffe](https://github.com/marvis/pytorch-caffe) [pytorch-resnet](https://github.com/ruotianluo/pytorch-resnet) | [googlenet-caffe2torch](https://github.com/kmatzen/googlenet-caffe2torch) [mocha](https://github.com/kuangliu/mocha) [loadcaffe](https://github.com/szagoruyko/loadcaffe) | [caffe_weight_converter](https://github.com/pierluigiferrari/caffe_weight_converter) [caffe2keras](https://github.com/qxcv/caffe2keras) [nn_tools](https://github.com/hahnyuan/nn_tools) [keras](https://github.com/MarcBS/keras) [caffe2keras](https://github.com/OdinLin/caffe2keras) [Deep_Learning_Model_Converter](https://github.com/jamescfli/Deep_Learning_Model_Converter) [MMdnn](https://github.com/Microsoft/MMdnn) | [pytorch-caffe-darknet-convert](https://github.com/marvis/pytorch-caffe-darknet-convert) | [MMdnn](https://github.com/Microsoft/MMdnn) [nn_tools](https://github.com/hahnyuan/nn_tools) [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow) | None | [CoreMLZoo](https://github.com/mdering/CoreMLZoo) [apple/coremltools](https://apple.github.io/coremltools/) [MMdnn](https://github.com/Microsoft/MMdnn) | [X2Paddle](https://github.com/PaddlePaddle/X2Paddle) | None |
|**[caffe2](https://github.com/caffe2/caffe2/wiki/Model-Zoo)**| None | None | - | ONNX | None | None | ONNX | None | None | None | None | None | None | None | None |
|**[CNTK](https://www.microsoft.com/en-us/cognitive-toolkit/features/model-gallery/)**| [MMdnn](https://github.com/Microsoft/MMdnn) | [MMdnn](https://github.com/Microsoft/MMdnn) | ONNX [MMdnn (through ONNX)](https://github.com/Microsoft/MMdnn) | - | None | None | ONNX [MMdnn](https://github.com/Microsoft/MMdnn) | None | [MMdnn](https://github.com/Microsoft/MMdnn) | None | [MMdnn](https://github.com/Microsoft/MMdnn) | None | [MMdnn](https://github.com/Microsoft/MMdnn) | None | None |
|**[theano](https://github.com/Theano/Theano/wiki/Related-projects)/[lasagne](https://github.com/Lasagne/Recipes)**| None | None | None | None |   -   | None | None | None | None | None | None | None | None | None | None |
|**[neon](https://github.com/NervanaSystems/ModelZoo)**| None | None | None | None | None |   -   | None | None | None | None | None | None | None | None | None |
|**[pytorch](https://github.com/pytorch/vision)** :sparkles:| [MMdnn](https://github.com/Microsoft/MMdnn) | [PytorchToCaffe](https://github.com/xxradon/PytorchToCaffe) [MMdnn](https://github.com/Microsoft/MMdnn) [pytorch2caffe](https://github.com/longcw/pytorch2caffe) [pytorch-caffe-darknet-convert](https://github.com/marvis/pytorch-caffe-darknet-convert) | [onnx-caffe2](https://github.com/onnx/onnx-caffe2) [MMdnn (through ONNX)](https://github.com/Microsoft/MMdnn) | ONNX [MMdnn](https://github.com/Microsoft/MMdnn) | None | None |   -   | None | [MMdnn](https://github.com/Microsoft/MMdnn) [pytorch2keras](https://github.com/nerox8664/pytorch2keras) [nn-transfer](https://github.com/gzuidhof/nn-transfer) | [pytorch-caffe-darknet-convert](https://github.com/marvis/pytorch-caffe-darknet-convert) | [MMdnn](https://github.com/Microsoft/MMdnn) [pytorch2keras](https://github.com/nerox8664/pytorch2keras) (over Keras) [pytorch-tf](https://github.com/leonidk/pytorch-tf) | None | [MMdnn](https://github.com/Microsoft/MMdnn) [onnx-coreml](https://github.com/onnx/onnx-coreml) | None | None |
|**[torch](https://github.com/torch/torch7/wiki/ModelZoo)** | None | [fb-caffe-exts/torch2caffe](https://github.com/facebook/fb-caffe-exts#torch2caffe) [mocha](https://github.com/kuangliu/mocha) [trans-torch](https://github.com/Teaonly/trans-torch) [th2caffe](https://github.com/e-lab/th2caffe) | [Torch2Caffe2](https://github.com/ca1773130n/Torch2Caffe2) | None | None | None |[convert_torch_to_pytorch](https://github.com/clcarwin/convert_torch_to_pytorch)|   -   | None | None | None | None | [torch2coreml](https://github.com/prisma-ai/torch2coreml) [torch2ios](https://github.com/woffle/torch2ios) | None | None |
|**[keras](https://github.com/fchollet/deep-learning-models)** :sparkles: | [MMdnn](https://github.com/Microsoft/MMdnn) | [MMdnn](https://github.com/Microsoft/MMdnn) [nn_tools](https://github.com/hahnyuan/nn_tools) [keras2caffe](https://github.com/uhfband/keras2caffe) | [MMdnn (through ONNX)](https://github.com/Microsoft/MMdnn) | [MMdnn](https://github.com/Microsoft/MMdnn) | None | None | [MMdnn](https://github.com/Microsoft/MMdnn) [nn-transfer](https://github.com/gzuidhof/nn-transfer) | None |   -   | None | [nn_tools](https://github.com/hahnyuan/nn_tools) [convert-to-tensorflow](https://github.com/goranrauker/convert-to-tensorflow) [keras_to_tensorflow](https://github.com/alanswx/keras_to_tensorflow) [keras_to_tensorflow](https://github.com/amir-abdi/keras_to_tensorflow) [MMdnn](https://github.com/Microsoft/MMdnn) | None | [apple/coremltools](https://apple.github.io/coremltools/) :sparkles: [model-converters](https://github.com/triagemd/model-converters) [keras_models](https://github.com/Bulochkin/keras_models) [MMdnn](https://github.com/Microsoft/MMdnn) | None | None |
|**[darknet](https://pjreddie.com/darknet/imagenet/)**| None | [pytorch-caffe-darknet-convert](https://github.com/marvis/pytorch-caffe-darknet-convert) | None | [MMdnn](https://github.com/Microsoft/MMdnn) | None | None | [pytorch-caffe-darknet-convert](https://github.com/marvis/pytorch-caffe-darknet-convert) | None | [MMdnn](https://github.com/Microsoft/MMdnn) |   -   | [DW2TF](https://github.com/jinyu121/DW2TF) [darkflow](https://github.com/thtrieu/darkflow) [lego_yolo](https://github.com/dEcmir/lego_yolo) | None | None | None | None |
|**[tensorflow](https://github.com/tensorflow/models)** :sparkles: | [MMdnn](https://github.com/Microsoft/MMdnn) | [MMdnn](https://github.com/Microsoft/MMdnn) [nn_tools](https://github.com/hahnyuan/nn_tools)| [MMdnn (through ONNX)](https://github.com/Microsoft/MMdnn) | [crosstalk](https://github.com/Microsoft/CNTK/tree/master/bindings/python/cntk/contrib/crosstalk) [MMdnn](https://github.com/Microsoft/MMdnn) | None | None | [pytorch-tf](https://github.com/leonidk/pytorch-tf) [MMdnn](https://github.com/Microsoft/MMdnn) | None | [model-converters](https://github.com/triagemd/model-converters) [nn_tools](https://github.com/hahnyuan/nn_tools) [convert-to-tensorflow](https://github.com/goranrauker/convert-to-tensorflow) [MMdnn](https://github.com/Microsoft/MMdnn) | None |   -   | None | [tfcoreml](https://github.com/tf-coreml/tf-coreml) [MMdnn](https://github.com/Microsoft/MMdnn) | [X2Paddle](https://github.com/PaddlePaddle/X2Paddle) | None |
|**[chainer](http://docs.chainer.org/en/stable/reference/caffe.html)**| None | None | None | None | None | None |[chainer2pytorch](https://github.com/vzhong/chainer2pytorch)| None | None | None | None | - | None | None | None |
|**[coreML/iOS](https://developer.apple.com/documentation/coreml)** :sparkles:| [MMdnn](https://github.com/Microsoft/MMdnn) | [MMdnn](https://github.com/Microsoft/MMdnn) | [MMdnn (through ONNX)](https://github.com/Microsoft/MMdnn) | [MMdnn](https://github.com/Microsoft/MMdnn) | None | None | [MMdnn](https://github.com/Microsoft/MMdnn) | None | [MMdnn](https://github.com/Microsoft/MMdnn) | None | [MMdnn](https://github.com/Microsoft/MMdnn) | None | - | None |
| [paddle](http://paddlepaddle.org/) | None | None | None | None | None | None | None | None | None | None | None | None | None | - | None |
|**[ONNX](https://github.com/onnx/onnx)**| None | None | None | None | None | None | None | None | None | None | None | None | None | [X2Paddle](https://github.com/PaddlePaddle/X2Paddle) | - |

# Brief Intro of Convertors

## Open Neural Network Exchange

**General framework for converting between all kinds of neural networks**

[ONNX](https://github.com/onnx) is an effort to unify converters for neural networks in order to bring some sanity to the NN world. Released by Facebook and Microsoft.
More info [here](http://onnx.ai).

## MMdnn

[MMdnn](https://github.com/Microsoft/MMdnn) is a set of tools to help users inter-operate among different deep learning frameworks. E.g. model conversion and visualization. Convert models between CaffeEmit, CNTK, CoreML, Keras, MXNet, ONNX, PyTorch and TensorFlow.

<img src="https://github.com/Microsoft/MMdnn/blob/master/docs/supported.jpg" width="400" height="400">

## MXNet convertor

**Convert to MXNet model**.

### [mdering/CoreMLZoo: A few models converted from caffe to CoreMLs format](https://github.com/mdering/CoreMLZoo)
A few deep learning models converted from various formats to CoreMLs format.
Models currently available:

- SqueezeNet
- VGG19
Please feel free to create a pull request with additional models.


### [mxnet/tools/caffe_converter](https://github.com/dmlc/mxnet/tree/master/tools/caffe_converter)

Key topics covered include the following:
* Converting Caffe trained models to MXNet
* Calling Caffe operators in MXNet
More concretly, please refer to [mxnet/tools/caffe_converter](https://github.com/dmlc/mxnet/tree/master/tools/caffe_converter) and [How to | Convert from Caffe to MXNet](https://github.com/dmlc/mxnet/blob/master/docs/how_to/caffe.md).

### [nicklhy/ResNet_caffe2mxnet](https://github.com/nicklhy/ResNet_caffe2mxnet)

This is a tool to convert the deep-residual-networks from caffe model to mxnet model. The weights are directly copied from caffe network blobs.

## Caffe convertor

**Convert to Caffe model**.

### [pytorch2caffe](https://github.com/longcw/pytorch2caffe)

Convert PyTorch model to Caffemodel.

### [cypw/MXNet2Caffe](https://github.com/cypw/MXNet2Caffe)

Convert MXNet model to Caffe model.

### [wranglerwong/Mxnet2Caffe: Convert MXNet model to Caffe model](https://github.com/wranglerwong/Mxnet2Caffe)

Convert MXNet model to Caffe model.

### [kuangliu/mocha](https://github.com/kuangliu/mocha)

Convert torch model to/from caffe model easily.

### [uhfband/keras2caffe: Keras to Caffe model converter tool](https://github.com/uhfband/keras2caffe)


This tool tested with Caffe 1.0, Keras 2.1.2 and TensorFlow 1.4.0

Working conversion examples:
- Inception V3
- Inception V4 (https://github.com/kentsommer/keras-inceptionV4)
- Xception V1
- SqueezeNet (https://github.com/rcmalli/keras-squeezenet)
- VGG16


Problem layers:
- ZeroPadding2D
- MaxPooling2D and AveragePooling2D with asymmetric padding

### [facebook/fb-caffe-exts/torch2caffe](https://github.com/facebook/fb-caffe-exts#torch2caffe)

Some handy utility libraries and tools for the Caffe deep learning framework, which has ** A library for converting pre-trained Torch models to the equivalent Caffe models.**

###  [pytorch-caffe-darknet-convert](https://github.com/marvis/pytorch-caffe-darknet-convert)

Convert between pytorch, caffe and darknet models. Caffe darknet models can be load directly by pytorch.

### [Teaonly/trans-torch](https://github.com/Teaonly/trans-torch)

Translating Torch model to other framework such as Caffe, MxNet ...

### [e-lab/th2caffe](https://github.com/e-lab/th2caffe)

A torch-nn to caffe converter for specific layers.

### [hahnyuan/nn_tools](https://github.com/hahnyuan/nn_tools)

a neural network convertor for models among caffe tensorflow keras

### [xxradon/PytorchToCaffe: Pytorch model to caffe model, supported pytorch 0.3, 0.3.1, 0.4, 0.4.1 ,1.0 , 1.0.1 , 1.2 ,1.3 .notice that only pytorch 1.1 have some bugs](https://github.com/xxradon/PytorchToCaffe)  

Providing a tool for neural network frameworks for pytorch and caffe.

The nn_tools is released under the MIT License (refer to the LICENSE file for details).

features: 

1. Converting a pytorch model to caffe model.
2. Some convenient tools of manipulate caffemodel and prototxt quickly(like get or set weights of layers).  
3. Support pytorch version >= 0.2.(Have tested on 0.3,0.3.1, 0.4, 0.4.1 ,1.0, 1.2)  
4. Analysing a model, get the operations number(ops) in every layers.  
Noting: pytorch version 1.1 is not supported now  

requirements  
1. Python2.7 or Python3.x
2. Each functions in this tools requires corresponding neural network python package (pytorch and so on).

## Caffe2 convertor

**Convert to Caffe2 model**.

### [CaffeToCaffe2](https://caffe2.ai/docs/caffe-migration.html#caffe-to-caffe2)
This is an official convertor, which not only provoide a script also an ipython notebook as below:
- [caffe2/Getting_Caffe1_Models_for_Translation.ipynb at master · caffe2/caffe2](https://github.com/caffe2/caffe2/blob/master/caffe2/python/tutorials/Getting_Caffe1_Models_for_Translation.ipynb)
- [caffe2/caffe_translator.py at master · caffe2/caffe2](https://github.com/caffe2/caffe2/blob/master/caffe2/python/caffe_translator.py)

### [onnx-caffe2](https://github.com/onnx/onnx-caffe2)

Convert PyTorch to Caffe2 (making it especially easy to deploy on mobile devices)

## CNTK convertor

**Convert to CNTK model**.

### [crosstalkcaffe/CaffeConverter](https://github.com/Microsoft/CNTK/tree/master/bindings/python/cntk/contrib/crosstalkcaffe)

The tool will help you convert trained models from Caffe to CNTK.

Convert trained models: giving a model script and its weights file, export to CNTK model.

### [crosstalk](https://github.com/Microsoft/CNTK/tree/master/bindings/python/cntk/contrib/crosstalk)

crosstalk is from CNTK contrib.

## Theano/Lasagne convertor

**Convert to Theano/Lasagne model**.

### [an-kumar/caffe-theano-conversion](https://github.com/an-kumar/caffe-theano-conversion)

This is part of a project for CS231N at Stanford University, written by Ankit Kumar, Mathematics major, Class of 2015

This is a repository that allows you to convert pretrained caffe models into models in Lasagne, a thin wrapper around Theano. You can also convert a caffe model's architecture to an equivalent one in Lasagne. You do not need caffe installed to use this module.

Currently, the following caffe layers are supported:
```
* Convolution
* LRN
* Pooling
* Inner Product
* Relu
* Softmax
```

### [kencoken/caffe-model-convert](https://github.com/kencoken/caffe-model-convert)

Convert models from Caffe to Theano format.

### [piergiaj/caffe-to-theano](https://github.com/piergiaj/caffe-to-theano)

Convert a Caffe Model to a Theano Model. This currently works on AlexNet, but should work for any Caffe model that only includes layers that have been impemented.

## Neon convertor  

**Convert to Neon model**.

### [NervanaSystems/caffe2neon](https://github.com/NervanaSystems/caffe2neon)

Tools to convert Caffe models to neon's serialization format.

This repo contains tools to convert Caffe models into a format compatible with the [neon deep learning library](https://github.com/NervanaSystems/caffe2neon). The main script, "decaffeinate.py", takes as input a caffe model definition file and the corresponding model weights file and returns a neon serialized model file. This output file can be used to instantiate the neon Model object, which will generate a model in neon that should replicate the behavior of the Caffe model.

## PyTorch convertor  

**Convert to PyTorch model**.

### [nerox8664/gluon2pytorch](https://github.com/nerox8664/gluon2pytorch)

Convert mxnet / gluon graph to PyTorch source + weights.

### [ruotianluo/pytorch-resnet](https://github.com/ruotianluo/pytorch-resnet)

Convert resnet trained in caffe to pytorch model.

### [clcarwin/convert_torch_to_pytorch](https://github.com/clcarwin/convert_torch_to_pytorch)

Convert torch t7 model to pytorch model and source.

### [vzhong/chainer2pytorch](https://github.com/vzhong/chainer2pytorch)

`chainer2pytorch` implements conversions from Chainer modules to PyTorch modules, setting parameters of each modules such that one can port over models on a module basis.

###  [pytorch-caffe](https://github.com/marvis/pytorch-caffe)

Load caffe prototxt and weights directly in pytorch without explicitly converting model from caffe to pytorch.

###  [nn-transfer](https://github.com/gzuidhof/nn-transfer)

Convert between Keras and PyTorch models.

## Torch convertor  

**Convert to Torch model**.

### [kmatzen/googlenet-caffe2torch](https://github.com/kmatzen/googlenet-caffe2torch)

Converts bvlc_googlenet.caffemodel to a Torch nn model.

Want to use the pre-trained GoogLeNet from the BVLC Model Zoo in Torch? Do you not want to use Caffe as an additional dependency inside Torch? Use these two scripts to build the network definition in Torch and copy the learned weights from the Caffe model.

### [kuangliu/mocha](https://github.com/kuangliu/mocha)

Convert torch model to/from caffe model easily.

### [szagoruyko/loadcaffe](https://github.com/szagoruyko/loadcaffe)

Convert caffe model to a Torch nn.Sequential model.

## Keras convertor  

**Convert to Keras model**.

### [pierluigiferrari/caffe_weight_converter](https://github.com/pierluigiferrari/caffe_weight_converter)

This is a Caffe-to-Keras weight converter, i.e. it converts `.caffemodel` weight files to Keras-2-compatible HDF5 weight files. It can also export `.caffemodel` weights as Numpy arrays for further processing.

This converter converts the weights of a model only (not the model definition), which has the great advantage that it doesn't break every time it encounters an unknown layer type like other converters to that try to translate the model definition as well. The downside, of course, is that you'll have to write the model definition yourself.

The repository also provides converted weights for some popular models.

### [qxcv/caffe2keras](https://github.com/qxcv/caffe2keras)

Note: This converter has been adapted from code in [Marc Bolaños fork of Caffe](https://github.com/MarcBS/keras). See acks for code provenance.

This is intended to serve as a conversion module for Caffe models to Keras models.

Please, be aware that this module is not regularly maintained. Thus, some layers or parameter definitions introduced in newer versions of either Keras or Caffe might not be compatible with the converter. Pull requests welcome!

### [hahnyuan/nn_tools](https://github.com/hahnyuan/nn_tools)

a neural network convertor for models among caffe tensorflow keras

### [MarcBS/keras](https://github.com/MarcBS/keras)

Keras' fork with several new functionalities. Caffe2Keras converter, multimodal layers, etc. https://github.com/MarcBS/keras

This fork of Keras offers the following contributions:

Caffe to Keras conversion module
Layer-specific learning rates
New layers for multimodal data
Contact email: marc.bolanos@ub.edu

GitHub page: https://github.com/MarcBS

MarcBS/keras is compatible with: Python 2.7 and Theano only.

### [OdinLin/caffe2keras](https://github.com/OdinLin/caffe2keras)

a simple tool to translate caffe model to keras model.

### [jamescfli/Deep_Learning_Model_Converter](https://github.com/jamescfli/Deep_Learning_Model_Converter)

###  [nn-transfer](https://github.com/gzuidhof/nn-transfer)

Convert between Keras and PyTorch models.

###  [pytorch2keras](https://github.com/nerox8664/pytorch2keras)

Convert pytorch models to Keras.

## Darknet convertor  

**Convert to Darknet model**.

### [pytorch-caffe-darknet-convert](https://github.com/marvis/pytorch-caffe-darknet-convert)

Convert between pytorch, caffe and darknet models. Caffe darknet models can be load directly by pytorch.

### []()

### []()

## TensorFlow convertor  

**Convert to TensorFlow model**.

###  [crosstalk](https://github.com/Microsoft/CNTK/tree/master/bindings/python/cntk/contrib/crosstalk)

crosstalk is from CNTK.

### [triagemd/model-converters: Tools for converting Keras models for use with other ML frameworks](https://github.com/triagemd/model-converters)

Tools for converting Keras models for use with other ML frameworks (coreML, TensorFlow).

### [hahnyuan/nn_tools](https://github.com/hahnyuan/nn_tools)

a neural network convertor for models among caffe tensorflow keras

### [ethereon/caffe-tensorflow: Caffe models in TensorFlow](https://github.com/ethereon/caffe-tensorflow)

Convert Caffe models to TensorFlow.

### [goranrauker/convert-to-tensorflow](https://github.com/goranrauker/convert-to-tensorflow)

Converts a variety of trained models to a frozen tensorflow protocol buffer file for use with the c++ tensorflow api. C++ code is included for using the frozen models.

### [thtrieu/darkflow](https://github.com/thtrieu/darkflow)

Translate darknet to tensorflow. Load trained weights, retrain/fine-tune using tensorflow, export constant graph def to mobile devices.

### [dEcmir/lego_yolo](https://github.com/dEcmir/lego_yolo)

Tensorflow code to to retrain yolo on a new dataset using weights from darknet

This repository contains experiments of transfer learning using YOLO on a new synthetical LEGO data set ROUGH AND UNDOCUMENTED!

### [alanswx/keras_to_tensorflow](https://github.com/alanswx/keras_to_tensorflow)

Convert keras models to tensorflow frozen graph for use on cell phones, etc.

### [amir-abdi/keras_to_tensorflow](https://github.com/amir-abdi/keras_to_tensorflow)

General code to convert a trained keras model into an inference tensorflow model.

### [leonidk/pytorch-tf](https://github.com/leonidk/pytorch-tf)

Converting a pretrained pytorch model to tensorflow

###  [pytorch2keras](https://github.com/nerox8664/pytorch2keras)

Convert pytorch models to Tensorflow (via Keras)

### [jinyu121/DW2TF: Darknet Weights to TensorFlow](https://github.com/jinyu121/DW2TF)

This is a simple convector which converts Darknet weights file (.weights) to Tensorflow weights file (.ckpt).

### [leonidk/pytorch-tf](https://github.com/leonidk/pytorch-tf)

No readme.

## Chainer convertor  

**Convert to Chainer model**.

## coreML convertor  

**Convert to coreML model**.

### [Apple: Converting Trained Models to Core ML](https://developer.apple.com/documentation/coreml)

Convert trained models created with third-party machine learning tools to the Core ML model format.

If your model is created and trained using a supported third-party machine learning tool, you can use Core ML Tools to convert it to the Core ML model format. Table 1 lists the supported models and third-party tools.

|Model type|Supported models|Supported tools|
|:--:|:--:|:--:|
|Neural networks|Feedforward, convolutional, recurrent|Caffe v1 <br/>Keras 1.2.2+|
|Tree ensembles|Random forests, boosted trees, decision trees|scikit-learn 0.18 <br/>XGBoost 0.6|
|Support vector machines|Scalar regression, multiclass classification|scikit-learn 0.18<br/>LIBSVM 3.22|
|Generalized linear models|Linear regression, logistic regression|scikit-learn 0.18|
|Feature engineering|Sparse vectorization, dense vectorization, categorical processing|scikit-learn 0.18|
|Pipeline models|Sequentially chained models|scikit-learn 0.18|


### [mxnet/tools/mxnet-to-coreml](https://github.com/apache/incubator-mxnet/tree/master/tools/coreml)
Convert MXNet models into Apple CoreML format.
This tool helps convert MXNet models into Apple CoreML format which can then be run on Apple devices.

### [prisma-ai/torch2coreml: Torch7 -> CoreML](https://github.com/prisma-ai/torch2coreml)
This tool helps convert Torch7 models into Apple CoreML format which can then be run on Apple devices.

### [woffle/torch2ios: Torch7 Library - Convert NN Models To iOS Format](https://github.com/woffle/torch2ios)

Torch7 Library - Convert NN Models To iOS Format.

Small lib to serialise Torch7 Networks for iOS. Supported Layers include Fully Connected, Pooling and Convolution Layers at present. The library stores the weights & biases (if any) for each layer necesarry for inference on iOS devices.

### [Bulochkin/keras_models: Keras models with python-based convertor to provide embedding in IOS platform](https://github.com/Bulochkin/keras_models)

Keras models with python-based convertor to provide embedding in IOS platform.

### [triagemd/model-converters: Tools for converting Keras models for use with other ML frameworks](https://github.com/triagemd/model-converters)

Tools for converting Keras models for use with other ML frameworks (coreML, TensorFlow).

### [Apple & Google: Tensorflow to CoreML converter](https://github.com/tf-coreml/tf-coreml)

Google collaborated with Apple to create a Tensorflow to CoreML converter [announcement](https://developers.googleblog.com/2017/12/announcing-core-ml-support.html).

Support for Core ML is provided through a tool that takes a TensorFlow model and converts it to the Core ML Model Format (.mlmodel).

## Paddle convertor  

**Convert to Paddle model**.

### [PaddlePaddle/X2Paddle: X2Paddle is a toolkit for converting trained model to PaddlePaddle from other deep learning frameworks. 支持主流深度学习框架模型转换至PaddlePaddle『飞桨』](https://github.com/PaddlePaddle/X2Paddle)  

X2Paddle is a toolkit for converting trained model to PaddlePaddle from other deep learning frameworks. 

More detailed models: [X2Paddle/x2paddle_model_zoo.md at develop · PaddlePaddle/X2Paddle](https://github.com/PaddlePaddle/X2Paddle/blob/develop/x2paddle_model_zoo.md)
