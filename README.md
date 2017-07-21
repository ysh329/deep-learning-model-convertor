The project will be updated continuously ......  

Pull requests are welcome!

# [Deep Learning Model Convertors](https://github.com/ysh329/deep-learning-model-convertor)

Note: **This is not one convertor for all frameworks, but a collection of different converters.** Because github is an open source platform, I hope we can help each other here, gather everyone's strength.

Because of these different frameworks, the awesome convertors of deep learning models for different frameworks occur. It should be noted that I did not test all the converters, so I could not guarantee that each was available. But I also hope this convertor collection may help you!

The sheet below is a overview of all convertors in github (not only contain official provided and more are user-self implementations). I just make a little work to collect these convertors. Also, hope everyone can support this project to help more people who're also crazy because of various frameworks.

| convertor | mxnet | caffe :sparkles: | theano/lasagne | neon | pytorch | torch :sparkles: | keras :sparkles: | darknet | tensorflow :sparkles: | chainer |
| --------- |:-----:| :-----:|:-----:|:----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|**mxnet**  |   -   | [MXNet2Caffe](https://github.com/cypw/MXNet2Caffe) | None | None | None | None | None | None | None | None |
|**caffe** :sparkles: | [mxnet/tools/caffe_converter](https://github.com/dmlc/mxnet/tree/master/tools/caffe_converter) [ResNet_caffe2mxnet](https://github.com/nicklhy/ResNet_caffe2mxnet)|  -  |[caffe_theano_conversion](https://github.com/an-kumar/caffe-theano-conversion) [caffe-model-convert](https://github.com/kencoken/caffe-model-convert) [caffe-to-theano](https://github.com/piergiaj/caffe-to-theano) |[caffe2neon](https://github.com/NervanaSystems/caffe2neon)| [pytorch-caffe-darknet-convert](https://github.com/marvis/pytorch-caffe-darknet-convert) |[googlenet-caffe2torch](https://github.com/kmatzen/googlenet-caffe2torch) [mocha](https://github.com/kuangliu/mocha)|[caffe2keras](https://github.com/qxcv/caffe2keras) [nn_tools](https://github.com/hahnyuan/nn_tools) [caffe2keras](https://github.com/masterhou/caffe2keras) [keras](https://github.com/MarcBS/keras) [caffe2keras](https://github.com/OdinLin/caffe2keras) [Deep_Learning_Model_Converter](https://github.com/jamescfli/Deep_Learning_Model_Converter)| [pytorch-caffe-darknet-convert](https://github.com/marvis/pytorch-caffe-darknet-convert) |[nn_tools](https://github.com/hahnyuan/nn_tools)| None |
|**theano/lasagne**| None | None |   -   | None | None | None | None | None | None | None |
|**neon**| None | None | None |   -   | None | None | None | None | None | None |
|**pytorch**| None | [pytorch-caffe-darknet-convert](https://github.com/marvis/pytorch-caffe-darknet-convert) | None | None |   -   | None | None | [pytorch-caffe-darknet-convert](https://github.com/marvis/pytorch-caffe-darknet-convert) | None | None |
|**torch** :sparkles: | None |[mocha](https://github.com/kuangliu/mocha) [trans-torch](https://github.com/Teaonly/trans-torch) [th2caffe](https://github.com/e-lab/th2caffe)| None | None |[convert_torch_to_pytorch](https://github.com/clcarwin/convert_torch_to_pytorch)|   -   | None | None | None | None |
|**keras** :sparkles: | None |[nn_tools](https://github.com/hahnyuan/nn_tools)| None | None | None | None |   -   | None |[nn_tools](https://github.com/hahnyuan/nn_tools) [convert-to-tensorflow](https://github.com/goranrauker/convert-to-tensorflow) [keras_to_tensorflow](https://github.com/alanswx/keras_to_tensorflow) | None |
|**darknet**| None | [pytorch-caffe-darknet-convert](https://github.com/marvis/pytorch-caffe-darknet-convert) | None | None | [pytorch-caffe-darknet-convert](https://github.com/marvis/pytorch-caffe-darknet-convert) | None | None |   -   | [darkflow](https://github.com/thtrieu/darkflow) [lego_yolo](https://github.com/dEcmir/lego_yolo) | None |
|**tensorflow** :sparkles: | None |[nn_tools](https://github.com/hahnyuan/nn_tools)| None | None | None | None |[nn_tools](https://github.com/hahnyuan/nn_tools) [convert-to-tensorflow](https://github.com/goranrauker/convert-to-tensorflow)| None |   -   | None |
|**chainer**| None | None | None | None |[chainer2pytorch](https://github.com/vzhong/chainer2pytorch)| None | None | None | None | - |

# Brief Intro. of Convertors

## MXNet convertor  

**Convert to MXNet model**.  

### [mxnet/tools/caffe_converter](https://github.com/dmlc/mxnet/tree/master/tools/caffe_converter)

Key topics covered include the following:  
* Converting Caffe trained models to MXNet  
* Calling Caffe operators in MXNet
More concretly, please refer to [mxnet/tools/caffe_converter](https://github.com/dmlc/mxnet/tree/master/tools/caffe_converter) and [How to | Convert from Caffe to MXNet](https://github.com/dmlc/mxnet/blob/master/docs/how_to/caffe.md).

### [nicklhy/ResNet_caffe2mxnet](https://github.com/nicklhy/ResNet_caffe2mxnet)

This is a tool to convert the deep-residual-networks from caffe model to mxnet model. The weights are directly copied from caffe network blobs.

## Caffe convertor  

**Convert to Caffe model**.  

### [cypw/MXNet2Caffe](https://github.com/cypw/MXNet2Caffe)

Convert MXNet model to Caffe model.

### [kuangliu/mocha](https://github.com/kuangliu/mocha)

Convert torch model to/from caffe model easily.

###  [pytorch-caffe-darknet-convert](https://github.com/marvis/pytorch-caffe-darknet-convert) 

Convert between pytorch, caffe and darknet models. Caffe darknet models can be load directly by pytorch.

### [Teaonly/trans-torch](https://github.com/Teaonly/trans-torch)

Translating Torch model to other framework such as Caffe, MxNet ...

### [e-lab/th2caffe](https://github.com/e-lab/th2caffe)

A torch-nn to caffe converter for specific layers.

### [hahnyuan/nn_tools](https://github.com/hahnyuan/nn_tools)

a neural network convertor for models among caffe tensorfloe keras

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

### [clcarwin/convert_torch_to_pytorch](https://github.com/clcarwin/convert_torch_to_pytorch)

Convert torch t7 model to pytorch model and source.

### [vzhong/chainer2pytorch](https://github.com/vzhong/chainer2pytorch)  

`chainer2pytorch` implements conversions from Chainer modules to PyTorch modules, setting parameters of each modules such that one can port over models on a module basis.

###  [pytorch-caffe-darknet-convert](https://github.com/marvis/pytorch-caffe-darknet-convert) 

Convert between pytorch, caffe and darknet models. Caffe darknet models can be load directly by pytorch.

## Torch convertor  

**Convert to Torch model**.  

### [kmatzen/googlenet-caffe2torch](https://github.com/kmatzen/googlenet-caffe2torch)

Converts bvlc_googlenet.caffemodel to a Torch nn model.

Want to use the pre-trained GoogLeNet from the BVLC Model Zoo in Torch? Do you not want to use Caffe as an additional dependency inside Torch? Use these two scripts to build the network definition in Torch and copy the learned weights from the Caffe model.

### [kuangliu/mocha](https://github.com/kuangliu/mocha)

Convert torch model to/from caffe model easily.

## Keras convertor  

**Convert to Keras model**.  

### [qxcv/caffe2keras](https://github.com/qxcv/caffe2keras)

Note: This converter has been adapted from code in [Marc Bolaños fork of Caffe](https://github.com/MarcBS/keras). See acks for code provenance.

This is intended to serve as a conversion module for Caffe models to Keras models.

Please, be aware that this module is not regularly maintained. Thus, some layers or parameter definitions introduced in newer versions of either Keras or Caffe might not be compatible with the converter. Pull requests welcome!

### [hahnyuan/nn_tools](https://github.com/hahnyuan/nn_tools)

a neural network convertor for models among caffe tensorfloe keras

### [masterhou/caffe2keras](https://github.com/masterhou/caffe2keras)

Caffe to Keras converter, From https://github.com/MarcBS/keras.
This is intended to serve as a conversion module for Caffe models to Keras Functional API models.

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

## Darknet convertor  

**Convert to Darknet model**.  

### [pytorch-caffe-darknet-convert](https://github.com/marvis/pytorch-caffe-darknet-convert) 

Convert between pytorch, caffe and darknet models. Caffe darknet models can be load directly by pytorch.

### []()

### []()

## TensorFlow convertor  

**Convert to TensorFlow model**.  

### [hahnyuan/nn_tools](https://github.com/hahnyuan/nn_tools)

a neural network convertor for models among caffe tensorfloe keras

### [goranrauker/convert-to-tensorflow](https://github.com/goranrauker/convert-to-tensorflow)

Converts a variety of trained models to a frozen tensorflow protocol buffer file for use with the c++ tensorflow api. C++ code is included for using the frozen models.

### [thtrieu/darkflow](https://github.com/thtrieu/darkflow)

Translate darknet to tensorflow. Load trained weights, retrain/fine-tune using tensorflow, export constant graph def to mobile devices.

### [dEcmir/lego_yolo](https://github.com/dEcmir/lego_yolo)

Tensorflow code to to retrain yolo on a new dataset using weights from darknet

This repository contains experiments of transfer learning using YOLO on a new synthetical LEGO data set ROUGH AND UNDOCUMENTED!

### [alanswx/keras_to_tensorflow](https://github.com/alanswx/keras_to_tensorflow)

Convert keras models to tensorflow frozen graph for use on cell phones, etc.

## Chainer convertor  

**Convert to Chainer model**. 
