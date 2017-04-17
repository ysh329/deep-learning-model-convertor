# Deep Learning Model Convertors
Because of these different frameworks, the awesome convertors of deep learning models for different frameworks occur. It should be noted that I did not test all the converters, so I could not guarantee that each was available. But I also hope this convertor may help you!

The sheet below is a overview of all convertors in github (not only contain official provided and more is user-self accomplished). I just make a little work to collect these convertors. Also, hope everyone can support this project to help more people who're also crazy because of various framework.

| convertor | mxnet | caffe | theano/lasagne | neon | pytorch | torch | keras | darknet | tensorflow |
| --------- |:-----:| :-----:|:-----:|:----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|**mxnet**  |   -   | [MXNet2Caffe](https://github.com/cypw/MXNet2Caffe) | None | None | None | None | None | None | None |
|**caffe**  | [mxnet/tools/caffe_converter](https://github.com/dmlc/mxnet/tree/master/tools/caffe_converter) [ResNet_caffe2mxnet](https://github.com/nicklhy/ResNet_caffe2mxnet)|  -  |[caffe_theano_conversion](https://github.com/an-kumar/caffe-theano-conversion) [caffe-model-convert](https://github.com/kencoken/caffe-model-convert) [caffe-to-theano](https://github.com/piergiaj/caffe-to-theano) |[caffe2neon](https://github.com/NervanaSystems/caffe2neon)| None |[googlenet-caffe2torch](https://github.com/kmatzen/googlenet-caffe2torch) [mocha](https://github.com/kuangliu/mocha)|[caffe2keras](https://github.com/qxcv/caffe2keras) [nn_tools](https://github.com/hahnyuan/nn_tools) [caffe2keras](https://github.com/masterhou/caffe2keras) [keras](https://github.com/MarcBS/keras) [caffe2keras](https://github.com/OdinLin/caffe2keras) [Deep_Learning_Model_Converter](https://github.com/jamescfli/Deep_Learning_Model_Converter)| None |[nn_tools](https://github.com/hahnyuan/nn_tools)|
|**theano/lasagne**| None | None |   -   | None | None | None | None | None | None |
|**neon**| None | None | None |   -   | None | None | None | None | None |
|**pytorch**| None | None | None | None |   -   | None | None | None | None |
|**torch**| None |[mocha](https://github.com/kuangliu/mocha) [trans-torch](https://github.com/Teaonly/trans-torch) [th2caffe](https://github.com/e-lab/th2caffe)| None | None |[convert_torch_to_pytorch](https://github.com/clcarwin/convert_torch_to_pytorch)|   -   | None | None | None |
|**keras**| None |[nn_tools](https://github.com/hahnyuan/nn_tools)| None | None | None | None |   -   | None |[nn_tools](https://github.com/hahnyuan/nn_tools) [convert-to-tensorflow](https://github.com/goranrauker/convert-to-tensorflow) [keras_to_tensorflow](https://github.com/alanswx/keras_to_tensorflow) |
|**darknet**| None | None | None | None | None | None | None |   -   | [darkflow](https://github.com/thtrieu/darkflow) [lego_yolo](https://github.com/dEcmir/lego_yolo) |
|**tensorflow**| None |[nn_tools](https://github.com/hahnyuan/nn_tools)| None | None | None | None |[nn_tools](https://github.com/hahnyuan/nn_tools) [convert-to-tensorflow](https://github.com/goranrauker/convert-to-tensorflow)| None |   -   |

# Brief Intro. of Convertors

## MXNet convertor  

Convert various model **to MXNet model**.  

### [mxnet/tools/caffe_converter](https://github.com/dmlc/mxnet/tree/master/tools/caffe_converter)

Key topics covered include the following:  
* Converting Caffe trained models to MXNet  
* Calling Caffe operators in MXNet
More concretly, please refer to [mxnet/tools/caffe_converter](https://github.com/dmlc/mxnet/tree/master/tools/caffe_converter) and [How to | Convert from Caffe to MXNet](https://github.com/dmlc/mxnet/blob/master/docs/how_to/caffe.md).

### [nicklhy/ResNet_caffe2mxnet](https://github.com/nicklhy/ResNet_caffe2mxnet)

This is a tool to convert the deep-residual-networks from caffe model to mxnet model. The weights are directly copied from caffe network blobs.

## Caffe convertor  

Convert various model **to Caffe model**.  

### [cypw/MXNet2Caffe](https://github.com/cypw/MXNet2Caffe)

Convert MXNet model to Caffe model.

### [kuangliu/mocha](https://github.com/kuangliu/mocha)

Convert torch model to/from caffe model easily.

### [Teaonly/trans-torch](https://github.com/Teaonly/trans-torch)

Translating Torch model to other framework such as Caffe, MxNet ...

### [e-lab/th2caffe](https://github.com/e-lab/th2caffe)

A torch-nn to caffe converter for specific layers.

### [hahnyuan/nn_tools](https://github.com/hahnyuan/nn_tools)

a neural network convertor for models among caffe tensorfloe keras

## Theano/Lasagne convertor  

Convert various model **to Theano/Lasagne model**.  

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

Convert various model **to Neon model**.  

## PyTorch convertor  

Convert various model **to PyTorch model**.  

## Torch convertor  

Convert various model **to Torch model**.  

## Keras convertor  

Convert various model **to Keras model**.  

## Darknet convertor  

Convert various model **to Darknet model**.  

## TensorFlow convertor  

Convert various model **to TensorFlow model**.  
