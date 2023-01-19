import math
from hls4ml.converters.torchscript_to_hls import torchscript_handler
from hls4ml.converters.utils import *

@torchscript_handler('Conv1d')
def parse_conv1d_layer(torchscript_layer, layer_name, input_shapes, data_reader, config):
    assert('Conv1d' in torchscript_layer.original_name)
    
    layer = {}
    
    layer['name'] = layer_name
    layer['class_name'] = 'Conv1D'
    layer['data_format'] = 'channels_first' #torchscript default (can't change)
    
    #Input info
    (
        layer['in_width'],
        layer['n_chan']
    ) = parse_data_format(input_shapes[0], 'channels_first') #Keras's default is channels_last
    
    #Additional parameters
    layer['n_filt'] = torchscript_layer.out_channels
    layer['filt_width'] = torchscript_layer.kernel_size[0] 
    layer['stride_width'] = torchscript_layer.stride[0]
    layer['pad_left'] = layer['pad_right'] = torchscript_layer.padding[0]
    layer['dilation'] = torchscript_layer.dilation[0]
    
    if torchscript_layer.padding[0] == 0: # No padding, i.e., 'VALID' padding in Keras/Tensorflow
        layer['padding'] = 'valid'
    else: #Only 'valid' and 'same' padding are available in Keras
        layer['padding'] = 'same'
    
    #Ouput info
    (layer['out_width'],_,_) = compute_padding_1d(layer['padding'],
                                                  layer['in_width'],
                                                  layer['stride_width'],
                                                  layer['filt_width'])
    
    output_shape=[input_shapes[0][0], layer['n_filt'], layer['out_width']] #Channel first as default
    
    return layer, output_shape

@torchscript_handler('Conv2d')
def parse_conv2d_layer(torchscript_layer, layer_name, input_shapes, data_reader, config):
    assert('Conv2d' in torchscript_layer.original_name)
    
    layer = {}
    
    layer['name'] = layer_name
    layer['class_name'] = 'Conv2D'
    layer['data_format'] = 'channels_first' #torchscript default (can't change)
    

    print (parse_data_format(input_shapes[0], 'channels_first'))
    #Input info
    (
        layer['in_height'],
        layer['in_width'],
        layer['n_chan']
    ) = parse_data_format(input_shapes[0], 'channels_first') #Keras's default is channels_last
    
    #Additional parameters
    layer['n_filt'] = torchscript_layer.out_channels
    layer['filt_height'] = torchscript_layer.kernel_size[0]
    layer['filt_width'] = torchscript_layer.kernel_size[1]
    layer['stride_height'] = torchscript_layer.stride[0]
    layer['stride_width'] = torchscript_layer.stride[1]
    layer['dilation'] = torchscript_layer.dilation[0]
    layer['pad_top'] = layer['pad_bottom'] = torchscript_layer.padding[0]
    layer['pad_left'] = layer['pad_right'] = torchscript_layer.padding[1]
    
    if all(x == 0 for x in torchscript_layer.padding): # No padding, i.e., 'VALID' padding in Keras/Tensorflow
        layer['padding'] = 'valid'
    else: #Only 'valid' and 'same' padding are available in Keras
        layer['padding'] = 'same'
    
    #Ouput info
    (layer['out_height'], layer['out_width'],_,_,_,_) = compute_padding_2d(layer['padding'],
                                                                           layer['in_height'],
                                                                           layer['in_width'],
                                                                           layer['stride_height'],
                                                                           layer['stride_width'],
                                                                           layer['filt_height'],
                                                                           layer['filt_width'])
    
    output_shape = [input_shapes[0][0], layer['n_filt'], layer['out_height'], layer['out_width']]
    
    return layer, output_shape