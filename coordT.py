import keras
import tensorflow as tf
import numpy as np
from keras.layers import *
class CoordConv2DTranspose (Conv2D):
    @interfaces.legacy_deconv2d_support
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 output_padding=None,
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 with_r=False,
                 **kwargs):
        super (CoordConv2DTranspose, self).__init__ (
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

        self.output_padding = output_padding
        self.with_r = with_r
        if self.output_padding is not None:
            self.output_padding = conv_utils.normalize_tuple (
                self.output_padding, 2, 'output_padding')
            for stride, out_pad in zip (self.strides, self.output_padding):
                if out_pad >= stride:
                    raise ValueError ('Stride ' + str (self.strides) + ' must be '
                                                                       'greater than output padding ' +
                                      str (self.output_padding))

    def build(self, input_shape):
        if len (input_shape) != 4:
            raise ValueError ('Inputs should have rank ' +
                              str (4) +
                              '; Received input shape:', str (input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError ('The channel dimension of the inputs '
                              'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (self.filters, input_dim + 2)

        print(kernel_shape)

        self.kernel = self.add_weight (shape=kernel_shape,
                                       initializer=self.kernel_initializer,
                                       name='kernel',
                                       regularizer=self.kernel_regularizer,
                                       constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight (shape=(self.filters,),
                                         initializer=self.bias_initializer,
                                         name='bias',
                                         regularizer=self.bias_regularizer,
                                         constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec (ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def addCoord(self, inputs):

        shape = tf.shape (inputs)
        batch_size_tensor, x_dim, y_dim, c = shape[0], shape[1], shape[2], shape[3]

        xx_ones = tf.ones ([batch_size_tensor, x_dim], dtype=tf.float32)

        xx_ones = tf.expand_dims (xx_ones, axis=-1)

        xx_range = tf.tile (tf.expand_dims (tf.range (x_dim), 0), [batch_size_tensor, 1])
        xx_range = tf.cast (xx_range, tf.float32)
        xx_range = tf.expand_dims (xx_range, axis=1)

        xx_channel = tf.matmul (xx_ones, xx_range)
        xx_channel = tf.expand_dims (xx_channel, axis=-1)

        xx_channel = xx_channel / tf.cast (x_dim - 1, tf.float32)

        xx_channel = xx_channel * 2 - 1

        ret = tf.concat ([inputs, xx_channel, tf.transpose (xx_channel, (0, 2, 1, 3))], axis=-1)

        if self.with_r:
            rr = tf.sqrt (tf.square (xx_channel - .5) + tf.square (tf.transpose (xx_channel) - .5))
            ret = tf.concat ([ret, rr], axis=-1)
        return ret

    def call(self, inputs):

        inputs = self.addCoord (inputs)
        input_shape = K.shape (inputs)
        batch_size = input_shape[0]
        if self.data_format == 'channels_first':
            h_axis, w_axis = 2, 3
        else:
            h_axis, w_axis = 1, 2

        height, width = input_shape[h_axis], input_shape[w_axis]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides
        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        # Infer the dynamic output shape:
        out_height = conv_utils.deconv_length (height,
                                               stride_h,
                                               kernel_h,
                                               self.padding)
        out_width = conv_utils.deconv_length (width,
                                              stride_w,
                                              kernel_w,
                                              self.padding)
        if self.data_format == 'channels_first':
            output_shape = (batch_size, self.filters, out_height, out_width)
        else:
            output_shape = (batch_size, out_height, out_width, self.filters)

        outputs = K.conv2d_transpose (
            inputs,
            self.kernel,
            output_shape,
            self.strides,
            padding=self.padding,
            data_format=self.data_format)

        if self.use_bias:
            outputs = K.bias_add (
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation (outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = list (input_shape)
        if self.data_format == 'channels_first':
            c_axis, h_axis, w_axis = 1, 2, 3
        else:
            c_axis, h_axis, w_axis = 3, 1, 2

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides
        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        output_shape[c_axis] = self.filters
        output_shape[h_axis] = conv_utils.deconv_length (output_shape[h_axis],
                                                         stride_h,
                                                         kernel_h,
                                                         self.padding)
        output_shape[w_axis] = conv_utils.deconv_length (output_shape[w_axis],
                                                         stride_w,
                                                         kernel_w,
                                                         self.padding)
        return tuple (output_shape)

    def get_config(self):
        config = super (CoordConv2DTranspose, self).get_config ()
        config.pop ('dilation_rate')
        config['output_padding'] = self.output_padding
        return config

if __name__ == '__main__':
	conv2dT = CoordConv2DTranspose(32, (3, 3))
	input_shape = [None, 64, 64, 3]
	data_shape = (1, 64, 64, 3)
	conv2dT.build(input_shape=input_shape)
	init_gb = tf.global_variables_initializer()
	init_lo = tf.local_variables_initializer()
	with tf.Session() as sess:
    		sess.run([init_gb, init_lo])
    		inp =  tf.placeholder(tf.float32, input_shape)
   	 	out = conv2dT(inp)
    
    		data = np.random.normal(0, 1, data_shape)
    		feed_dict = {inp:data}
    		out = sess.run(out, feed_dict=feed_dict)
    		print(data.shape, out.shape)

