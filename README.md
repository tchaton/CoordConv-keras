# CoordConv for Keras
Keras implementation of CoordConv from the paper [An intriguing failing of convolutional neural networks and the CoordConv solution](https://arxiv.org/abs/1807.03247).

Extends the `CoordinateChannel` concatenation from only 2D rank (images) to 1D (text / time series) and 3D tensors (video / voxels).

# Usage

Import `coord.py` and call it *before* any convolution layer in order to attach the coordinate channels to the input.

There are **3 different versions of CoordinateChannel** - 1D, 2D and 3D for each of `Conv1D`, `Conv2D` and `Conv3D`. 

```python
from coord import CoordConv2d

# prior to first conv
ip = Input(shape=(64, 64, 2))
x = CoordConv2d()(ip)
x = Conv2D(...)(x)  # This defines the `CoordConv` from the paper.
...
x = CoordConv2d(with_r=True)(x)
x = Conv2D(...)(x)  # This adds the 3rd channel for the radius.
```

# Refers

Other Keras implementation : https://github.com/titu1994/keras-coordconv/blob/master/README.md
Pytorch Implementation : https://github.com/mkocabas/CoordConv-pytorch

# Requirements

- Keras 2.2.0+
- Either Tensorflow, Theano or CNTK backend.
- Matplotlib (to plot images only)
