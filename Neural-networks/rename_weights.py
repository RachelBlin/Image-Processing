#import keras

"""model = keras.models.load_model("/home/rblin/Documents/weights/test_rename/temp.h5")
model.summary()


for i, layer in enumerate(model.layers):
    layer.name = 'layer_' + str(i)

model.summary()"""



import h5py
import numpy as np

f = h5py.File("/home/rblin/Documents/weights/test_rename/temp_rgb.h5", "a")

f.copy("/model_weights/P3/P3", "/model_weights/P32/P32")
del f["model_weights"]['P3']

f["model_weights"]["P32"].attrs["weight_names"] = b'P32/kernel:0', b'P32/bias:0'

f.copy("/model_weights/P4/P4", "/model_weights/P42/P42")
del f["model_weights"]['P4']

f["model_weights"]["P42"].attrs["weight_names"] = b'P42/kernel:0', b'P42/bias:0'

f.copy("/model_weights/P5/P5", "/model_weights/P52/P52")
del f["model_weights"]['P5']

f["model_weights"]["P52"].attrs["weight_names"] = b'P52/kernel:0', b'P52/bias:0'

f.copy("/model_weights/P6/P6", "/model_weights/P62/P62")
del f["model_weights"]['P6']

f["model_weights"]["P62"].attrs["weight_names"] = b'P62/kernel:0', b'P62/bias:0'

f.copy("/model_weights/P7/P7", "/model_weights/P72/P72")
del f["model_weights"]['P7']

f["model_weights"]["P72"].attrs["weight_names"] = b'P72/kernel:0', b'P72/bias:0'

f.copy("/model_weights/input_1", "/model_weights/input_12")
del f["model_weights"]['input_1']

f.copy("/model_weights/padding_conv1", "/model_weights/padding_conv12")
del f["model_weights"]['padding_conv1']

f.copy("/model_weights/conv1/conv1", "/model_weights/conv12/conv12")
del f["model_weights"]['conv1']

f["model_weights"]["conv12"].attrs["weight_names"] = [b'conv12/kernel:0']

f.copy("/model_weights/conv1_relu", "/model_weights/conv1_relu2")
del f["model_weights"]['conv1_relu']

f.copy("/model_weights/pool1", "/model_weights/pool12")
del f["model_weights"]['pool1']

f.copy("/model_weights/res2a_branch2a/res2a_branch2a", "/model_weights/res2a_branch2a2/res2a_branch2a2")
del f["model_weights"]['res2a_branch2a']

f["model_weights"]["res2a_branch2a2"].attrs["weight_names"] = [b'res2a_branch2a2/kernel:0']

f.copy("/model_weights/bn2a_branch2a/bn2a_branch2a", "/model_weights/bn2a_branch2a2/bn2a_branch2a2")
del f["model_weights"]['bn2a_branch2a']

f["model_weights"]["bn2a_branch2a2"].attrs["weight_names"] = b'bn2a_branch2a2/gamma:0', b'bn2a_branch2a2/beta:0', b'bn2a_branch2a2/moving_mean:0', b'bn2a_branch2a2/moving_variance:0'

f.copy("/model_weights/res2a_branch2a_relu", "/model_weights/res2a_branch2a_relu2")
del f["model_weights"]['res2a_branch2a_relu']

f.copy("/model_weights/padding2a_branch2b", "/model_weights/padding2a_branch2b2")
del f["model_weights"]['padding2a_branch2b']

f.copy("/model_weights/res2a_branch2b/res2a_branch2b", "/model_weights/res2a_branch2b2/res2a_branch2b2")
del f["model_weights"]['res2a_branch2b']

f["model_weights"]["res2a_branch2b2"].attrs["weight_names"] = [b'res2a_branch2b2/kernel:0']

f.copy("/model_weights/bn2a_branch2b/bn2a_branch2b", "/model_weights/bn2a_branch2b2/bn2a_branch2b2")
del f["model_weights"]['bn2a_branch2b']

f["model_weights"]["bn2a_branch2b2"].attrs["weight_names"] = b'bn2a_branch2b2/gamma:0', b'bn2a_branch2b2/beta:0', b'bn2a_branch2b2/moving_mean:0', b'bn2a_branch2b2/moving_variance:0'

f.copy("/model_weights/res2a_branch2b_relu", "/model_weights/res2a_branch2b_relu2")
del f["model_weights"]['res2a_branch2b_relu']

f.copy("/model_weights/res2a_branch2c/res2a_branch2c", "/model_weights/res2a_branch2c2/res2a_branch2c2")
del f["model_weights"]['res2a_branch2c']

f["model_weights"]["res2a_branch2c2"].attrs["weight_names"] = [b'res2a_branch2c2/kernel:0']

f.copy("/model_weights/res2a_branch1/res2a_branch1", "/model_weights/res2a_branch12/res2a_branch12")
del f["model_weights"]['res2a_branch1']

f["model_weights"]["res2a_branch12"].attrs["weight_names"] = [b'res2a_branch12/kernel:0']

f.copy("/model_weights/bn2a_branch2c/bn2a_branch2c", "/model_weights/bn2a_branch2c2/bn2a_branch2c2")
del f["model_weights"]['bn2a_branch2c']

f["model_weights"]["bn2a_branch2c2"].attrs["weight_names"] = b'bn2a_branch2c2/gamma:0', b'bn2a_branch2c2/beta:0', b'bn2a_branch2c2/moving_mean:0', b'bn2a_branch2c2/moving_variance:0'

f.copy("/model_weights/bn2a_branch1/bn2a_branch1", "/model_weights/bn2a_branch12/bn2a_branch12")
del f["model_weights"]['bn2a_branch1']

f["model_weights"]["bn2a_branch12"].attrs["weight_names"] = b'bn2a_branch12/gamma:0', b'bn2a_branch12/beta:0', b'bn2a_branch12/moving_mean:0', b'bn2a_branch12/moving_variance:0'

f.copy("/model_weights/res2a", "/model_weights/res2a2")
del f["model_weights"]['res2a']

f.copy("/model_weights/res2a_relu", "/model_weights/res2a_relu2")
del f["model_weights"]['res2a_relu']

f.copy("/model_weights/res2b_branch2a/res2b_branch2a", "/model_weights/res2b_branch2a2/res2b_branch2a2")
del f["model_weights"]['res2b_branch2a']

f["model_weights"]["res2b_branch2a2"].attrs["weight_names"] = [b'res2b_branch2a2/kernel:0']

f.copy("/model_weights/bn2b_branch2a/bn2b_branch2a", "/model_weights/bn2b_branch2a2/bn2b_branch2a2")
del f["model_weights"]['bn2b_branch2a']

f["model_weights"]["bn2b_branch2a2"].attrs["weight_names"] = b'bn2b_branch2a2/gamma:0', b'bn2b_branch2a2/beta:0', b'bn2b_branch2a2/moving_mean:0', b'bn2b_branch2a2/moving_variance:0'

f.copy("/model_weights/res2b_branch2a_relu", "/model_weights/res2b_branch2a_relu2")
del f["model_weights"]['res2b_branch2a_relu']

f.copy("/model_weights/padding2b_branch2b", "/model_weights/padding2b_branch2b2")
del f["model_weights"]['padding2b_branch2b']

f.copy("/model_weights/res2b_branch2b/res2b_branch2b", "/model_weights/res2b_branch2b2/res2b_branch2b2")
del f["model_weights"]['res2b_branch2b']

f["model_weights"]["res2b_branch2b2"].attrs["weight_names"] = [b'res2b_branch2b2/kernel:0']

f.copy("/model_weights/bn2b_branch2b/bn2b_branch2b", "/model_weights/bn2b_branch2b2/bn2b_branch2b2")
del f["model_weights"]['bn2b_branch2b']

f["model_weights"]["bn2b_branch2b2"].attrs["weight_names"] = b'bn2b_branch2b2/gamma:0', b'bn2b_branch2b2/beta:0', b'bn2b_branch2b2/moving_mean:0', b'bn2b_branch2b2/moving_variance:0'

f.copy("/model_weights/res2b_branch2b_relu", "/model_weights/res2b_branch2b_relu2")
del f["model_weights"]['res2b_branch2b_relu']

f.copy("/model_weights/res2b_branch2c/res2b_branch2c", "/model_weights/res2b_branch2c2/res2b_branch2c2")
del f["model_weights"]['res2b_branch2c']

f["model_weights"]["res2b_branch2c2"].attrs["weight_names"] = [b'res2b_branch2c2/kernel:0']

f.copy("/model_weights/bn2b_branch2c/bn2b_branch2c", "/model_weights/bn2b_branch2c2/bn2b_branch2c2")
del f["model_weights"]['bn2b_branch2c']

f["model_weights"]["bn2b_branch2c2"].attrs["weight_names"] = b'bn2b_branch2c2/gamma:0', b'bn2b_branch2c2/beta:0', b'bn2b_branch2c2/moving_mean:0', b'bn2b_branch2c2/moving_variance:0'

f.copy("/model_weights/res2b", "/model_weights/res2b2")
del f["model_weights"]['res2b']

f.copy("/model_weights/res2b_relu", "/model_weights/res2b_relu2")
del f["model_weights"]['res2b_relu']

f.copy("/model_weights/res2c_branch2a/res2c_branch2a", "/model_weights/res2c_branch2a2/res2c_branch2a2")
del f["model_weights"]['res2c_branch2a']

f["model_weights"]["res2c_branch2a2"].attrs["weight_names"] = [b'res2c_branch2a2/kernel:0']

f.copy("/model_weights/bn2c_branch2a/bn2c_branch2a", "/model_weights/bn2c_branch2a2/bn2c_branch2a2")
del f["model_weights"]['bn2c_branch2a']

f["model_weights"]["bn2c_branch2a2"].attrs["weight_names"] = b'bn2c_branch2a2/gamma:0', b'bn2c_branch2a2/beta:0', b'bn2c_branch2a2/moving_mean:0', b'bn2c_branch2a2/moving_variance:0'

f.copy("/model_weights/res2c_branch2a_relu", "/model_weights/res2c_branch2a_relu2")
del f["model_weights"]['res2c_branch2a_relu']

f.copy("/model_weights/padding2c_branch2b", "/model_weights/padding2c_branch2b2")
del f["model_weights"]['padding2c_branch2b']

f.copy("/model_weights/res2c_branch2b/res2c_branch2b", "/model_weights/res2c_branch2b2/res2c_branch2b2")
del f["model_weights"]['res2c_branch2b']

f["model_weights"]["res2c_branch2b2"].attrs["weight_names"] = [b'res2c_branch2b2/kernel:0']

f.copy("/model_weights/bn2c_branch2b/bn2c_branch2b", "/model_weights/bn2c_branch2b2/bn2c_branch2b2")
del f["model_weights"]['bn2c_branch2b']

f["model_weights"]["bn2c_branch2b2"].attrs["weight_names"] = b'bn2c_branch2b2/gamma:0', b'bn2c_branch2b2/beta:0', b'bn2c_branch2b2/moving_mean:0', b'bn2c_branch2b2/moving_variance:0'

f.copy("/model_weights/res2c_branch2b_relu", "/model_weights/res2c_branch2b_relu2")
del f["model_weights"]['res2c_branch2b_relu']

f.copy("/model_weights/res2c_branch2c/res2c_branch2c", "/model_weights/res2c_branch2c2/res2c_branch2c2")
del f["model_weights"]['res2c_branch2c']

f["model_weights"]["res2c_branch2c2"].attrs["weight_names"] = [b'res2c_branch2c2/kernel:0']

f.copy("/model_weights/bn2c_branch2c/bn2c_branch2c", "/model_weights/bn2c_branch2c2/bn2c_branch2c2")
del f["model_weights"]['bn2c_branch2c']

f["model_weights"]["bn2c_branch2c2"].attrs["weight_names"] = b'bn2c_branch2c2/gamma:0', b'bn2c_branch2c2/beta:0', b'bn2c_branch2c2/moving_mean:0', b'bn2c_branch2c2/moving_variance:0'

f.copy("/model_weights/res2c", "/model_weights/res2c2")
del f["model_weights"]['res2c']

f.copy("/model_weights/res2c_relu", "/model_weights/res2c_relu2")
del f["model_weights"]['res2c_relu']

f.copy("/model_weights/res3a_branch2a/res3a_branch2a", "/model_weights/res3a_branch2a2/res3a_branch2a2")
del f["model_weights"]['res3a_branch2a']

f["model_weights"]["res3a_branch2a2"].attrs["weight_names"] = [b'res3a_branch2a2/kernel:0']

f.copy("/model_weights/bn3a_branch2a/bn3a_branch2a", "/model_weights/bn3a_branch2a2/bn3a_branch2a2")
del f["model_weights"]['bn3a_branch2a']

f["model_weights"]["bn3a_branch2a2"].attrs["weight_names"] = b'bn3a_branch2a2/gamma:0', b'bn3a_branch2a2/beta:0', b'bn3a_branch2a2/moving_mean:0', b'bn3a_branch2a2/moving_variance:0'

f.copy("/model_weights/res3a_branch2a_relu", "/model_weights/res3a_branch2a_relu2")
del f["model_weights"]['res3a_branch2a_relu']

f.copy("/model_weights/padding3a_branch2b", "/model_weights/padding3a_branch2b2")
del f["model_weights"]['padding3a_branch2b']

f.copy("/model_weights/res3a_branch2b/res3a_branch2b", "/model_weights/res3a_branch2b2/res3a_branch2b2")
del f["model_weights"]['res3a_branch2b']

f["model_weights"]["res3a_branch2b2"].attrs["weight_names"] = [b'res3a_branch2b2/kernel:0']

f.copy("/model_weights/bn3a_branch2b/bn3a_branch2b", "/model_weights/bn3a_branch2b2/bn3a_branch2b2")
del f["model_weights"]['bn3a_branch2b']

f["model_weights"]["bn3a_branch2b2"].attrs["weight_names"] = b'bn3a_branch2b2/gamma:0', b'bn3a_branch2b2/beta:0', b'bn3a_branch2b2/moving_mean:0', b'bn3a_branch2b2/moving_variance:0'

f.copy("/model_weights/res3a_branch2b_relu", "/model_weights/res3a_branch2b_relu2")
del f["model_weights"]['res3a_branch2b_relu']

f.copy("/model_weights/res3a_branch2c/res3a_branch2c", "/model_weights/res3a_branch2c2/res3a_branch2c2")
del f["model_weights"]['res3a_branch2c']

f["model_weights"]["res3a_branch2c2"].attrs["weight_names"] = [b'res3a_branch2c2/kernel:0']

f.copy("/model_weights/res3a_branch1/res3a_branch1", "/model_weights/res3a_branch12/res3a_branch12")
del f["model_weights"]['res3a_branch1']

f["model_weights"]["res3a_branch12"].attrs["weight_names"] = [b'res3a_branch12/kernel:0']

f.copy("/model_weights/bn3a_branch2c/bn3a_branch2c", "/model_weights/bn3a_branch2c2/bn3a_branch2c2")
del f["model_weights"]['bn3a_branch2c']

f["model_weights"]["bn3a_branch2c2"].attrs["weight_names"] = b'bn3a_branch2c2/gamma:0', b'bn3a_branch2c2/beta:0', b'bn3a_branch2c2/moving_mean:0', b'bn3a_branch2c2/moving_variance:0'

f.copy("/model_weights/bn3a_branch1/bn3a_branch1", "/model_weights/bn3a_branch12/bn3a_branch12")
del f["model_weights"]['bn3a_branch1']

f["model_weights"]["bn3a_branch12"].attrs["weight_names"] = b'bn3a_branch12/gamma:0', b'bn3a_branch12/beta:0', b'bn3a_branch12/moving_mean:0', b'bn3a_branch12/moving_variance:0'

f.copy("/model_weights/res3a", "/model_weights/res3a2")
del f["model_weights"]['res3a']

f.copy("/model_weights/res3a_relu", "/model_weights/res3a_relu2")
del f["model_weights"]['res3a_relu']

f.copy("/model_weights/res3b_branch2a/res3b_branch2a", "/model_weights/res3b_branch2a2/res3b_branch2a2")
del f["model_weights"]['res3b_branch2a']

f["model_weights"]["res3b_branch2a2"].attrs["weight_names"] = [b'res3b_branch2a2/kernel:0']

f.copy("/model_weights/bn3b_branch2a/bn3b_branch2a", "/model_weights/bn3b_branch2a2/bn3b_branch2a2")
del f["model_weights"]['bn3b_branch2a']

f["model_weights"]["bn3b_branch2a2"].attrs["weight_names"] = b'bn3b_branch2a2/gamma:0', b'bn3b_branch2a2/beta:0', b'bn3b_branch2a2/moving_mean:0', b'bn3b_branch2a2/moving_variance:0'

f.copy("/model_weights/res3b_branch2a_relu", "/model_weights/res3b_branch2a_relu2")
del f["model_weights"]['res3b_branch2a_relu']

f.copy("/model_weights/padding3b_branch2b", "/model_weights/padding3b_branch2b2")
del f["model_weights"]['padding3b_branch2b']

f.copy("/model_weights/res3b_branch2b/res3b_branch2b", "/model_weights/res3b_branch2b2/res3b_branch2b2")
del f["model_weights"]['res3b_branch2b']

f["model_weights"]["res3b_branch2b2"].attrs["weight_names"] = [b'res3b_branch2b2/kernel:0']

f.copy("/model_weights/bn3b_branch2b/bn3b_branch2b", "/model_weights/bn3b_branch2b2/bn3b_branch2b2")
del f["model_weights"]['bn3b_branch2b']

f["model_weights"]["bn3b_branch2b2"].attrs["weight_names"] = b'bn3b_branch2b2/gamma:0', b'bn3b_branch2b2/beta:0', b'bn3b_branch2b2/moving_mean:0', b'bn3b_branch2b2/moving_variance:0'

f.copy("/model_weights/res3b_branch2b_relu", "/model_weights/res3b_branch2b_relu2")
del f["model_weights"]['res3b_branch2b_relu']

f.copy("/model_weights/res3b_branch2c/res3b_branch2c", "/model_weights/res3b_branch2c2/res3b_branch2c2")
del f["model_weights"]['res3b_branch2c']

f["model_weights"]["res3b_branch2c2"].attrs["weight_names"] = [b'res3b_branch2c2/kernel:0']

f.copy("/model_weights/bn3b_branch2c/bn3b_branch2c", "/model_weights/bn3b_branch2c2/bn3b_branch2c2")
del f["model_weights"]['bn3b_branch2c']

f["model_weights"]["bn3b_branch2c2"].attrs["weight_names"] = b'bn3b_branch2c2/gamma:0', b'bn3b_branch2c2/beta:0', b'bn3b_branch2c2/moving_mean:0', b'bn3b_branch2c2/moving_variance:0'

f.copy("/model_weights/res3b", "/model_weights/res3b2")
del f["model_weights"]['res3b']

f.copy("/model_weights/res3b_relu", "/model_weights/res3b_relu2")
del f["model_weights"]['res3b_relu']

f.copy("/model_weights/res3c_branch2a/res3c_branch2a", "/model_weights/res3c_branch2a2/res3c_branch2a2")
del f["model_weights"]['res3c_branch2a']

f["model_weights"]["res3c_branch2a2"].attrs["weight_names"] = [b'res3c_branch2a2/kernel:0']

f.copy("/model_weights/bn3c_branch2a/bn3c_branch2a", "/model_weights/bn3c_branch2a2/bn3c_branch2a2")
del f["model_weights"]['bn3c_branch2a']

f["model_weights"]["bn3c_branch2a2"].attrs["weight_names"] = b'bn3c_branch2a2/gamma:0', b'bn3c_branch2a2/beta:0', b'bn3c_branch2a2/moving_mean:0', b'bn3c_branch2a2/moving_variance:0'

f.copy("/model_weights/res3c_branch2a_relu", "/model_weights/res3c_branch2a_relu2")
del f["model_weights"]['res3c_branch2a_relu']

f.copy("/model_weights/padding3c_branch2b", "/model_weights/padding3c_branch2b2")
del f["model_weights"]['padding3c_branch2b']

f.copy("/model_weights/res3c_branch2b/res3c_branch2b", "/model_weights/res3c_branch2b2/res3c_branch2b2")
del f["model_weights"]['res3c_branch2b']

f["model_weights"]["res3c_branch2b2"].attrs["weight_names"] = [b'res3c_branch2b2/kernel:0']

f.copy("/model_weights/bn3c_branch2b/bn3c_branch2b", "/model_weights/bn3c_branch2b2/bn3c_branch2b2")
del f["model_weights"]['bn3c_branch2b']

f["model_weights"]["bn3c_branch2b2"].attrs["weight_names"] = b'bn3c_branch2b2/gamma:0', b'bn3c_branch2b2/beta:0', b'bn3c_branch2b2/moving_mean:0', b'bn3c_branch2b2/moving_variance:0'

f.copy("/model_weights/res3c_branch2b_relu", "/model_weights/res3c_branch2b_relu2")
del f["model_weights"]['res3c_branch2b_relu']

f.copy("/model_weights/res3c_branch2c/res3c_branch2c", "/model_weights/res3c_branch2c2/res3c_branch2c2")
del f["model_weights"]['res3c_branch2c']

f["model_weights"]["res3c_branch2c2"].attrs["weight_names"] = [b'res3c_branch2c2/kernel:0']

f.copy("/model_weights/bn3c_branch2c/bn3c_branch2c", "/model_weights/bn3c_branch2c2/bn3c_branch2c2")
del f["model_weights"]['bn3c_branch2c']

f["model_weights"]["bn3c_branch2c2"].attrs["weight_names"] = b'bn3c_branch2c2/gamma:0', b'bn3c_branch2c2/beta:0', b'bn3c_branch2c2/moving_mean:0', b'bn3c_branch2c2/moving_variance:0'

f.copy("/model_weights/res3c", "/model_weights/res3c2")
del f["model_weights"]['res3c']

f.copy("/model_weights/res3c_relu", "/model_weights/res3c_relu2")
del f["model_weights"]['res3c_relu']

f.copy("/model_weights/res3d_branch2a/res3d_branch2a", "/model_weights/res3d_branch2a2/res3d_branch2a2")
del f["model_weights"]['res3d_branch2a']

f["model_weights"]["res3d_branch2a2"].attrs["weight_names"] = [b'res3d_branch2a2/kernel:0']

f.copy("/model_weights/bn3d_branch2a/bn3d_branch2a", "/model_weights/bn3d_branch2a2/bn3d_branch2a2")
del f["model_weights"]['bn3d_branch2a']

f["model_weights"]["bn3d_branch2a2"].attrs["weight_names"] = b'bn3d_branch2a2/gamma:0', b'bn3d_branch2a2/beta:0', b'bn3d_branch2a2/moving_mean:0', b'bn3d_branch2a2/moving_variance:0'

f.copy("/model_weights/res3d_branch2a_relu", "/model_weights/res3d_branch2a_relu2")
del f["model_weights"]['res3d_branch2a_relu']

f.copy("/model_weights/padding3d_branch2b", "/model_weights/padding3d_branch2b2")
del f["model_weights"]['padding3d_branch2b']

f.copy("/model_weights/res3d_branch2b/res3d_branch2b", "/model_weights/res3d_branch2b2/res3d_branch2b2")
del f["model_weights"]['res3d_branch2b']

f["model_weights"]["res3d_branch2b2"].attrs["weight_names"] = [b'res3d_branch2b2/kernel:0']

f.copy("/model_weights/bn3d_branch2b/bn3d_branch2b", "/model_weights/bn3d_branch2b2/bn3d_branch2b2")
del f["model_weights"]['bn3d_branch2b']

f["model_weights"]["bn3d_branch2b2"].attrs["weight_names"] = b'bn3d_branch2b2/gamma:0', b'bn3d_branch2b2/beta:0', b'bn3d_branch2b2/moving_mean:0', b'bn3d_branch2b2/moving_variance:0'

f.copy("/model_weights/res3d_branch2b_relu", "/model_weights/res3d_branch2b_relu2")
del f["model_weights"]['res3d_branch2b_relu']

f.copy("/model_weights/res3d_branch2c/res3d_branch2c", "/model_weights/res3d_branch2c2/res3d_branch2c2")
del f["model_weights"]['res3d_branch2c']

f["model_weights"]["res3d_branch2c2"].attrs["weight_names"] = [b'res3d_branch2c2/kernel:0']

f.copy("/model_weights/bn3d_branch2c/bn3d_branch2c", "/model_weights/bn3d_branch2c2/bn3d_branch2c2")
del f["model_weights"]['bn3d_branch2c']

f["model_weights"]["bn3d_branch2c2"].attrs["weight_names"] = b'bn3d_branch2c2/gamma:0', b'bn3d_branch2c2/beta:0', b'bn3d_branch2c2/moving_mean:0', b'bn3d_branch2c2/moving_variance:0'

f.copy("/model_weights/res3d", "/model_weights/res3d2")
del f["model_weights"]['res3d']

f.copy("/model_weights/res3d_relu", "/model_weights/res3d_relu2")
del f["model_weights"]['res3d_relu']

f.copy("/model_weights/res4a_branch2a/res4a_branch2a", "/model_weights/res4a_branch2a2/res4a_branch2a2")
del f["model_weights"]['res4a_branch2a']

f["model_weights"]["res4a_branch2a2"].attrs["weight_names"] = [b'res4a_branch2a2/kernel:0']

f.copy("/model_weights/bn4a_branch2a/bn4a_branch2a", "/model_weights/bn4a_branch2a2/bn4a_branch2a2")
del f["model_weights"]['bn4a_branch2a']

f["model_weights"]["bn4a_branch2a2"].attrs["weight_names"] = b'bn4a_branch2a2/gamma:0', b'bn4a_branch2a2/beta:0', b'bn4a_branch2a2/moving_mean:0', b'bn4a_branch2a2/moving_variance:0'

f.copy("/model_weights/res4a_branch2a_relu", "/model_weights/res4a_branch2a_relu2")
del f["model_weights"]['res4a_branch2a_relu']

f.copy("/model_weights/padding4a_branch2b", "/model_weights/padding4a_branch2b2")
del f["model_weights"]['padding4a_branch2b']

f.copy("/model_weights/res4a_branch2b/res4a_branch2b", "/model_weights/res4a_branch2b2/res4a_branch2b2")
del f["model_weights"]['res4a_branch2b']

f["model_weights"]["res4a_branch2b2"].attrs["weight_names"] = [b'res4a_branch2b2/kernel:0']

f.copy("/model_weights/bn4a_branch2b/bn4a_branch2b", "/model_weights/bn4a_branch2b2/bn4a_branch2b2")
del f["model_weights"]['bn4a_branch2b']

f["model_weights"]["bn4a_branch2b2"].attrs["weight_names"] = b'bn4a_branch2b2/gamma:0', b'bn4a_branch2b2/beta:0', b'bn4a_branch2b2/moving_mean:0', b'bn4a_branch2b2/moving_variance:0'

f.copy("/model_weights/res4a_branch2b_relu", "/model_weights/res4a_branch2b_relu2")
del f["model_weights"]['res4a_branch2b_relu']

f.copy("/model_weights/res4a_branch2c/res4a_branch2c", "/model_weights/res4a_branch2c2/res4a_branch2c2")
del f["model_weights"]['res4a_branch2c']

f["model_weights"]["res4a_branch2c2"].attrs["weight_names"] = [b'res4a_branch2c2/kernel:0']

f.copy("/model_weights/res4a_branch1/res4a_branch1", "/model_weights/res4a_branch12/res4a_branch12")
del f["model_weights"]['res4a_branch1']

f["model_weights"]["res4a_branch12"].attrs["weight_names"] = [b'res4a_branch12/kernel:0']

f.copy("/model_weights/bn4a_branch2c/bn4a_branch2c", "/model_weights/bn4a_branch2c2/bn4a_branch2c2")
del f["model_weights"]['bn4a_branch2c']

f["model_weights"]["bn4a_branch2c2"].attrs["weight_names"] = b'bn4a_branch2c2/gamma:0', b'bn4a_branch2c2/beta:0', b'bn4a_branch2c2/moving_mean:0', b'bn4a_branch2c2/moving_variance:0'

f.copy("/model_weights/bn4a_branch1/bn4a_branch1", "/model_weights/bn4a_branch12/bn4a_branch12")
del f["model_weights"]['bn4a_branch1']

f["model_weights"]["bn4a_branch12"].attrs["weight_names"] = b'bn4a_branch12/gamma:0', b'bn4a_branch12/beta:0', b'bn4a_branch12/moving_mean:0', b'bn4a_branch12/moving_variance:0'

f.copy("/model_weights/res4a", "/model_weights/res4a2")
del f["model_weights"]['res4a']

f.copy("/model_weights/res4a_relu", "/model_weights/res4a_relu2")
del f["model_weights"]['res4a_relu']

f.copy("/model_weights/res4b_branch2a/res4b_branch2a", "/model_weights/res4b_branch2a2/res4b_branch2a2")
del f["model_weights"]['res4b_branch2a']

f["model_weights"]["res4b_branch2a2"].attrs["weight_names"] = [b'res4b_branch2a2/kernel:0']

f.copy("/model_weights/bn4b_branch2a/bn4b_branch2a", "/model_weights/bn4b_branch2a2/bn4b_branch2a2")
del f["model_weights"]['bn4b_branch2a']

f["model_weights"]["bn4b_branch2a2"].attrs["weight_names"] = b'bn4b_branch2a2/gamma:0', b'bn4b_branch2a2/beta:0', b'bn4b_branch2a2/moving_mean:0', b'bn4b_branch2a2/moving_variance:0'

f.copy("/model_weights/res4b_branch2a_relu", "/model_weights/res4b_branch2a_relu2")
del f["model_weights"]['res4b_branch2a_relu']

f.copy("/model_weights/padding4b_branch2b", "/model_weights/padding4b_branch2b2")
del f["model_weights"]['padding4b_branch2b']

f.copy("/model_weights/res4b_branch2b/res4b_branch2b", "/model_weights/res4b_branch2b2/res4b_branch2b2")
del f["model_weights"]['res4b_branch2b']

f["model_weights"]["res4b_branch2b2"].attrs["weight_names"] = [b'res4b_branch2b2/kernel:0']

f.copy("/model_weights/bn4b_branch2b/bn4b_branch2b", "/model_weights/bn4b_branch2b2/bn4b_branch2b2")
del f["model_weights"]['bn4b_branch2b']

f["model_weights"]["bn4b_branch2b2"].attrs["weight_names"] = b'bn4b_branch2b2/gamma:0', b'bn4b_branch2b2/beta:0', b'bn4b_branch2b2/moving_mean:0', b'bn4b_branch2b2/moving_variance:0'

f.copy("/model_weights/res4b_branch2b_relu", "/model_weights/res4b_branch2b_relu2")
del f["model_weights"]['res4b_branch2b_relu']

f.copy("/model_weights/res4b_branch2c/res4b_branch2c", "/model_weights/res4b_branch2c2/res4b_branch2c2")
del f["model_weights"]['res4b_branch2c']

f["model_weights"]["res4b_branch2c2"].attrs["weight_names"] = [b'res4b_branch2c2/kernel:0']

f.copy("/model_weights/bn4b_branch2c/bn4b_branch2c", "/model_weights/bn4b_branch2c2/bn4b_branch2c2")
del f["model_weights"]['bn4b_branch2c']

f["model_weights"]["bn4b_branch2c2"].attrs["weight_names"] = b'bn4b_branch2c2/gamma:0', b'bn4b_branch2c2/beta:0', b'bn4b_branch2c2/moving_mean:0', b'bn4b_branch2c2/moving_variance:0'

f.copy("/model_weights/res4b", "/model_weights/res4b2")
del f["model_weights"]['res4b']

f.copy("/model_weights/res4b_relu", "/model_weights/res4b_relu2")
del f["model_weights"]['res4b_relu']

f.copy("/model_weights/res4c_branch2a/res4c_branch2a", "/model_weights/res4c_branch2a2/res4c_branch2a2")
del f["model_weights"]['res4c_branch2a']

f["model_weights"]["res4c_branch2a2"].attrs["weight_names"] = [b'res4c_branch2a2/kernel:0']

f.copy("/model_weights/bn4c_branch2a/bn4c_branch2a", "/model_weights/bn4c_branch2a2/bn4c_branch2a2")
del f["model_weights"]['bn4c_branch2a']

f["model_weights"]["bn4c_branch2a2"].attrs["weight_names"] = b'bn4c_branch2a2/gamma:0', b'bn4c_branch2a2/beta:0', b'bn4c_branch2a2/moving_mean:0', b'bn4c_branch2a2/moving_variance:0'

f.copy("/model_weights/res4c_branch2a_relu", "/model_weights/res4c_branch2a_relu2")
del f["model_weights"]['res4c_branch2a_relu']

f.copy("/model_weights/padding4c_branch2b", "/model_weights/padding4c_branch2b2")
del f["model_weights"]['padding4c_branch2b']

f.copy("/model_weights/res4c_branch2b/res4c_branch2b", "/model_weights/res4c_branch2b2/res4c_branch2b2")
del f["model_weights"]['res4c_branch2b']

f["model_weights"]["res4c_branch2b2"].attrs["weight_names"] = [b'res4c_branch2b2/kernel:0']

f.copy("/model_weights/bn4c_branch2b/bn4c_branch2b", "/model_weights/bn4c_branch2b2/bn4c_branch2b2")
del f["model_weights"]['bn4c_branch2b']

f["model_weights"]["bn4c_branch2b2"].attrs["weight_names"] = b'bn4c_branch2b2/gamma:0', b'bn4c_branch2b2/beta:0', b'bn4c_branch2b2/moving_mean:0', b'bn4c_branch2b2/moving_variance:0'

f.copy("/model_weights/res4c_branch2b_relu", "/model_weights/res4c_branch2b_relu2")
del f["model_weights"]['res4c_branch2b_relu']

f.copy("/model_weights/res4c_branch2c/res4c_branch2c", "/model_weights/res4c_branch2c2/res4c_branch2c2")
del f["model_weights"]['res4c_branch2c']

f["model_weights"]["res4c_branch2c2"].attrs["weight_names"] = [b'res4c_branch2c2/kernel:0']

f.copy("/model_weights/bn4c_branch2c/bn4c_branch2c", "/model_weights/bn4c_branch2c2/bn4c_branch2c2")
del f["model_weights"]['bn4c_branch2c']

f["model_weights"]["bn4c_branch2c2"].attrs["weight_names"] = b'bn4c_branch2c2/gamma:0', b'bn4c_branch2c2/beta:0', b'bn4c_branch2c2/moving_mean:0', b'bn4c_branch2c2/moving_variance:0'

f.copy("/model_weights/res4c", "/model_weights/res4c2")
del f["model_weights"]['res4c']

f.copy("/model_weights/res4c_relu", "/model_weights/res4c_relu2")
del f["model_weights"]['res4c_relu']

f.copy("/model_weights/res4d_branch2a/res4d_branch2a", "/model_weights/res4d_branch2a2/res4d_branch2a2")
del f["model_weights"]['res4d_branch2a']

f["model_weights"]["res4d_branch2a2"].attrs["weight_names"] = [b'res4d_branch2a2/kernel:0']

f.copy("/model_weights/bn4d_branch2a/bn4d_branch2a", "/model_weights/bn4d_branch2a2/bn4d_branch2a2")
del f["model_weights"]['bn4d_branch2a']

f["model_weights"]["bn4d_branch2a2"].attrs["weight_names"] = b'bn4d_branch2a2/gamma:0', b'bn4d_branch2a2/beta:0', b'bn4d_branch2a2/moving_mean:0', b'bn4d_branch2a2/moving_variance:0'

f.copy("/model_weights/res4d_branch2a_relu", "/model_weights/res4d_branch2a_relu2")
del f["model_weights"]['res4d_branch2a_relu']

f.copy("/model_weights/padding4d_branch2b", "/model_weights/padding4d_branch2b2")
del f["model_weights"]['padding4d_branch2b']

f.copy("/model_weights/res4d_branch2b/res4d_branch2b", "/model_weights/res4d_branch2b2/res4d_branch2b2")
del f["model_weights"]['res4d_branch2b']

f["model_weights"]["res4d_branch2b2"].attrs["weight_names"] = [b'res4d_branch2b2/kernel:0']

f.copy("/model_weights/bn4d_branch2b/bn4d_branch2b", "/model_weights/bn4d_branch2b2/bn4d_branch2b2")
del f["model_weights"]['bn4d_branch2b']

f["model_weights"]["bn4d_branch2b2"].attrs["weight_names"] = b'bn4d_branch2b2/gamma:0', b'bn4d_branch2b2/beta:0', b'bn4d_branch2b2/moving_mean:0', b'bn4d_branch2b2/moving_variance:0'

f.copy("/model_weights/res4d_branch2b_relu", "/model_weights/res4d_branch2b_relu2")
del f["model_weights"]['res4d_branch2b_relu']

f.copy("/model_weights/res4d_branch2c/res4d_branch2c", "/model_weights/res4d_branch2c2/res4d_branch2c2")
del f["model_weights"]['res4d_branch2c']

f["model_weights"]["res4d_branch2c2"].attrs["weight_names"] = [b'res4d_branch2c2/kernel:0']

f.copy("/model_weights/bn4d_branch2c/bn4d_branch2c", "/model_weights/bn4d_branch2c2/bn4d_branch2c2")
del f["model_weights"]['bn4d_branch2c']

f["model_weights"]["bn4d_branch2c2"].attrs["weight_names"] = b'bn4d_branch2c2/gamma:0', b'bn4d_branch2c2/beta:0', b'bn4d_branch2c2/moving_mean:0', b'bn4d_branch2c2/moving_variance:0'

f.copy("/model_weights/res4d", "/model_weights/res4d2")
del f["model_weights"]['res4d']

f.copy("/model_weights/res4d_relu", "/model_weights/res4d_relu2")
del f["model_weights"]['res4d_relu']

f.copy("/model_weights/res4e_branch2a/res4e_branch2a", "/model_weights/res4e_branch2a2/res4e_branch2a2")
del f["model_weights"]['res4e_branch2a']

f["model_weights"]["res4e_branch2a2"].attrs["weight_names"] = [b'res4e_branch2a2/kernel:0']

f.copy("/model_weights/bn4e_branch2a/bn4e_branch2a", "/model_weights/bn4e_branch2a2/bn4e_branch2a2")
del f["model_weights"]['bn4e_branch2a']

f["model_weights"]["bn4e_branch2a2"].attrs["weight_names"] = b'bn4e_branch2a2/gamma:0', b'bn4e_branch2a2/beta:0', b'bn4e_branch2a2/moving_mean:0', b'bn4e_branch2a2/moving_variance:0'

f.copy("/model_weights/res4e_branch2a_relu", "/model_weights/res4e_branch2a_relu2")
del f["model_weights"]['res4e_branch2a_relu']

f.copy("/model_weights/padding4e_branch2b", "/model_weights/padding4e_branch2b2")
del f["model_weights"]['padding4e_branch2b']

f.copy("/model_weights/res4e_branch2b/res4e_branch2b", "/model_weights/res4e_branch2b2/res4e_branch2b2")
del f["model_weights"]['res4e_branch2b']

f["model_weights"]["res4e_branch2b2"].attrs["weight_names"] = [b'res4e_branch2b2/kernel:0']

f.copy("/model_weights/bn4e_branch2b/bn4e_branch2b", "/model_weights/bn4e_branch2b2/bn4e_branch2b2")
del f["model_weights"]['bn4e_branch2b']

f["model_weights"]["bn4e_branch2b2"].attrs["weight_names"] = b'bn4e_branch2b2/gamma:0', b'bn4e_branch2b2/beta:0', b'bn4e_branch2b2/moving_mean:0', b'bn4e_branch2b2/moving_variance:0'

f.copy("/model_weights/res4e_branch2b_relu", "/model_weights/res4e_branch2b_relu2")
del f["model_weights"]['res4e_branch2b_relu']

f.copy("/model_weights/res4e_branch2c/res4e_branch2c", "/model_weights/res4e_branch2c2/res4e_branch2c2")
del f["model_weights"]['res4e_branch2c']

f["model_weights"]["res4e_branch2c2"].attrs["weight_names"] = [b'res4e_branch2c2/kernel:0']

f.copy("/model_weights/bn4e_branch2c/bn4e_branch2c", "/model_weights/bn4e_branch2c2/bn4e_branch2c2")
del f["model_weights"]['bn4e_branch2c']

f["model_weights"]["bn4e_branch2c2"].attrs["weight_names"] = b'bn4e_branch2c2/gamma:0', b'bn4e_branch2c2/beta:0', b'bn4e_branch2c2/moving_mean:0', b'bn4e_branch2c2/moving_variance:0'

f.copy("/model_weights/res4e", "/model_weights/res4e2")
del f["model_weights"]['res4e']

f.copy("/model_weights/res4e_relu", "/model_weights/res4e_relu2")
del f["model_weights"]['res4e_relu']

f.copy("/model_weights/res4f_branch2a/res4f_branch2a", "/model_weights/res4f_branch2a2/res4f_branch2a2")
del f["model_weights"]['res4f_branch2a']

f["model_weights"]["res4f_branch2a2"].attrs["weight_names"] = [b'res4f_branch2a2/kernel:0']

f.copy("/model_weights/bn4f_branch2a/bn4f_branch2a", "/model_weights/bn4f_branch2a2/bn4f_branch2a2")
del f["model_weights"]['bn4f_branch2a']

f["model_weights"]["bn4f_branch2a2"].attrs["weight_names"] = b'bn4f_branch2a2/gamma:0', b'bn4f_branch2a2/beta:0', b'bn4f_branch2a2/moving_mean:0', b'bn4f_branch2a2/moving_variance:0'

f.copy("/model_weights/res4f_branch2a_relu", "/model_weights/res4f_branch2a_relu2")
del f["model_weights"]['res4f_branch2a_relu']

f.copy("/model_weights/padding4f_branch2b", "/model_weights/padding4f_branch2b2")
del f["model_weights"]['padding4f_branch2b']

f.copy("/model_weights/res4f_branch2b/res4f_branch2b", "/model_weights/res4f_branch2b2/res4f_branch2b2")
del f["model_weights"]['res4f_branch2b']

f["model_weights"]["res4f_branch2b2"].attrs["weight_names"] = [b'res4f_branch2b2/kernel:0']

f.copy("/model_weights/bn4f_branch2b/bn4f_branch2b", "/model_weights/bn4f_branch2b2/bn4f_branch2b2")
del f["model_weights"]['bn4f_branch2b']

f["model_weights"]["bn4f_branch2b2"].attrs["weight_names"] = b'bn4f_branch2b2/gamma:0', b'bn4f_branch2b2/beta:0', b'bn4f_branch2b2/moving_mean:0', b'bn4f_branch2b2/moving_variance:0'

f.copy("/model_weights/res4f_branch2b_relu", "/model_weights/res4f_branch2b_relu2")
del f["model_weights"]['res4f_branch2b_relu']

f.copy("/model_weights/res4f_branch2c/res4f_branch2c", "/model_weights/res4f_branch2c2/res4f_branch2c2")
del f["model_weights"]['res4f_branch2c']

f["model_weights"]["res4f_branch2c2"].attrs["weight_names"] = [b'res4f_branch2c2/kernel:0']

f.copy("/model_weights/bn4f_branch2c/bn4f_branch2c", "/model_weights/bn4f_branch2c2/bn4f_branch2c2")
del f["model_weights"]['bn4f_branch2c']

f["model_weights"]["bn4f_branch2c2"].attrs["weight_names"] = b'bn4f_branch2c2/gamma:0', b'bn4f_branch2c2/beta:0', b'bn4f_branch2c2/moving_mean:0', b'bn4f_branch2c2/moving_variance:0'

f.copy("/model_weights/res4f", "/model_weights/res4f2")
del f["model_weights"]['res4f']

f.copy("/model_weights/res4f_relu", "/model_weights/res4f_relu2")
del f["model_weights"]['res4f_relu']

f.copy("/model_weights/res5a_branch2a/res5a_branch2a", "/model_weights/res5a_branch2a2/res5a_branch2a2")
del f["model_weights"]['res5a_branch2a']

f["model_weights"]["res5a_branch2a2"].attrs["weight_names"] = [b'res5a_branch2a2/kernel:0']

f.copy("/model_weights/bn5a_branch2a/bn5a_branch2a", "/model_weights/bn5a_branch2a2/bn5a_branch2a2")
del f["model_weights"]['bn5a_branch2a']

f["model_weights"]["bn5a_branch2a2"].attrs["weight_names"] = b'bn5a_branch2a2/gamma:0', b'bn5a_branch2a2/beta:0', b'bn5a_branch2a2/moving_mean:0', b'bn5a_branch2a2/moving_variance:0'

f.copy("/model_weights/res5a_branch2a_relu", "/model_weights/res5a_branch2a_relu2")
del f["model_weights"]['res5a_branch2a_relu']

f.copy("/model_weights/padding5a_branch2b", "/model_weights/padding5a_branch2b2")
del f["model_weights"]['padding5a_branch2b']

f.copy("/model_weights/res5a_branch2b/res5a_branch2b", "/model_weights/res5a_branch2b2/res5a_branch2b2")
del f["model_weights"]['res5a_branch2b']

f["model_weights"]["res5a_branch2b2"].attrs["weight_names"] = [b'res5a_branch2b2/kernel:0']

f.copy("/model_weights/bn5a_branch2b/bn5a_branch2b", "/model_weights/bn5a_branch2b2/bn5a_branch2b2")
del f["model_weights"]['bn5a_branch2b']

f["model_weights"]["bn5a_branch2b2"].attrs["weight_names"] = b'bn5a_branch2b2/gamma:0', b'bn5a_branch2b2/beta:0', b'bn5a_branch2b2/moving_mean:0', b'bn5a_branch2b2/moving_variance:0'

f.copy("/model_weights/res5a_branch2b_relu", "/model_weights/res5a_branch2b_relu2")
del f["model_weights"]['res5a_branch2b_relu']

f.copy("/model_weights/res5a_branch2c/res5a_branch2c", "/model_weights/res5a_branch2c2/res5a_branch2c2")
del f["model_weights"]['res5a_branch2c']

f["model_weights"]["res5a_branch2c2"].attrs["weight_names"] = [b'res5a_branch2c2/kernel:0']

f.copy("/model_weights/res5a_branch1/res5a_branch1", "/model_weights/res5a_branch12/res5a_branch12")
del f["model_weights"]['res5a_branch1']

f["model_weights"]["res5a_branch12"].attrs["weight_names"] = [b'res5a_branch12/kernel:0']

f.copy("/model_weights/bn5a_branch2c/bn5a_branch2c", "/model_weights/bn5a_branch2c2/bn5a_branch2c2")
del f["model_weights"]['bn5a_branch2c']

f["model_weights"]["bn5a_branch2c2"].attrs["weight_names"] = b'bn5a_branch2c2/gamma:0', b'bn5a_branch2c2/beta:0', b'bn5a_branch2c2/moving_mean:0', b'bn5a_branch2c2/moving_variance:0'

f.copy("/model_weights/bn5a_branch1/bn5a_branch1", "/model_weights/bn5a_branch12/bn5a_branch12")
del f["model_weights"]['bn5a_branch1']

f["model_weights"]["bn5a_branch12"].attrs["weight_names"] = b'bn5a_branch12/gamma:0', b'bn5a_branch12/beta:0', b'bn5a_branch12/moving_mean:0', b'bn5a_branch12/moving_variance:0'

f.copy("/model_weights/res5a", "/model_weights/res5a2")
del f["model_weights"]['res5a']

f.copy("/model_weights/res5a_relu", "/model_weights/res5a_relu2")
del f["model_weights"]['res5a_relu']

f.copy("/model_weights/res5b_branch2a/res5b_branch2a", "/model_weights/res5b_branch2a2/res5b_branch2a2")
del f["model_weights"]['res5b_branch2a']

f["model_weights"]["res5b_branch2a2"].attrs["weight_names"] = [b'res5b_branch2a2/kernel:0']

f.copy("/model_weights/bn5b_branch2a/bn5b_branch2a", "/model_weights/bn5b_branch2a2/bn5b_branch2a2")
del f["model_weights"]['bn5b_branch2a']

f["model_weights"]["bn5b_branch2a2"].attrs["weight_names"] = b'bn5b_branch2a2/gamma:0', b'bn5b_branch2a2/beta:0', b'bn5b_branch2a2/moving_mean:0', b'bn5b_branch2a2/moving_variance:0'

f.copy("/model_weights/res5b_branch2a_relu", "/model_weights/res5b_branch2a_relu2")
del f["model_weights"]['res5b_branch2a_relu']

f.copy("/model_weights/padding5b_branch2b", "/model_weights/padding5b_branch2b2")
del f["model_weights"]['padding5b_branch2b']

f.copy("/model_weights/res5b_branch2b/res5b_branch2b", "/model_weights/res5b_branch2b2/res5b_branch2b2")
del f["model_weights"]['res5b_branch2b']

f["model_weights"]["res5b_branch2b2"].attrs["weight_names"] = [b'res5b_branch2b2/kernel:0']

f.copy("/model_weights/bn5b_branch2b/bn5b_branch2b", "/model_weights/bn5b_branch2b2/bn5b_branch2b2")
del f["model_weights"]['bn5b_branch2b']

f["model_weights"]["bn5b_branch2b2"].attrs["weight_names"] = b'bn5b_branch2b2/gamma:0', b'bn5b_branch2b2/beta:0', b'bn5b_branch2b2/moving_mean:0', b'bn5b_branch2b2/moving_variance:0'

f.copy("/model_weights/res5b_branch2b_relu", "/model_weights/res5b_branch2b_relu2")
del f["model_weights"]['res5b_branch2b_relu']

f.copy("/model_weights/res5b_branch2c/res5b_branch2c", "/model_weights/res5b_branch2c2/res5b_branch2c2")
del f["model_weights"]['res5b_branch2c']

f["model_weights"]["res5b_branch2c2"].attrs["weight_names"] = [b'res5b_branch2c2/kernel:0']

f.copy("/model_weights/bn5b_branch2c/bn5b_branch2c", "/model_weights/bn5b_branch2c2/bn5b_branch2c2")
del f["model_weights"]['bn5b_branch2c']

f["model_weights"]["bn5b_branch2c2"].attrs["weight_names"] = b'bn5b_branch2c2/gamma:0', b'bn5b_branch2c2/beta:0', b'bn5b_branch2c2/moving_mean:0', b'bn5b_branch2c2/moving_variance:0'

f.copy("/model_weights/res5b", "/model_weights/res5b2")
del f["model_weights"]['res5b']

f.copy("/model_weights/res5b_relu", "/model_weights/res5b_relu2")
del f["model_weights"]['res5b_relu']

f.copy("/model_weights/res5c_branch2a/res5c_branch2a", "/model_weights/res5c_branch2a2/res5c_branch2a2")
del f["model_weights"]['res5c_branch2a']

f["model_weights"]["res5c_branch2a2"].attrs["weight_names"] = [b'res5c_branch2a2/kernel:0']

f.copy("/model_weights/bn5c_branch2a/bn5c_branch2a", "/model_weights/bn5c_branch2a2/bn5c_branch2a2")
del f["model_weights"]['bn5c_branch2a']

f["model_weights"]["bn5c_branch2a2"].attrs["weight_names"] = b'bn5c_branch2a2/gamma:0', b'bn5c_branch2a2/beta:0', b'bn5c_branch2a2/moving_mean:0', b'bn5c_branch2a2/moving_variance:0'

f.copy("/model_weights/res5c_branch2a_relu", "/model_weights/res5c_branch2a_relu2")
del f["model_weights"]['res5c_branch2a_relu']

f.copy("/model_weights/padding5c_branch2b", "/model_weights/padding5c_branch2b2")
del f["model_weights"]['padding5c_branch2b']

f.copy("/model_weights/res5c_branch2b/res5c_branch2b", "/model_weights/res5c_branch2b2/res5c_branch2b2")
del f["model_weights"]['res5c_branch2b']

f["model_weights"]["res5c_branch2b2"].attrs["weight_names"] = [b'res5c_branch2b2/kernel:0']

f.copy("/model_weights/bn5c_branch2b/bn5c_branch2b", "/model_weights/bn5c_branch2b2/bn5c_branch2b2")
del f["model_weights"]['bn5c_branch2b']

f["model_weights"]["bn5c_branch2b2"].attrs["weight_names"] = b'bn5c_branch2b2/gamma:0', b'bn5c_branch2b2/beta:0', b'bn5c_branch2b2/moving_mean:0', b'bn5c_branch2b2/moving_variance:0'

f.copy("/model_weights/res5c_branch2b_relu", "/model_weights/res5c_branch2b_relu2")
del f["model_weights"]['res5c_branch2b_relu']

f.copy("/model_weights/res5c_branch2c/res5c_branch2c", "/model_weights/res5c_branch2c2/res5c_branch2c2")
del f["model_weights"]['res5c_branch2c']

f["model_weights"]["res5c_branch2c2"].attrs["weight_names"] = [b'res5c_branch2c2/kernel:0']

f.copy("/model_weights/bn5c_branch2c/bn5c_branch2c", "/model_weights/bn5c_branch2c2/bn5c_branch2c2")
del f["model_weights"]['bn5c_branch2c']

f["model_weights"]["bn5c_branch2c2"].attrs["weight_names"] = b'bn5c_branch2c2/gamma:0', b'bn5c_branch2c2/beta:0', b'bn5c_branch2c2/moving_mean:0', b'bn5c_branch2c2/moving_variance:0'

f.copy("/model_weights/res5c", "/model_weights/res5c2")
del f["model_weights"]['res5c']

f.copy("/model_weights/res5c_relu", "/model_weights/res5c_relu2")
del f["model_weights"]['res5c_relu']

f.copy("/model_weights/C5_reduced/C5_reduced", "/model_weights/C5_reduced2/C5_reduced2")
del f["model_weights"]['C5_reduced']

f["model_weights"]["C5_reduced2"].attrs["weight_names"] = b'C5_reduced2/kernel:0', b'C5_reduced2/bias:0'

f.copy("/model_weights/P5_upsampled", "/model_weights/P5_upsampled2")
del f["model_weights"]['P5_upsampled']

f.copy("/model_weights/C4_reduced/C4_reduced", "/model_weights/C4_reduced2/C4_reduced2")
del f["model_weights"]['C4_reduced']

f["model_weights"]["C4_reduced2"].attrs["weight_names"] = b'C4_reduced2/kernel:0', b'C4_reduced2/bias:0'

f.copy("/model_weights/P4_merged", "/model_weights/P4_merged2")
del f["model_weights"]['P4_merged']

f.copy("/model_weights/P4_upsampled", "/model_weights/P4_upsampled2")
del f["model_weights"]['P4_upsampled']

f.copy("/model_weights/C3_reduced/C3_reduced", "/model_weights/C3_reduced2/C3_reduced2")
del f["model_weights"]['C3_reduced']

f["model_weights"]["C3_reduced2"].attrs["weight_names"] = b'C3_reduced2/kernel:0', b'C3_reduced2/bias:0'

f.copy("/model_weights/P3_merged", "/model_weights/P3_merged2")
del f["model_weights"]['P3_merged']

f.copy("/model_weights/C6_relu", "/model_weights/C6_relu2")
del f["model_weights"]['C6_relu']

f.copy("/model_weights/regression_submodel", "/model_weights/regression_submodel2")
del f["model_weights"]['regression_submodel']

f["model_weights"]["regression_submodel2"].attrs["weight_names"] = b'pyramid_regression_0/kernel:0', b'pyramid_regression_0/bias:0', b'pyramid_regression_1/kernel:0', b'pyramid_regression_1/bias:0',  b'pyramid_regression_2/kernel:0', b'pyramid_regression_2/bias:0', b'pyramid_regression_3/kernel:0', b'pyramid_regression_3/bias:0', b'pyramid_regression/kernel:0', b'pyramid_regression/bias:0'

f.copy("/model_weights/regression", "/model_weights/regression2")
del f["model_weights"]['regression']

f.copy("/model_weights/classification_submodel", "/model_weights/classification_submodel2")
del f["model_weights"]['classification_submodel']

f.copy("/model_weights/classification", "/model_weights/classification2")
del f["model_weights"]['classification']

with open("/home/rblin/Documents/Image-Processing/Neural-networks/model_config.txt", "r") as t:
	text = t.readlines()
	#print(text[-1].encode('utf-8'))
	#f.attrs["model_config"] = text[-1][:-1].encode('utf-8')
	f.attrs["model_config"] = text[-1].encode('utf-8')

with open("/home/rblin/Documents/Image-Processing/Neural-networks/training_config.txt", "r") as t:
	text = t.readlines()
	#f.attrs["training_config"] = text[-1][:-1].encode('utf-8')
	f.attrs["training_config"] = text[-1].encode('utf-8')

with open("/home/rblin/Documents/Image-Processing/Neural-networks/layer_names.txt", "r") as t:
	text = t.readlines()
	#np_array = np.array(text[-1][:-1].split(',')).astype(np.bytes_)
	np_array = np.array(text[-1].split(',')).astype(np.bytes_)
	f["model_weights"].attrs["layer_names"] = np_array

f.close()

