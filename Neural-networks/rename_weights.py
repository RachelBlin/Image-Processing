#import keras

"""model = keras.models.load_model("/home/rblin/Documents/weights/test_rename/temp.h5")
model.summary()


for i, layer in enumerate(model.layers):
    layer.name = 'layer_' + str(i)

model.summary()"""



import h5py
import numpy as np


def rename_resnet50(filepath):

    f = h5py.File(filepath, "a")

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

def rename_resnet101(filepath):

    f = h5py.File(filepath, "a")

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

    f.copy("/model_weights/bn_conv1/bn_conv1", "/model_weights/bn_conv12/bn_conv12")
    del f["model_weights"]['bn_conv1']

    f["model_weights"]["bn_conv12"].attrs["weight_names"] = b'bn_conv12/gamma:0', b'bn_conv12/beta:0', b'bn_conv12/moving_mean:0', b'bn_conv12/moving_variance:0'

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

    f["model_weights"]["bn3a_branch12"].attrs[
        "weight_names"] = b'bn3a_branch12/gamma:0', b'bn3a_branch12/beta:0', b'bn3a_branch12/moving_mean:0', b'bn3a_branch12/moving_variance:0'

    f.copy("/model_weights/res3a", "/model_weights/res3a2")
    del f["model_weights"]['res3a']

    f.copy("/model_weights/res3a_relu", "/model_weights/res3a_relu2")
    del f["model_weights"]['res3a_relu']

    f.copy("/model_weights/res3b1_branch2a/res3b1_branch2a", "/model_weights/res3b1_branch2a2/res3b1_branch2a2")
    del f["model_weights"]['res3b1_branch2a']

    f["model_weights"]["res3b1_branch2a2"].attrs["weight_names"] = [b'res3b1_branch2a2/kernel:0']

    f.copy("/model_weights/bn3b1_branch2a/bn3b1_branch2a", "/model_weights/bn3b1_branch2a2/bn3b1_branch2a2")
    del f["model_weights"]['bn3b1_branch2a']

    f["model_weights"]["bn3b1_branch2a2"].attrs["weight_names"] = b'bn3b1_branch2a2/gamma:0', b'bn3b1_branch2a2/beta:0', b'bn3b1_branch2a2/moving_mean:0', b'bn3b1_branch2a2/moving_variance:0'

    f.copy("/model_weights/res3b1_branch2a_relu", "/model_weights/res3b1_branch2a_relu2")
    del f["model_weights"]['res3b1_branch2a_relu']

    f.copy("/model_weights/padding3b1_branch2b", "/model_weights/padding3b1_branch2b2")
    del f["model_weights"]['padding3b1_branch2b']

    f.copy("/model_weights/res3b1_branch2b/res3b1_branch2b", "/model_weights/res3b1_branch2b2/res3b1_branch2b2")
    del f["model_weights"]['res3b1_branch2b']

    f["model_weights"]["res3b1_branch2b2"].attrs["weight_names"] = [b'res3b1_branch2b2/kernel:0']

    f.copy("/model_weights/bn3b1_branch2b/bn3b1_branch2b", "/model_weights/bn3b1_branch2b2/bn3b1_branch2b2")
    del f["model_weights"]['bn3b1_branch2b']

    f["model_weights"]["bn3b1_branch2b2"].attrs["weight_names"] = b'bn3b1_branch2b2/gamma:0', b'bn3b1_branch2b2/beta:0', b'bn3b1_branch2b2/moving_mean:0', b'bn3b1_branch2b2/moving_variance:0'

    f.copy("/model_weights/res3b1_branch2b_relu", "/model_weights/res3b1_branch2b_relu2")
    del f["model_weights"]['res3b1_branch2b_relu']

    f.copy("/model_weights/res3b1_branch2c/res3b1_branch2c", "/model_weights/res3b1_branch2c2/res3b1_branch2c2")
    del f["model_weights"]['res3b1_branch2c']

    f["model_weights"]["res3b1_branch2c2"].attrs["weight_names"] = [b'res3b1_branch2c2/kernel:0']

    f.copy("/model_weights/bn3b1_branch2c/bn3b1_branch2c", "/model_weights/bn3b1_branch2c2/bn3b1_branch2c2")
    del f["model_weights"]['bn3b1_branch2c']

    f["model_weights"]["bn3b1_branch2c2"].attrs["weight_names"] = b'bn3b1_branch2c2/gamma:0', b'bn3b1_branch2c2/beta:0', b'bn3b1_branch2c2/moving_mean:0', b'bn3b1_branch2c2/moving_variance:0'

    f.copy("/model_weights/res3b1", "/model_weights/res3b12")
    del f["model_weights"]['res3b1']

    f.copy("/model_weights/res3b1_relu", "/model_weights/res3b1_relu2")
    del f["model_weights"]['res3b1_relu']

    f.copy("/model_weights/res3b2_branch2a/res3b2_branch2a", "/model_weights/res3b2_branch2a2/res3b2_branch2a2")
    del f["model_weights"]['res3b2_branch2a']

    f["model_weights"]["res3b2_branch2a2"].attrs["weight_names"] = [b'res3b2_branch2a2/kernel:0']

    f.copy("/model_weights/bn3b2_branch2a/bn3b2_branch2a", "/model_weights/bn3b2_branch2a2/bn3b2_branch2a2")
    del f["model_weights"]['bn3b2_branch2a']

    f["model_weights"]["bn3b2_branch2a2"].attrs["weight_names"] = b'bn3b2_branch2a2/gamma:0', b'bn3b2_branch2a2/beta:0', b'bn3b2_branch2a2/moving_mean:0', b'bn3b2_branch2a2/moving_variance:0'

    f.copy("/model_weights/res3b2_branch2a_relu", "/model_weights/res3b2_branch2a_relu2")
    del f["model_weights"]['res3b2_branch2a_relu']

    f.copy("/model_weights/padding3b2_branch2b", "/model_weights/padding3b2_branch2b2")
    del f["model_weights"]['padding3b2_branch2b']

    f.copy("/model_weights/res3b2_branch2b/res3b2_branch2b", "/model_weights/res3b2_branch2b2/res3b2_branch2b2")
    del f["model_weights"]['res3b2_branch2b']

    f["model_weights"]["res3b2_branch2b2"].attrs["weight_names"] = [b'res3b2_branch2b2/kernel:0']

    f.copy("/model_weights/bn3b2_branch2b/bn3b2_branch2b", "/model_weights/bn3b2_branch2b2/bn3b2_branch2b2")
    del f["model_weights"]['bn3b2_branch2b']

    f["model_weights"]["bn3b2_branch2b2"].attrs["weight_names"] = b'bn3b2_branch2b2/gamma:0', b'bn3b2_branch2b2/beta:0', b'bn3b2_branch2b2/moving_mean:0', b'bn3b2_branch2b2/moving_variance:0'

    f.copy("/model_weights/res3b2_branch2b_relu", "/model_weights/res3b2_branch2b_relu2")
    del f["model_weights"]['res3b2_branch2b_relu']

    f.copy("/model_weights/res3b2_branch2c/res3b2_branch2c", "/model_weights/res3b2_branch2c2/res3b2_branch2c2")
    del f["model_weights"]['res3b2_branch2c']

    f["model_weights"]["res3b2_branch2c2"].attrs["weight_names"] = [b'res3b2_branch2c2/kernel:0']

    f.copy("/model_weights/bn3b2_branch2c/bn3b2_branch2c", "/model_weights/bn3b2_branch2c2/bn3b2_branch2c2")
    del f["model_weights"]['bn3b2_branch2c']

    f["model_weights"]["bn3b2_branch2c2"].attrs["weight_names"] = b'bn3b2_branch2c2/gamma:0', b'bn3b2_branch2c2/beta:0', b'bn3b2_branch2c2/moving_mean:0', b'bn3b2_branch2c2/moving_variance:0'

    f.copy("/model_weights/res3b2", "/model_weights/res3b22")
    del f["model_weights"]['res3b2']

    f.copy("/model_weights/res3b2_relu", "/model_weights/res3b2_relu2")
    del f["model_weights"]['res3b2_relu']

    f.copy("/model_weights/res3b3_branch2a/res3b3_branch2a", "/model_weights/res3b3_branch2a2/res3b3_branch2a2")
    del f["model_weights"]['res3b3_branch2a']

    f["model_weights"]["res3b3_branch2a2"].attrs["weight_names"] = [b'res3b3_branch2a2/kernel:0']

    f.copy("/model_weights/bn3b3_branch2a/bn3b3_branch2a", "/model_weights/bn3b3_branch2a2/bn3b3_branch2a2")
    del f["model_weights"]['bn3b3_branch2a']

    f["model_weights"]["bn3b3_branch2a2"].attrs["weight_names"] = b'bn3b3_branch2a2/gamma:0', b'bn3b3_branch2a2/beta:0', b'bn3b3_branch2a2/moving_mean:0', b'bn3b3_branch2a2/moving_variance:0'

    f.copy("/model_weights/res3b3_branch2a_relu", "/model_weights/res3b3_branch2a_relu2")
    del f["model_weights"]['res3b3_branch2a_relu']

    f.copy("/model_weights/padding3b3_branch2b", "/model_weights/padding3b3_branch2b2")
    del f["model_weights"]['padding3b3_branch2b']

    f.copy("/model_weights/res3b3_branch2b/res3b3_branch2b", "/model_weights/res3b3_branch2b2/res3b3_branch2b2")
    del f["model_weights"]['res3b3_branch2b']

    f["model_weights"]["res3b3_branch2b2"].attrs["weight_names"] = [b'res3b3_branch2b2/kernel:0']

    f.copy("/model_weights/bn3b3_branch2b/bn3b3_branch2b", "/model_weights/bn3b3_branch2b2/bn3b3_branch2b2")
    del f["model_weights"]['bn3b3_branch2b']

    f["model_weights"]["bn3b3_branch2b2"].attrs["weight_names"] = b'bn3b3_branch2b2/gamma:0', b'bn3b3_branch2b2/beta:0', b'bn3b3_branch2b2/moving_mean:0', b'bn3b3_branch2b2/moving_variance:0'

    f.copy("/model_weights/res3b3_branch2b_relu", "/model_weights/res3b3_branch2b_relu2")
    del f["model_weights"]['res3b3_branch2b_relu']

    f.copy("/model_weights/res3b3_branch2c/res3b3_branch2c", "/model_weights/res3b3_branch2c2/res3b3_branch2c2")
    del f["model_weights"]['res3b3_branch2c']

    f["model_weights"]["res3b3_branch2c2"].attrs["weight_names"] = [b'res3b3_branch2c2/kernel:0']

    f.copy("/model_weights/bn3b3_branch2c/bn3b3_branch2c", "/model_weights/bn3b3_branch2c2/bn3b3_branch2c2")
    del f["model_weights"]['bn3b3_branch2c']

    f["model_weights"]["bn3b3_branch2c2"].attrs["weight_names"] = b'bn3b3_branch2c2/gamma:0', b'bn3b3_branch2c2/beta:0', b'bn3b3_branch2c2/moving_mean:0', b'bn3b3_branch2c2/moving_variance:0'

    f.copy("/model_weights/res3b3", "/model_weights/res3b32")
    del f["model_weights"]['res3b3']

    f.copy("/model_weights/res3b3_relu", "/model_weights/res3b3_relu2")
    del f["model_weights"]['res3b3_relu']

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

    f.copy("/model_weights/res4b1_branch2a/res4b1_branch2a", "/model_weights/res4b1_branch2a2/res4b1_branch2a2")
    del f["model_weights"]['res4b1_branch2a']

    f["model_weights"]["res4b1_branch2a2"].attrs["weight_names"] = [b'res4b1_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b1_branch2a/bn4b1_branch2a", "/model_weights/bn4b1_branch2a2/bn4b1_branch2a2")
    del f["model_weights"]['bn4b1_branch2a']

    f["model_weights"]["bn4b1_branch2a2"].attrs["weight_names"] = b'bn4b1_branch2a2/gamma:0', b'bn4b1_branch2a2/beta:0', b'bn4b1_branch2a2/moving_mean:0', b'bn4b1_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b1_branch2a_relu", "/model_weights/res4b1_branch2a_relu2")
    del f["model_weights"]['res4b1_branch2a_relu']

    f.copy("/model_weights/padding4b1_branch2b", "/model_weights/padding4b1_branch2b2")
    del f["model_weights"]['padding4b1_branch2b']

    f.copy("/model_weights/res4b1_branch2b/res4b1_branch2b", "/model_weights/res4b1_branch2b2/res4b1_branch2b2")
    del f["model_weights"]['res4b1_branch2b']

    f["model_weights"]["res4b1_branch2b2"].attrs["weight_names"] = [b'res4b1_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b1_branch2b/bn4b1_branch2b", "/model_weights/bn4b1_branch2b2/bn4b1_branch2b2")
    del f["model_weights"]['bn4b1_branch2b']

    f["model_weights"]["bn4b1_branch2b2"].attrs["weight_names"] = b'bn4b1_branch2b2/gamma:0', b'bn4b1_branch2b2/beta:0', b'bn4b1_branch2b2/moving_mean:0', b'bn4b1_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b1_branch2b_relu", "/model_weights/res4b1_branch2b_relu2")
    del f["model_weights"]['res4b1_branch2b_relu']

    f.copy("/model_weights/res4b1_branch2c/res4b1_branch2c", "/model_weights/res4b1_branch2c2/res4b1_branch2c2")
    del f["model_weights"]['res4b1_branch2c']

    f["model_weights"]["res4b1_branch2c2"].attrs["weight_names"] = [b'res4b1_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b1_branch2c/bn4b1_branch2c", "/model_weights/bn4b1_branch2c2/bn4b1_branch2c2")
    del f["model_weights"]['bn4b1_branch2c']

    f["model_weights"]["bn4b1_branch2c2"].attrs["weight_names"] = b'bn4b1_branch2c2/gamma:0', b'bn4b1_branch2c2/beta:0', b'bn4b1_branch2c2/moving_mean:0', b'bn4b1_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b12", "/model_weights/res4b1222")
    del f["model_weights"]['res4b12']

    f.copy("/model_weights/res4b1", "/model_weights/res4b122")
    del f["model_weights"]['res4b1']

    f.copy("/model_weights/res4b1_relu", "/model_weights/res4b1_relu2")
    del f["model_weights"]['res4b1_relu']

    f.copy("/model_weights/res4b2_branch2a/res4b2_branch2a", "/model_weights/res4b2_branch2a2/res4b2_branch2a2")
    del f["model_weights"]['res4b2_branch2a']

    f["model_weights"]["res4b2_branch2a2"].attrs["weight_names"] = [b'res4b2_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b2_branch2a/bn4b2_branch2a", "/model_weights/bn4b2_branch2a2/bn4b2_branch2a2")
    del f["model_weights"]['bn4b2_branch2a']

    f["model_weights"]["bn4b2_branch2a2"].attrs["weight_names"] = b'bn4b2_branch2a2/gamma:0', b'bn4b2_branch2a2/beta:0', b'bn4b2_branch2a2/moving_mean:0', b'bn4b2_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b2_branch2a_relu", "/model_weights/res4b2_branch2a_relu2")
    del f["model_weights"]['res4b2_branch2a_relu']

    f.copy("/model_weights/padding4b2_branch2b", "/model_weights/padding4b2_branch2b2")
    del f["model_weights"]['padding4b2_branch2b']

    f.copy("/model_weights/res4b2_branch2b/res4b2_branch2b", "/model_weights/res4b2_branch2b2/res4b2_branch2b2")
    del f["model_weights"]['res4b2_branch2b']

    f["model_weights"]["res4b2_branch2b2"].attrs["weight_names"] = [b'res4b2_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b2_branch2b/bn4b2_branch2b", "/model_weights/bn4b2_branch2b2/bn4b2_branch2b2")
    del f["model_weights"]['bn4b2_branch2b']

    f["model_weights"]["bn4b2_branch2b2"].attrs["weight_names"] = b'bn4b2_branch2b2/gamma:0', b'bn4b2_branch2b2/beta:0', b'bn4b2_branch2b2/moving_mean:0', b'bn4b2_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b2_branch2b_relu", "/model_weights/res4b2_branch2b_relu2")
    del f["model_weights"]['res4b2_branch2b_relu']

    f.copy("/model_weights/res4b2_branch2c/res4b2_branch2c", "/model_weights/res4b2_branch2c2/res4b2_branch2c2")
    del f["model_weights"]['res4b2_branch2c']

    f["model_weights"]["res4b2_branch2c2"].attrs["weight_names"] = [b'res4b2_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b2_branch2c/bn4b2_branch2c", "/model_weights/bn4b2_branch2c2/bn4b2_branch2c2")
    del f["model_weights"]['bn4b2_branch2c']

    f["model_weights"]["bn4b2_branch2c2"].attrs["weight_names"] = b'bn4b2_branch2c2/gamma:0', b'bn4b2_branch2c2/beta:0', b'bn4b2_branch2c2/moving_mean:0', b'bn4b2_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b22", "/model_weights/res4b2222")
    del f["model_weights"]['res4b22']

    f.copy("/model_weights/res4b2", "/model_weights/res4b222")
    del f["model_weights"]['res4b2']

    f.copy("/model_weights/res4b2_relu", "/model_weights/res4b2_relu2")
    del f["model_weights"]['res4b2_relu']

    f.copy("/model_weights/res4b3_branch2a/res4b3_branch2a", "/model_weights/res4b3_branch2a2/res4b3_branch2a2")
    del f["model_weights"]['res4b3_branch2a']

    f["model_weights"]["res4b3_branch2a2"].attrs["weight_names"] = [b'res4b3_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b3_branch2a/bn4b3_branch2a", "/model_weights/bn4b3_branch2a2/bn4b3_branch2a2")
    del f["model_weights"]['bn4b3_branch2a']

    f["model_weights"]["bn4b3_branch2a2"].attrs["weight_names"] = b'bn4b3_branch2a2/gamma:0', b'bn4b3_branch2a2/beta:0', b'bn4b3_branch2a2/moving_mean:0', b'bn4b3_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b3_branch2a_relu", "/model_weights/res4b3_branch2a_relu2")
    del f["model_weights"]['res4b3_branch2a_relu']

    f.copy("/model_weights/padding4b3_branch2b", "/model_weights/padding4b3_branch2b2")
    del f["model_weights"]['padding4b3_branch2b']

    f.copy("/model_weights/res4b3_branch2b/res4b3_branch2b", "/model_weights/res4b3_branch2b2/res4b3_branch2b2")
    del f["model_weights"]['res4b3_branch2b']

    f["model_weights"]["res4b3_branch2b2"].attrs["weight_names"] = [b'res4b3_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b3_branch2b/bn4b3_branch2b", "/model_weights/bn4b3_branch2b2/bn4b3_branch2b2")
    del f["model_weights"]['bn4b3_branch2b']

    f["model_weights"]["bn4b3_branch2b2"].attrs["weight_names"] = b'bn4b3_branch2b2/gamma:0', b'bn4b3_branch2b2/beta:0', b'bn4b3_branch2b2/moving_mean:0', b'bn4b3_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b3_branch2b_relu", "/model_weights/res4b3_branch2b_relu2")
    del f["model_weights"]['res4b3_branch2b_relu']

    f.copy("/model_weights/res4b3_branch2c/res4b3_branch2c", "/model_weights/res4b3_branch2c2/res4b3_branch2c2")
    del f["model_weights"]['res4b3_branch2c']

    f["model_weights"]["res4b3_branch2c2"].attrs["weight_names"] = [b'res4b3_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b3_branch2c/bn4b3_branch2c", "/model_weights/bn4b3_branch2c2/bn4b3_branch2c2")
    del f["model_weights"]['bn4b3_branch2c']

    f["model_weights"]["bn4b3_branch2c2"].attrs["weight_names"] = b'bn4b3_branch2c2/gamma:0', b'bn4b3_branch2c2/beta:0', b'bn4b3_branch2c2/moving_mean:0', b'bn4b3_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b3", "/model_weights/res4b32")
    del f["model_weights"]['res4b3']

    f.copy("/model_weights/res4b3_relu", "/model_weights/res4b3_relu2")
    del f["model_weights"]['res4b3_relu']

    f.copy("/model_weights/res4b4_branch2a/res4b4_branch2a", "/model_weights/res4b4_branch2a2/res4b4_branch2a2")
    del f["model_weights"]['res4b4_branch2a']

    f["model_weights"]["res4b4_branch2a2"].attrs["weight_names"] = [b'res4b4_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b4_branch2a/bn4b4_branch2a", "/model_weights/bn4b4_branch2a2/bn4b4_branch2a2")
    del f["model_weights"]['bn4b4_branch2a']

    f["model_weights"]["bn4b4_branch2a2"].attrs["weight_names"] = b'bn4b4_branch2a2/gamma:0', b'bn4b4_branch2a2/beta:0', b'bn4b4_branch2a2/moving_mean:0', b'bn4b4_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b4_branch2a_relu", "/model_weights/res4b4_branch2a_relu2")
    del f["model_weights"]['res4b4_branch2a_relu']

    f.copy("/model_weights/padding4b4_branch2b", "/model_weights/padding4b4_branch2b2")
    del f["model_weights"]['padding4b4_branch2b']

    f.copy("/model_weights/res4b4_branch2b/res4b4_branch2b", "/model_weights/res4b4_branch2b2/res4b4_branch2b2")
    del f["model_weights"]['res4b4_branch2b']

    f["model_weights"]["res4b4_branch2b2"].attrs["weight_names"] = [b'res4b4_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b4_branch2b/bn4b4_branch2b", "/model_weights/bn4b4_branch2b2/bn4b4_branch2b2")
    del f["model_weights"]['bn4b4_branch2b']

    f["model_weights"]["bn4b4_branch2b2"].attrs["weight_names"] = b'bn4b4_branch2b2/gamma:0', b'bn4b4_branch2b2/beta:0', b'bn4b4_branch2b2/moving_mean:0', b'bn4b4_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b4_branch2b_relu", "/model_weights/res4b4_branch2b_relu2")
    del f["model_weights"]['res4b4_branch2b_relu']

    f.copy("/model_weights/res4b4_branch2c/res4b4_branch2c", "/model_weights/res4b4_branch2c2/res4b4_branch2c2")
    del f["model_weights"]['res4b4_branch2c']

    f["model_weights"]["res4b4_branch2c2"].attrs["weight_names"] = [b'res4b4_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b4_branch2c/bn4b4_branch2c", "/model_weights/bn4b4_branch2c2/bn4b4_branch2c2")
    del f["model_weights"]['bn4b4_branch2c']

    f["model_weights"]["bn4b4_branch2c2"].attrs["weight_names"] = b'bn4b4_branch2c2/gamma:0', b'bn4b4_branch2c2/beta:0', b'bn4b4_branch2c2/moving_mean:0', b'bn4b4_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b4", "/model_weights/res4b42")
    del f["model_weights"]['res4b4']

    f.copy("/model_weights/res4b4_relu", "/model_weights/res4b4_relu2")
    del f["model_weights"]['res4b4_relu']

    f.copy("/model_weights/res4b5_branch2a/res4b5_branch2a", "/model_weights/res4b5_branch2a2/res4b5_branch2a2")
    del f["model_weights"]['res4b5_branch2a']

    f["model_weights"]["res4b5_branch2a2"].attrs["weight_names"] = [b'res4b5_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b5_branch2a/bn4b5_branch2a", "/model_weights/bn4b5_branch2a2/bn4b5_branch2a2")
    del f["model_weights"]['bn4b5_branch2a']

    f["model_weights"]["bn4b5_branch2a2"].attrs["weight_names"] = b'bn4b5_branch2a2/gamma:0', b'bn4b5_branch2a2/beta:0', b'bn4b5_branch2a2/moving_mean:0', b'bn4b5_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b5_branch2a_relu", "/model_weights/res4b5_branch2a_relu2")
    del f["model_weights"]['res4b5_branch2a_relu']

    f.copy("/model_weights/padding4b5_branch2b", "/model_weights/padding4b5_branch2b2")
    del f["model_weights"]['padding4b5_branch2b']

    f.copy("/model_weights/res4b5_branch2b/res4b5_branch2b", "/model_weights/res4b5_branch2b2/res4b5_branch2b2")
    del f["model_weights"]['res4b5_branch2b']

    f["model_weights"]["res4b5_branch2b2"].attrs["weight_names"] = [b'res4b5_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b5_branch2b/bn4b5_branch2b", "/model_weights/bn4b5_branch2b2/bn4b5_branch2b2")
    del f["model_weights"]['bn4b5_branch2b']

    f["model_weights"]["bn4b5_branch2b2"].attrs["weight_names"] = b'bn4b5_branch2b2/gamma:0', b'bn4b5_branch2b2/beta:0', b'bn4b5_branch2b2/moving_mean:0', b'bn4b5_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b5_branch2b_relu", "/model_weights/res4b5_branch2b_relu2")
    del f["model_weights"]['res4b5_branch2b_relu']

    f.copy("/model_weights/res4b5_branch2c/res4b5_branch2c", "/model_weights/res4b5_branch2c2/res4b5_branch2c2")
    del f["model_weights"]['res4b5_branch2c']

    f["model_weights"]["res4b5_branch2c2"].attrs["weight_names"] = [b'res4b5_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b5_branch2c/bn4b5_branch2c", "/model_weights/bn4b5_branch2c2/bn4b5_branch2c2")
    del f["model_weights"]['bn4b5_branch2c']

    f["model_weights"]["bn4b5_branch2c2"].attrs["weight_names"] = b'bn4b5_branch2c2/gamma:0', b'bn4b5_branch2c2/beta:0', b'bn4b5_branch2c2/moving_mean:0', b'bn4b5_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b5", "/model_weights/res4b52")
    del f["model_weights"]['res4b5']

    f.copy("/model_weights/res4b5_relu", "/model_weights/res4b5_relu2")
    del f["model_weights"]['res4b5_relu']

    f.copy("/model_weights/res4b6_branch2a/res4b6_branch2a", "/model_weights/res4b6_branch2a2/res4b6_branch2a2")
    del f["model_weights"]['res4b6_branch2a']

    f["model_weights"]["res4b6_branch2a2"].attrs["weight_names"] = [b'res4b6_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b6_branch2a/bn4b6_branch2a", "/model_weights/bn4b6_branch2a2/bn4b6_branch2a2")
    del f["model_weights"]['bn4b6_branch2a']

    f["model_weights"]["bn4b6_branch2a2"].attrs["weight_names"] = b'bn4b6_branch2a2/gamma:0', b'bn4b6_branch2a2/beta:0', b'bn4b6_branch2a2/moving_mean:0', b'bn4b6_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b6_branch2a_relu", "/model_weights/res4b6_branch2a_relu2")
    del f["model_weights"]['res4b6_branch2a_relu']

    f.copy("/model_weights/padding4b6_branch2b", "/model_weights/padding4b6_branch2b2")
    del f["model_weights"]['padding4b6_branch2b']

    f.copy("/model_weights/res4b6_branch2b/res4b6_branch2b", "/model_weights/res4b6_branch2b2/res4b6_branch2b2")
    del f["model_weights"]['res4b6_branch2b']

    f["model_weights"]["res4b6_branch2b2"].attrs["weight_names"] = [b'res4b6_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b6_branch2b/bn4b6_branch2b", "/model_weights/bn4b6_branch2b2/bn4b6_branch2b2")
    del f["model_weights"]['bn4b6_branch2b']

    f["model_weights"]["bn4b6_branch2b2"].attrs["weight_names"] = b'bn4b6_branch2b2/gamma:0', b'bn4b6_branch2b2/beta:0', b'bn4b6_branch2b2/moving_mean:0', b'bn4b6_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b6_branch2b_relu", "/model_weights/res4b6_branch2b_relu2")
    del f["model_weights"]['res4b6_branch2b_relu']

    f.copy("/model_weights/res4b6_branch2c/res4b6_branch2c", "/model_weights/res4b6_branch2c2/res4b6_branch2c2")
    del f["model_weights"]['res4b6_branch2c']

    f["model_weights"]["res4b6_branch2c2"].attrs["weight_names"] = [b'res4b6_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b6_branch2c/bn4b6_branch2c", "/model_weights/bn4b6_branch2c2/bn4b6_branch2c2")
    del f["model_weights"]['bn4b6_branch2c']

    f["model_weights"]["bn4b6_branch2c2"].attrs["weight_names"] = b'bn4b6_branch2c2/gamma:0', b'bn4b6_branch2c2/beta:0', b'bn4b6_branch2c2/moving_mean:0', b'bn4b6_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b6", "/model_weights/res4b62")
    del f["model_weights"]['res4b6']

    f.copy("/model_weights/res4b6_relu", "/model_weights/res4b6_relu2")
    del f["model_weights"]['res4b6_relu']

    f.copy("/model_weights/res4b7_branch2a/res4b7_branch2a", "/model_weights/res4b7_branch2a2/res4b7_branch2a2")
    del f["model_weights"]['res4b7_branch2a']

    f["model_weights"]["res4b7_branch2a2"].attrs["weight_names"] = [b'res4b7_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b7_branch2a/bn4b7_branch2a", "/model_weights/bn4b7_branch2a2/bn4b7_branch2a2")
    del f["model_weights"]['bn4b7_branch2a']

    f["model_weights"]["bn4b7_branch2a2"].attrs["weight_names"] = b'bn4b7_branch2a2/gamma:0', b'bn4b7_branch2a2/beta:0', b'bn4b7_branch2a2/moving_mean:0', b'bn4b7_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b7_branch2a_relu", "/model_weights/res4b7_branch2a_relu2")
    del f["model_weights"]['res4b7_branch2a_relu']

    f.copy("/model_weights/padding4b7_branch2b", "/model_weights/padding4b7_branch2b2")
    del f["model_weights"]['padding4b7_branch2b']

    f.copy("/model_weights/res4b7_branch2b/res4b7_branch2b", "/model_weights/res4b7_branch2b2/res4b7_branch2b2")
    del f["model_weights"]['res4b7_branch2b']

    f["model_weights"]["res4b7_branch2b2"].attrs["weight_names"] = [b'res4b7_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b7_branch2b/bn4b7_branch2b", "/model_weights/bn4b7_branch2b2/bn4b7_branch2b2")
    del f["model_weights"]['bn4b7_branch2b']

    f["model_weights"]["bn4b7_branch2b2"].attrs["weight_names"] = b'bn4b7_branch2b2/gamma:0', b'bn4b7_branch2b2/beta:0', b'bn4b7_branch2b2/moving_mean:0', b'bn4b7_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b7_branch2b_relu", "/model_weights/res4b7_branch2b_relu2")
    del f["model_weights"]['res4b7_branch2b_relu']

    f.copy("/model_weights/res4b7_branch2c/res4b7_branch2c", "/model_weights/res4b7_branch2c2/res4b7_branch2c2")
    del f["model_weights"]['res4b7_branch2c']

    f["model_weights"]["res4b7_branch2c2"].attrs["weight_names"] = [b'res4b7_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b7_branch2c/bn4b7_branch2c", "/model_weights/bn4b7_branch2c2/bn4b7_branch2c2")
    del f["model_weights"]['bn4b7_branch2c']

    f["model_weights"]["bn4b7_branch2c2"].attrs["weight_names"] = b'bn4b7_branch2c2/gamma:0', b'bn4b7_branch2c2/beta:0', b'bn4b7_branch2c2/moving_mean:0', b'bn4b7_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b7", "/model_weights/res4b72")
    del f["model_weights"]['res4b7']

    f.copy("/model_weights/res4b7_relu", "/model_weights/res4b7_relu2")
    del f["model_weights"]['res4b7_relu']

    f.copy("/model_weights/res4b8_branch2a/res4b8_branch2a", "/model_weights/res4b8_branch2a2/res4b8_branch2a2")
    del f["model_weights"]['res4b8_branch2a']

    f["model_weights"]["res4b8_branch2a2"].attrs["weight_names"] = [b'res4b8_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b8_branch2a/bn4b8_branch2a", "/model_weights/bn4b8_branch2a2/bn4b8_branch2a2")
    del f["model_weights"]['bn4b8_branch2a']

    f["model_weights"]["bn4b8_branch2a2"].attrs["weight_names"] = b'bn4b8_branch2a2/gamma:0', b'bn4b8_branch2a2/beta:0', b'bn4b8_branch2a2/moving_mean:0', b'bn4b8_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b8_branch2a_relu", "/model_weights/res4b8_branch2a_relu2")
    del f["model_weights"]['res4b8_branch2a_relu']

    f.copy("/model_weights/padding4b8_branch2b", "/model_weights/padding4b8_branch2b2")
    del f["model_weights"]['padding4b8_branch2b']

    f.copy("/model_weights/res4b8_branch2b/res4b8_branch2b", "/model_weights/res4b8_branch2b2/res4b8_branch2b2")
    del f["model_weights"]['res4b8_branch2b']

    f["model_weights"]["res4b8_branch2b2"].attrs["weight_names"] = [b'res4b8_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b8_branch2b/bn4b8_branch2b", "/model_weights/bn4b8_branch2b2/bn4b8_branch2b2")
    del f["model_weights"]['bn4b8_branch2b']

    f["model_weights"]["bn4b8_branch2b2"].attrs["weight_names"] = b'bn4b8_branch2b2/gamma:0', b'bn4b8_branch2b2/beta:0', b'bn4b8_branch2b2/moving_mean:0', b'bn4b8_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b8_branch2b_relu", "/model_weights/res4b8_branch2b_relu2")
    del f["model_weights"]['res4b8_branch2b_relu']

    f.copy("/model_weights/res4b8_branch2c/res4b8_branch2c", "/model_weights/res4b8_branch2c2/res4b8_branch2c2")
    del f["model_weights"]['res4b8_branch2c']

    f["model_weights"]["res4b8_branch2c2"].attrs["weight_names"] = [b'res4b8_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b8_branch2c/bn4b8_branch2c", "/model_weights/bn4b8_branch2c2/bn4b8_branch2c2")
    del f["model_weights"]['bn4b8_branch2c']

    f["model_weights"]["bn4b8_branch2c2"].attrs["weight_names"] = b'bn4b8_branch2c2/gamma:0', b'bn4b8_branch2c2/beta:0', b'bn4b8_branch2c2/moving_mean:0', b'bn4b8_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b8", "/model_weights/res4b82")
    del f["model_weights"]['res4b8']

    f.copy("/model_weights/res4b8_relu", "/model_weights/res4b8_relu2")
    del f["model_weights"]['res4b8_relu']

    f.copy("/model_weights/res4b9_branch2a/res4b9_branch2a", "/model_weights/res4b9_branch2a2/res4b9_branch2a2")
    del f["model_weights"]['res4b9_branch2a']

    f["model_weights"]["res4b9_branch2a2"].attrs["weight_names"] = [b'res4b9_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b9_branch2a/bn4b9_branch2a", "/model_weights/bn4b9_branch2a2/bn4b9_branch2a2")
    del f["model_weights"]['bn4b9_branch2a']

    f["model_weights"]["bn4b9_branch2a2"].attrs["weight_names"] = b'bn4b9_branch2a2/gamma:0', b'bn4b9_branch2a2/beta:0', b'bn4b9_branch2a2/moving_mean:0', b'bn4b9_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b9_branch2a_relu", "/model_weights/res4b9_branch2a_relu2")
    del f["model_weights"]['res4b9_branch2a_relu']

    f.copy("/model_weights/padding4b9_branch2b", "/model_weights/padding4b9_branch2b2")
    del f["model_weights"]['padding4b9_branch2b']

    f.copy("/model_weights/res4b9_branch2b/res4b9_branch2b", "/model_weights/res4b9_branch2b2/res4b9_branch2b2")
    del f["model_weights"]['res4b9_branch2b']

    f["model_weights"]["res4b9_branch2b2"].attrs["weight_names"] = [b'res4b9_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b9_branch2b/bn4b9_branch2b", "/model_weights/bn4b9_branch2b2/bn4b9_branch2b2")
    del f["model_weights"]['bn4b9_branch2b']

    f["model_weights"]["bn4b9_branch2b2"].attrs["weight_names"] = b'bn4b9_branch2b2/gamma:0', b'bn4b9_branch2b2/beta:0', b'bn4b9_branch2b2/moving_mean:0', b'bn4b9_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b9_branch2b_relu", "/model_weights/res4b9_branch2b_relu2")
    del f["model_weights"]['res4b9_branch2b_relu']

    f.copy("/model_weights/res4b9_branch2c/res4b9_branch2c", "/model_weights/res4b9_branch2c2/res4b9_branch2c2")
    del f["model_weights"]['res4b9_branch2c']

    f["model_weights"]["res4b9_branch2c2"].attrs["weight_names"] = [b'res4b9_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b9_branch2c/bn4b9_branch2c", "/model_weights/bn4b9_branch2c2/bn4b9_branch2c2")
    del f["model_weights"]['bn4b9_branch2c']

    f["model_weights"]["bn4b9_branch2c2"].attrs["weight_names"] = b'bn4b9_branch2c2/gamma:0', b'bn4b9_branch2c2/beta:0', b'bn4b9_branch2c2/moving_mean:0', b'bn4b9_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b9", "/model_weights/res4b92")
    del f["model_weights"]['res4b9']

    f.copy("/model_weights/res4b9_relu", "/model_weights/res4b9_relu2")
    del f["model_weights"]['res4b9_relu']

    f.copy("/model_weights/res4b10_branch2a/res4b10_branch2a", "/model_weights/res4b10_branch2a2/res4b10_branch2a2")
    del f["model_weights"]['res4b10_branch2a']

    f["model_weights"]["res4b10_branch2a2"].attrs["weight_names"] = [b'res4b10_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b10_branch2a/bn4b10_branch2a", "/model_weights/bn4b10_branch2a2/bn4b10_branch2a2")
    del f["model_weights"]['bn4b10_branch2a']

    f["model_weights"]["bn4b10_branch2a2"].attrs["weight_names"] = b'bn4b10_branch2a2/gamma:0', b'bn4b10_branch2a2/beta:0', b'bn4b10_branch2a2/moving_mean:0', b'bn4b10_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b10_branch2a_relu", "/model_weights/res4b10_branch2a_relu2")
    del f["model_weights"]['res4b10_branch2a_relu']

    f.copy("/model_weights/padding4b10_branch2b", "/model_weights/padding4b10_branch2b2")
    del f["model_weights"]['padding4b10_branch2b']

    f.copy("/model_weights/res4b10_branch2b/res4b10_branch2b", "/model_weights/res4b10_branch2b2/res4b10_branch2b2")
    del f["model_weights"]['res4b10_branch2b']

    f["model_weights"]["res4b10_branch2b2"].attrs["weight_names"] = [b'res4b10_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b10_branch2b/bn4b10_branch2b", "/model_weights/bn4b10_branch2b2/bn4b10_branch2b2")
    del f["model_weights"]['bn4b10_branch2b']

    f["model_weights"]["bn4b10_branch2b2"].attrs["weight_names"] = b'bn4b10_branch2b2/gamma:0', b'bn4b10_branch2b2/beta:0', b'bn4b10_branch2b2/moving_mean:0', b'bn4b10_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b10_branch2b_relu", "/model_weights/res4b10_branch2b_relu2")
    del f["model_weights"]['res4b10_branch2b_relu']

    f.copy("/model_weights/res4b10_branch2c/res4b10_branch2c", "/model_weights/res4b10_branch2c2/res4b10_branch2c2")
    del f["model_weights"]['res4b10_branch2c']

    f["model_weights"]["res4b10_branch2c2"].attrs["weight_names"] = [b'res4b10_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b10_branch2c/bn4b10_branch2c", "/model_weights/bn4b10_branch2c2/bn4b10_branch2c2")
    del f["model_weights"]['bn4b10_branch2c']

    f["model_weights"]["bn4b10_branch2c2"].attrs["weight_names"] = b'bn4b10_branch2c2/gamma:0', b'bn4b10_branch2c2/beta:0', b'bn4b10_branch2c2/moving_mean:0', b'bn4b10_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b10", "/model_weights/res4b102")
    del f["model_weights"]['res4b10']

    f.copy("/model_weights/res4b10_relu", "/model_weights/res4b10_relu2")
    del f["model_weights"]['res4b10_relu']

    f.copy("/model_weights/res4b11_branch2a/res4b11_branch2a", "/model_weights/res4b11_branch2a2/res4b11_branch2a2")
    del f["model_weights"]['res4b11_branch2a']

    f["model_weights"]["res4b11_branch2a2"].attrs["weight_names"] = [b'res4b11_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b11_branch2a/bn4b11_branch2a", "/model_weights/bn4b11_branch2a2/bn4b11_branch2a2")
    del f["model_weights"]['bn4b11_branch2a']

    f["model_weights"]["bn4b11_branch2a2"].attrs["weight_names"] = b'bn4b11_branch2a2/gamma:0', b'bn4b11_branch2a2/beta:0', b'bn4b11_branch2a2/moving_mean:0', b'bn4b11_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b11_branch2a_relu", "/model_weights/res4b11_branch2a_relu2")
    del f["model_weights"]['res4b11_branch2a_relu']

    f.copy("/model_weights/padding4b11_branch2b", "/model_weights/padding4b11_branch2b2")
    del f["model_weights"]['padding4b11_branch2b']

    f.copy("/model_weights/res4b11_branch2b/res4b11_branch2b", "/model_weights/res4b11_branch2b2/res4b11_branch2b2")
    del f["model_weights"]['res4b11_branch2b']

    f["model_weights"]["res4b11_branch2b2"].attrs["weight_names"] = [b'res4b11_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b11_branch2b/bn4b11_branch2b", "/model_weights/bn4b11_branch2b2/bn4b11_branch2b2")
    del f["model_weights"]['bn4b11_branch2b']

    f["model_weights"]["bn4b11_branch2b2"].attrs["weight_names"] = b'bn4b11_branch2b2/gamma:0', b'bn4b11_branch2b2/beta:0', b'bn4b11_branch2b2/moving_mean:0', b'bn4b11_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b11_branch2b_relu", "/model_weights/res4b11_branch2b_relu2")
    del f["model_weights"]['res4b11_branch2b_relu']

    f.copy("/model_weights/res4b11_branch2c/res4b11_branch2c", "/model_weights/res4b11_branch2c2/res4b11_branch2c2")
    del f["model_weights"]['res4b11_branch2c']

    f["model_weights"]["res4b11_branch2c2"].attrs["weight_names"] = [b'res4b11_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b11_branch2c/bn4b11_branch2c", "/model_weights/bn4b11_branch2c2/bn4b11_branch2c2")
    del f["model_weights"]['bn4b11_branch2c']

    f["model_weights"]["bn4b11_branch2c2"].attrs["weight_names"] = b'bn4b11_branch2c2/gamma:0', b'bn4b11_branch2c2/beta:0', b'bn4b11_branch2c2/moving_mean:0', b'bn4b11_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b11", "/model_weights/res4b112")
    del f["model_weights"]['res4b11']

    f.copy("/model_weights/res4b11_relu", "/model_weights/res4b11_relu2")
    del f["model_weights"]['res4b11_relu']

    f.copy("/model_weights/res4b12_branch2a/res4b12_branch2a", "/model_weights/res4b12_branch2a2/res4b12_branch2a2")
    del f["model_weights"]['res4b12_branch2a']

    f["model_weights"]["res4b12_branch2a2"].attrs["weight_names"] = [b'res4b12_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b12_branch2a/bn4b12_branch2a", "/model_weights/bn4b12_branch2a2/bn4b12_branch2a2")
    del f["model_weights"]['bn4b12_branch2a']

    f["model_weights"]["bn4b12_branch2a2"].attrs["weight_names"] = b'bn4b12_branch2a2/gamma:0', b'bn4b12_branch2a2/beta:0', b'bn4b12_branch2a2/moving_mean:0', b'bn4b12_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b12_branch2a_relu", "/model_weights/res4b12_branch2a_relu2")
    del f["model_weights"]['res4b12_branch2a_relu']

    f.copy("/model_weights/padding4b12_branch2b", "/model_weights/padding4b12_branch2b2")
    del f["model_weights"]['padding4b12_branch2b']

    f.copy("/model_weights/res4b12_branch2b/res4b12_branch2b", "/model_weights/res4b12_branch2b2/res4b12_branch2b2")
    del f["model_weights"]['res4b12_branch2b']

    f["model_weights"]["res4b12_branch2b2"].attrs["weight_names"] = [b'res4b12_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b12_branch2b/bn4b12_branch2b", "/model_weights/bn4b12_branch2b2/bn4b12_branch2b2")
    del f["model_weights"]['bn4b12_branch2b']

    f["model_weights"]["bn4b12_branch2b2"].attrs["weight_names"] = b'bn4b12_branch2b2/gamma:0', b'bn4b12_branch2b2/beta:0', b'bn4b12_branch2b2/moving_mean:0', b'bn4b12_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b12_branch2b_relu", "/model_weights/res4b12_branch2b_relu2")
    del f["model_weights"]['res4b12_branch2b_relu']

    f.copy("/model_weights/res4b12_branch2c/res4b12_branch2c", "/model_weights/res4b12_branch2c2/res4b12_branch2c2")
    del f["model_weights"]['res4b12_branch2c']

    f["model_weights"]["res4b12_branch2c2"].attrs["weight_names"] = [b'res4b12_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b12_branch2c/bn4b12_branch2c", "/model_weights/bn4b12_branch2c2/bn4b12_branch2c2")
    del f["model_weights"]['bn4b12_branch2c']

    f["model_weights"]["bn4b12_branch2c2"].attrs["weight_names"] = b'bn4b12_branch2c2/gamma:0', b'bn4b12_branch2c2/beta:0', b'bn4b12_branch2c2/moving_mean:0', b'bn4b12_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b12_relu", "/model_weights/res4b12_relu2")
    del f["model_weights"]['res4b12_relu']

    f.copy("/model_weights/res4b13_branch2a/res4b13_branch2a", "/model_weights/res4b13_branch2a2/res4b13_branch2a2")
    del f["model_weights"]['res4b13_branch2a']

    f["model_weights"]["res4b13_branch2a2"].attrs["weight_names"] = [b'res4b13_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b13_branch2a/bn4b13_branch2a", "/model_weights/bn4b13_branch2a2/bn4b13_branch2a2")
    del f["model_weights"]['bn4b13_branch2a']

    f["model_weights"]["bn4b13_branch2a2"].attrs["weight_names"] = b'bn4b13_branch2a2/gamma:0', b'bn4b13_branch2a2/beta:0', b'bn4b13_branch2a2/moving_mean:0', b'bn4b13_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b13_branch2a_relu", "/model_weights/res4b13_branch2a_relu2")
    del f["model_weights"]['res4b13_branch2a_relu']

    f.copy("/model_weights/padding4b13_branch2b", "/model_weights/padding4b13_branch2b2")
    del f["model_weights"]['padding4b13_branch2b']

    f.copy("/model_weights/res4b13_branch2b/res4b13_branch2b", "/model_weights/res4b13_branch2b2/res4b13_branch2b2")
    del f["model_weights"]['res4b13_branch2b']

    f["model_weights"]["res4b13_branch2b2"].attrs["weight_names"] = [b'res4b13_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b13_branch2b/bn4b13_branch2b", "/model_weights/bn4b13_branch2b2/bn4b13_branch2b2")
    del f["model_weights"]['bn4b13_branch2b']

    f["model_weights"]["bn4b13_branch2b2"].attrs["weight_names"] = b'bn4b13_branch2b2/gamma:0', b'bn4b13_branch2b2/beta:0', b'bn4b13_branch2b2/moving_mean:0', b'bn4b13_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b13_branch2b_relu", "/model_weights/res4b13_branch2b_relu2")
    del f["model_weights"]['res4b13_branch2b_relu']

    f.copy("/model_weights/res4b13_branch2c/res4b13_branch2c", "/model_weights/res4b13_branch2c2/res4b13_branch2c2")
    del f["model_weights"]['res4b13_branch2c']

    f["model_weights"]["res4b13_branch2c2"].attrs["weight_names"] = [b'res4b13_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b13_branch2c/bn4b13_branch2c", "/model_weights/bn4b13_branch2c2/bn4b13_branch2c2")
    del f["model_weights"]['bn4b13_branch2c']

    f["model_weights"]["bn4b13_branch2c2"].attrs["weight_names"] = b'bn4b13_branch2c2/gamma:0', b'bn4b13_branch2c2/beta:0', b'bn4b13_branch2c2/moving_mean:0', b'bn4b13_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b13", "/model_weights/res4b132")
    del f["model_weights"]['res4b13']

    f.copy("/model_weights/res4b13_relu", "/model_weights/res4b13_relu2")
    del f["model_weights"]['res4b13_relu']

    f.copy("/model_weights/res4b14_branch2a/res4b14_branch2a", "/model_weights/res4b14_branch2a2/res4b14_branch2a2")
    del f["model_weights"]['res4b14_branch2a']

    f["model_weights"]["res4b14_branch2a2"].attrs["weight_names"] = [b'res4b14_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b14_branch2a/bn4b14_branch2a", "/model_weights/bn4b14_branch2a2/bn4b14_branch2a2")
    del f["model_weights"]['bn4b14_branch2a']

    f["model_weights"]["bn4b14_branch2a2"].attrs["weight_names"] = b'bn4b14_branch2a2/gamma:0', b'bn4b14_branch2a2/beta:0', b'bn4b14_branch2a2/moving_mean:0', b'bn4b14_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b14_branch2a_relu", "/model_weights/res4b14_branch2a_relu2")
    del f["model_weights"]['res4b14_branch2a_relu']

    f.copy("/model_weights/padding4b14_branch2b", "/model_weights/padding4b14_branch2b2")
    del f["model_weights"]['padding4b14_branch2b']

    f.copy("/model_weights/res4b14_branch2b/res4b14_branch2b", "/model_weights/res4b14_branch2b2/res4b14_branch2b2")
    del f["model_weights"]['res4b14_branch2b']

    f["model_weights"]["res4b14_branch2b2"].attrs["weight_names"] = [b'res4b14_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b14_branch2b/bn4b14_branch2b", "/model_weights/bn4b14_branch2b2/bn4b14_branch2b2")
    del f["model_weights"]['bn4b14_branch2b']

    f["model_weights"]["bn4b14_branch2b2"].attrs["weight_names"] = b'bn4b14_branch2b2/gamma:0', b'bn4b14_branch2b2/beta:0', b'bn4b14_branch2b2/moving_mean:0', b'bn4b14_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b14_branch2b_relu", "/model_weights/res4b14_branch2b_relu2")
    del f["model_weights"]['res4b14_branch2b_relu']

    f.copy("/model_weights/res4b14_branch2c/res4b14_branch2c", "/model_weights/res4b14_branch2c2/res4b14_branch2c2")
    del f["model_weights"]['res4b14_branch2c']

    f["model_weights"]["res4b14_branch2c2"].attrs["weight_names"] = [b'res4b14_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b14_branch2c/bn4b14_branch2c", "/model_weights/bn4b14_branch2c2/bn4b14_branch2c2")
    del f["model_weights"]['bn4b14_branch2c']

    f["model_weights"]["bn4b14_branch2c2"].attrs["weight_names"] = b'bn4b14_branch2c2/gamma:0', b'bn4b14_branch2c2/beta:0', b'bn4b14_branch2c2/moving_mean:0', b'bn4b14_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b14", "/model_weights/res4b142")
    del f["model_weights"]['res4b14']

    f.copy("/model_weights/res4b14_relu", "/model_weights/res4b14_relu2")
    del f["model_weights"]['res4b14_relu']

    f.copy("/model_weights/res4b15_branch2a/res4b15_branch2a", "/model_weights/res4b15_branch2a2/res4b15_branch2a2")
    del f["model_weights"]['res4b15_branch2a']

    f["model_weights"]["res4b15_branch2a2"].attrs["weight_names"] = [b'res4b15_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b15_branch2a/bn4b15_branch2a", "/model_weights/bn4b15_branch2a2/bn4b15_branch2a2")
    del f["model_weights"]['bn4b15_branch2a']

    f["model_weights"]["bn4b15_branch2a2"].attrs["weight_names"] = b'bn4b15_branch2a2/gamma:0', b'bn4b15_branch2a2/beta:0', b'bn4b15_branch2a2/moving_mean:0', b'bn4b15_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b15_branch2a_relu", "/model_weights/res4b15_branch2a_relu2")
    del f["model_weights"]['res4b15_branch2a_relu']

    f.copy("/model_weights/padding4b15_branch2b", "/model_weights/padding4b15_branch2b2")
    del f["model_weights"]['padding4b15_branch2b']

    f.copy("/model_weights/res4b15_branch2b/res4b15_branch2b", "/model_weights/res4b15_branch2b2/res4b15_branch2b2")
    del f["model_weights"]['res4b15_branch2b']

    f["model_weights"]["res4b15_branch2b2"].attrs["weight_names"] = [b'res4b15_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b15_branch2b/bn4b15_branch2b", "/model_weights/bn4b15_branch2b2/bn4b15_branch2b2")
    del f["model_weights"]['bn4b15_branch2b']

    f["model_weights"]["bn4b15_branch2b2"].attrs["weight_names"] = b'bn4b15_branch2b2/gamma:0', b'bn4b15_branch2b2/beta:0', b'bn4b15_branch2b2/moving_mean:0', b'bn4b15_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b15_branch2b_relu", "/model_weights/res4b15_branch2b_relu2")
    del f["model_weights"]['res4b15_branch2b_relu']

    f.copy("/model_weights/res4b15_branch2c/res4b15_branch2c", "/model_weights/res4b15_branch2c2/res4b15_branch2c2")
    del f["model_weights"]['res4b15_branch2c']

    f["model_weights"]["res4b15_branch2c2"].attrs["weight_names"] = [b'res4b15_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b15_branch2c/bn4b15_branch2c", "/model_weights/bn4b15_branch2c2/bn4b15_branch2c2")
    del f["model_weights"]['bn4b15_branch2c']

    f["model_weights"]["bn4b15_branch2c2"].attrs["weight_names"] = b'bn4b15_branch2c2/gamma:0', b'bn4b15_branch2c2/beta:0', b'bn4b15_branch2c2/moving_mean:0', b'bn4b15_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b15", "/model_weights/res4b152")
    del f["model_weights"]['res4b15']

    f.copy("/model_weights/res4b15_relu", "/model_weights/res4b15_relu2")
    del f["model_weights"]['res4b15_relu']

    f.copy("/model_weights/res4b16_branch2a/res4b16_branch2a", "/model_weights/res4b16_branch2a2/res4b16_branch2a2")
    del f["model_weights"]['res4b16_branch2a']

    f["model_weights"]["res4b16_branch2a2"].attrs["weight_names"] = [b'res4b16_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b16_branch2a/bn4b16_branch2a", "/model_weights/bn4b16_branch2a2/bn4b16_branch2a2")
    del f["model_weights"]['bn4b16_branch2a']

    f["model_weights"]["bn4b16_branch2a2"].attrs["weight_names"] = b'bn4b16_branch2a2/gamma:0', b'bn4b16_branch2a2/beta:0', b'bn4b16_branch2a2/moving_mean:0', b'bn4b16_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b16_branch2a_relu", "/model_weights/res4b16_branch2a_relu2")
    del f["model_weights"]['res4b16_branch2a_relu']

    f.copy("/model_weights/padding4b16_branch2b", "/model_weights/padding4b16_branch2b2")
    del f["model_weights"]['padding4b16_branch2b']

    f.copy("/model_weights/res4b16_branch2b/res4b16_branch2b", "/model_weights/res4b16_branch2b2/res4b16_branch2b2")
    del f["model_weights"]['res4b16_branch2b']

    f["model_weights"]["res4b16_branch2b2"].attrs["weight_names"] = [b'res4b16_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b16_branch2b/bn4b16_branch2b", "/model_weights/bn4b16_branch2b2/bn4b16_branch2b2")
    del f["model_weights"]['bn4b16_branch2b']

    f["model_weights"]["bn4b16_branch2b2"].attrs["weight_names"] = b'bn4b16_branch2b2/gamma:0', b'bn4b16_branch2b2/beta:0', b'bn4b16_branch2b2/moving_mean:0', b'bn4b16_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b16_branch2b_relu", "/model_weights/res4b16_branch2b_relu2")
    del f["model_weights"]['res4b16_branch2b_relu']

    f.copy("/model_weights/res4b16_branch2c/res4b16_branch2c", "/model_weights/res4b16_branch2c2/res4b16_branch2c2")
    del f["model_weights"]['res4b16_branch2c']

    f["model_weights"]["res4b16_branch2c2"].attrs["weight_names"] = [b'res4b16_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b16_branch2c/bn4b16_branch2c", "/model_weights/bn4b16_branch2c2/bn4b16_branch2c2")
    del f["model_weights"]['bn4b16_branch2c']

    f["model_weights"]["bn4b16_branch2c2"].attrs["weight_names"] = b'bn4b16_branch2c2/gamma:0', b'bn4b16_branch2c2/beta:0', b'bn4b16_branch2c2/moving_mean:0', b'bn4b16_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b16", "/model_weights/res4b162")
    del f["model_weights"]['res4b16']

    f.copy("/model_weights/res4b16_relu", "/model_weights/res4b16_relu2")
    del f["model_weights"]['res4b16_relu']

    f.copy("/model_weights/res4b17_branch2a/res4b17_branch2a", "/model_weights/res4b17_branch2a2/res4b17_branch2a2")
    del f["model_weights"]['res4b17_branch2a']

    f["model_weights"]["res4b17_branch2a2"].attrs["weight_names"] = [b'res4b17_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b17_branch2a/bn4b17_branch2a", "/model_weights/bn4b17_branch2a2/bn4b17_branch2a2")
    del f["model_weights"]['bn4b17_branch2a']

    f["model_weights"]["bn4b17_branch2a2"].attrs["weight_names"] = b'bn4b17_branch2a2/gamma:0', b'bn4b17_branch2a2/beta:0', b'bn4b17_branch2a2/moving_mean:0', b'bn4b17_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b17_branch2a_relu", "/model_weights/res4b17_branch2a_relu2")
    del f["model_weights"]['res4b17_branch2a_relu']

    f.copy("/model_weights/padding4b17_branch2b", "/model_weights/padding4b17_branch2b2")
    del f["model_weights"]['padding4b17_branch2b']

    f.copy("/model_weights/res4b17_branch2b/res4b17_branch2b", "/model_weights/res4b17_branch2b2/res4b17_branch2b2")
    del f["model_weights"]['res4b17_branch2b']

    f["model_weights"]["res4b17_branch2b2"].attrs["weight_names"] = [b'res4b17_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b17_branch2b/bn4b17_branch2b", "/model_weights/bn4b17_branch2b2/bn4b17_branch2b2")
    del f["model_weights"]['bn4b17_branch2b']

    f["model_weights"]["bn4b17_branch2b2"].attrs["weight_names"] = b'bn4b17_branch2b2/gamma:0', b'bn4b17_branch2b2/beta:0', b'bn4b17_branch2b2/moving_mean:0', b'bn4b17_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b17_branch2b_relu", "/model_weights/res4b17_branch2b_relu2")
    del f["model_weights"]['res4b17_branch2b_relu']

    f.copy("/model_weights/res4b17_branch2c/res4b17_branch2c", "/model_weights/res4b17_branch2c2/res4b17_branch2c2")
    del f["model_weights"]['res4b17_branch2c']

    f["model_weights"]["res4b17_branch2c2"].attrs["weight_names"] = [b'res4b17_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b17_branch2c/bn4b17_branch2c", "/model_weights/bn4b17_branch2c2/bn4b17_branch2c2")
    del f["model_weights"]['bn4b17_branch2c']

    f["model_weights"]["bn4b17_branch2c2"].attrs["weight_names"] = b'bn4b17_branch2c2/gamma:0', b'bn4b17_branch2c2/beta:0', b'bn4b17_branch2c2/moving_mean:0', b'bn4b17_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b17", "/model_weights/res4b172")
    del f["model_weights"]['res4b17']
    
    f.copy("/model_weights/res4b17_relu", "/model_weights/res4b17_relu2")
    del f["model_weights"]['res4b17_relu']

    f.copy("/model_weights/res4b18_branch2a/res4b18_branch2a", "/model_weights/res4b18_branch2a2/res4b18_branch2a2")
    del f["model_weights"]['res4b18_branch2a']

    f["model_weights"]["res4b18_branch2a2"].attrs["weight_names"] = [b'res4b18_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b18_branch2a/bn4b18_branch2a", "/model_weights/bn4b18_branch2a2/bn4b18_branch2a2")
    del f["model_weights"]['bn4b18_branch2a']

    f["model_weights"]["bn4b18_branch2a2"].attrs["weight_names"] = b'bn4b18_branch2a2/gamma:0', b'bn4b18_branch2a2/beta:0', b'bn4b18_branch2a2/moving_mean:0', b'bn4b18_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b18_branch2a_relu", "/model_weights/res4b18_branch2a_relu2")
    del f["model_weights"]['res4b18_branch2a_relu']

    f.copy("/model_weights/padding4b18_branch2b", "/model_weights/padding4b18_branch2b2")
    del f["model_weights"]['padding4b18_branch2b']

    f.copy("/model_weights/res4b18_branch2b/res4b18_branch2b", "/model_weights/res4b18_branch2b2/res4b18_branch2b2")
    del f["model_weights"]['res4b18_branch2b']

    f["model_weights"]["res4b18_branch2b2"].attrs["weight_names"] = [b'res4b18_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b18_branch2b/bn4b18_branch2b", "/model_weights/bn4b18_branch2b2/bn4b18_branch2b2")
    del f["model_weights"]['bn4b18_branch2b']

    f["model_weights"]["bn4b18_branch2b2"].attrs["weight_names"] = b'bn4b18_branch2b2/gamma:0', b'bn4b18_branch2b2/beta:0', b'bn4b18_branch2b2/moving_mean:0', b'bn4b18_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b18_branch2b_relu", "/model_weights/res4b18_branch2b_relu2")
    del f["model_weights"]['res4b18_branch2b_relu']

    f.copy("/model_weights/res4b18_branch2c/res4b18_branch2c", "/model_weights/res4b18_branch2c2/res4b18_branch2c2")
    del f["model_weights"]['res4b18_branch2c']

    f["model_weights"]["res4b18_branch2c2"].attrs["weight_names"] = [b'res4b18_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b18_branch2c/bn4b18_branch2c", "/model_weights/bn4b18_branch2c2/bn4b18_branch2c2")
    del f["model_weights"]['bn4b18_branch2c']

    f["model_weights"]["bn4b18_branch2c2"].attrs["weight_names"] = b'bn4b18_branch2c2/gamma:0', b'bn4b18_branch2c2/beta:0', b'bn4b18_branch2c2/moving_mean:0', b'bn4b18_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b18", "/model_weights/res4b182")
    del f["model_weights"]['res4b18']

    f.copy("/model_weights/res4b18_relu", "/model_weights/res4b18_relu2")
    del f["model_weights"]['res4b18_relu']

    f.copy("/model_weights/res4b19_branch2a/res4b19_branch2a", "/model_weights/res4b19_branch2a2/res4b19_branch2a2")
    del f["model_weights"]['res4b19_branch2a']

    f["model_weights"]["res4b19_branch2a2"].attrs["weight_names"] = [b'res4b19_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b19_branch2a/bn4b19_branch2a", "/model_weights/bn4b19_branch2a2/bn4b19_branch2a2")
    del f["model_weights"]['bn4b19_branch2a']

    f["model_weights"]["bn4b19_branch2a2"].attrs["weight_names"] = b'bn4b19_branch2a2/gamma:0', b'bn4b19_branch2a2/beta:0', b'bn4b19_branch2a2/moving_mean:0', b'bn4b19_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b19_branch2a_relu", "/model_weights/res4b19_branch2a_relu2")
    del f["model_weights"]['res4b19_branch2a_relu']

    f.copy("/model_weights/padding4b19_branch2b", "/model_weights/padding4b19_branch2b2")
    del f["model_weights"]['padding4b19_branch2b']

    f.copy("/model_weights/res4b19_branch2b/res4b19_branch2b", "/model_weights/res4b19_branch2b2/res4b19_branch2b2")
    del f["model_weights"]['res4b19_branch2b']

    f["model_weights"]["res4b19_branch2b2"].attrs["weight_names"] = [b'res4b19_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b19_branch2b/bn4b19_branch2b", "/model_weights/bn4b19_branch2b2/bn4b19_branch2b2")
    del f["model_weights"]['bn4b19_branch2b']

    f["model_weights"]["bn4b19_branch2b2"].attrs["weight_names"] = b'bn4b19_branch2b2/gamma:0', b'bn4b19_branch2b2/beta:0', b'bn4b19_branch2b2/moving_mean:0', b'bn4b19_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b19_branch2b_relu", "/model_weights/res4b19_branch2b_relu2")
    del f["model_weights"]['res4b19_branch2b_relu']

    f.copy("/model_weights/res4b19_branch2c/res4b19_branch2c", "/model_weights/res4b19_branch2c2/res4b19_branch2c2")
    del f["model_weights"]['res4b19_branch2c']

    f["model_weights"]["res4b19_branch2c2"].attrs["weight_names"] = [b'res4b19_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b19_branch2c/bn4b19_branch2c", "/model_weights/bn4b19_branch2c2/bn4b19_branch2c2")
    del f["model_weights"]['bn4b19_branch2c']

    f["model_weights"]["bn4b19_branch2c2"].attrs["weight_names"] = b'bn4b19_branch2c2/gamma:0', b'bn4b19_branch2c2/beta:0', b'bn4b19_branch2c2/moving_mean:0', b'bn4b19_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b19", "/model_weights/res4b192")
    del f["model_weights"]['res4b19']

    f.copy("/model_weights/res4b19_relu", "/model_weights/res4b19_relu2")
    del f["model_weights"]['res4b19_relu']

    f.copy("/model_weights/res4b20_branch2a/res4b20_branch2a", "/model_weights/res4b20_branch2a2/res4b20_branch2a2")
    del f["model_weights"]['res4b20_branch2a']

    f["model_weights"]["res4b20_branch2a2"].attrs["weight_names"] = [b'res4b20_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b20_branch2a/bn4b20_branch2a", "/model_weights/bn4b20_branch2a2/bn4b20_branch2a2")
    del f["model_weights"]['bn4b20_branch2a']

    f["model_weights"]["bn4b20_branch2a2"].attrs["weight_names"] = b'bn4b20_branch2a2/gamma:0', b'bn4b20_branch2a2/beta:0', b'bn4b20_branch2a2/moving_mean:0', b'bn4b20_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b20_branch2a_relu", "/model_weights/res4b20_branch2a_relu2")
    del f["model_weights"]['res4b20_branch2a_relu']

    f.copy("/model_weights/padding4b20_branch2b", "/model_weights/padding4b20_branch2b2")
    del f["model_weights"]['padding4b20_branch2b']

    f.copy("/model_weights/res4b20_branch2b/res4b20_branch2b", "/model_weights/res4b20_branch2b2/res4b20_branch2b2")
    del f["model_weights"]['res4b20_branch2b']

    f["model_weights"]["res4b20_branch2b2"].attrs["weight_names"] = [b'res4b20_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b20_branch2b/bn4b20_branch2b", "/model_weights/bn4b20_branch2b2/bn4b20_branch2b2")
    del f["model_weights"]['bn4b20_branch2b']

    f["model_weights"]["bn4b20_branch2b2"].attrs["weight_names"] = b'bn4b20_branch2b2/gamma:0', b'bn4b20_branch2b2/beta:0', b'bn4b20_branch2b2/moving_mean:0', b'bn4b20_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b20_branch2b_relu", "/model_weights/res4b20_branch2b_relu2")
    del f["model_weights"]['res4b20_branch2b_relu']

    f.copy("/model_weights/res4b20_branch2c/res4b20_branch2c", "/model_weights/res4b20_branch2c2/res4b20_branch2c2")
    del f["model_weights"]['res4b20_branch2c']

    f["model_weights"]["res4b20_branch2c2"].attrs["weight_names"] = [b'res4b20_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b20_branch2c/bn4b20_branch2c", "/model_weights/bn4b20_branch2c2/bn4b20_branch2c2")
    del f["model_weights"]['bn4b20_branch2c']

    f["model_weights"]["bn4b20_branch2c2"].attrs["weight_names"] = b'bn4b20_branch2c2/gamma:0', b'bn4b20_branch2c2/beta:0', b'bn4b20_branch2c2/moving_mean:0', b'bn4b20_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b20", "/model_weights/res4b202")
    del f["model_weights"]['res4b20']

    f.copy("/model_weights/res4b20_relu", "/model_weights/res4b20_relu2")
    del f["model_weights"]['res4b20_relu']

    f.copy("/model_weights/res4b21_branch2a/res4b21_branch2a", "/model_weights/res4b21_branch2a2/res4b21_branch2a2")
    del f["model_weights"]['res4b21_branch2a']

    f["model_weights"]["res4b21_branch2a2"].attrs["weight_names"] = [b'res4b21_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b21_branch2a/bn4b21_branch2a", "/model_weights/bn4b21_branch2a2/bn4b21_branch2a2")
    del f["model_weights"]['bn4b21_branch2a']

    f["model_weights"]["bn4b21_branch2a2"].attrs["weight_names"] = b'bn4b21_branch2a2/gamma:0', b'bn4b21_branch2a2/beta:0', b'bn4b21_branch2a2/moving_mean:0', b'bn4b21_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b21_branch2a_relu", "/model_weights/res4b21_branch2a_relu2")
    del f["model_weights"]['res4b21_branch2a_relu']

    f.copy("/model_weights/padding4b21_branch2b", "/model_weights/padding4b21_branch2b2")
    del f["model_weights"]['padding4b21_branch2b']

    f.copy("/model_weights/res4b21_branch2b/res4b21_branch2b", "/model_weights/res4b21_branch2b2/res4b21_branch2b2")
    del f["model_weights"]['res4b21_branch2b']

    f["model_weights"]["res4b21_branch2b2"].attrs["weight_names"] = [b'res4b21_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b21_branch2b/bn4b21_branch2b", "/model_weights/bn4b21_branch2b2/bn4b21_branch2b2")
    del f["model_weights"]['bn4b21_branch2b']

    f["model_weights"]["bn4b21_branch2b2"].attrs["weight_names"] = b'bn4b21_branch2b2/gamma:0', b'bn4b21_branch2b2/beta:0', b'bn4b21_branch2b2/moving_mean:0', b'bn4b21_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b21_branch2b_relu", "/model_weights/res4b21_branch2b_relu2")
    del f["model_weights"]['res4b21_branch2b_relu']

    f.copy("/model_weights/res4b21_branch2c/res4b21_branch2c", "/model_weights/res4b21_branch2c2/res4b21_branch2c2")
    del f["model_weights"]['res4b21_branch2c']

    f["model_weights"]["res4b21_branch2c2"].attrs["weight_names"] = [b'res4b21_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b21_branch2c/bn4b21_branch2c", "/model_weights/bn4b21_branch2c2/bn4b21_branch2c2")
    del f["model_weights"]['bn4b21_branch2c']

    f["model_weights"]["bn4b21_branch2c2"].attrs["weight_names"] = b'bn4b21_branch2c2/gamma:0', b'bn4b21_branch2c2/beta:0', b'bn4b21_branch2c2/moving_mean:0', b'bn4b21_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b21", "/model_weights/res4b212")
    del f["model_weights"]['res4b21']

    f.copy("/model_weights/res4b21_relu", "/model_weights/res4b21_relu2")
    del f["model_weights"]['res4b21_relu']

    f.copy("/model_weights/res4b22_branch2a/res4b22_branch2a", "/model_weights/res4b22_branch2a2/res4b22_branch2a2")
    del f["model_weights"]['res4b22_branch2a']

    f["model_weights"]["res4b22_branch2a2"].attrs["weight_names"] = [b'res4b22_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b22_branch2a/bn4b22_branch2a", "/model_weights/bn4b22_branch2a2/bn4b22_branch2a2")
    del f["model_weights"]['bn4b22_branch2a']

    f["model_weights"]["bn4b22_branch2a2"].attrs["weight_names"] = b'bn4b22_branch2a2/gamma:0', b'bn4b22_branch2a2/beta:0', b'bn4b22_branch2a2/moving_mean:0', b'bn4b22_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b22_branch2a_relu", "/model_weights/res4b22_branch2a_relu2")
    del f["model_weights"]['res4b22_branch2a_relu']

    f.copy("/model_weights/padding4b22_branch2b", "/model_weights/padding4b22_branch2b2")
    del f["model_weights"]['padding4b22_branch2b']

    f.copy("/model_weights/res4b22_branch2b/res4b22_branch2b", "/model_weights/res4b22_branch2b2/res4b22_branch2b2")
    del f["model_weights"]['res4b22_branch2b']

    f["model_weights"]["res4b22_branch2b2"].attrs["weight_names"] = [b'res4b22_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b22_branch2b/bn4b22_branch2b", "/model_weights/bn4b22_branch2b2/bn4b22_branch2b2")
    del f["model_weights"]['bn4b22_branch2b']

    f["model_weights"]["bn4b22_branch2b2"].attrs["weight_names"] = b'bn4b22_branch2b2/gamma:0', b'bn4b22_branch2b2/beta:0', b'bn4b22_branch2b2/moving_mean:0', b'bn4b22_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b22_branch2b_relu", "/model_weights/res4b22_branch2b_relu2")
    del f["model_weights"]['res4b22_branch2b_relu']

    f.copy("/model_weights/res4b22_branch2c/res4b22_branch2c", "/model_weights/res4b22_branch2c2/res4b22_branch2c2")
    del f["model_weights"]['res4b22_branch2c']

    f["model_weights"]["res4b22_branch2c2"].attrs["weight_names"] = [b'res4b22_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b22_branch2c/bn4b22_branch2c", "/model_weights/bn4b22_branch2c2/bn4b22_branch2c2")
    del f["model_weights"]['bn4b22_branch2c']

    f["model_weights"]["bn4b22_branch2c2"].attrs["weight_names"] = b'bn4b22_branch2c2/gamma:0', b'bn4b22_branch2c2/beta:0', b'bn4b22_branch2c2/moving_mean:0', b'bn4b22_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b22_relu", "/model_weights/res4b22_relu2")
    del f["model_weights"]['res4b22_relu']

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

    f["model_weights"]["regression_submodel2"].attrs["weight_names"] = b'pyramid_regression_0/kernel:0', b'pyramid_regression_0/bias:0', b'pyramid_regression_1/kernel:0', b'pyramid_regression_1/bias:0', b'pyramid_regression_2/kernel:0', b'pyramid_regression_2/bias:0', b'pyramid_regression_3/kernel:0', b'pyramid_regression_3/bias:0', b'pyramid_regression/kernel:0', b'pyramid_regression/bias:0'

    f.copy("/model_weights/regression", "/model_weights/regression2")
    del f["model_weights"]['regression']

    f.copy("/model_weights/classification_submodel", "/model_weights/classification_submodel2")
    del f["model_weights"]['classification_submodel']

    f.copy("/model_weights/classification", "/model_weights/classification2")
    del f["model_weights"]['classification']

    with open("/home/rblin/Documents/Image-Processing/Neural-networks/model_config101.txt", "r") as t:
        text = t.readlines()
        # f.attrs["model_config"] = text[-1][:-1].encode('utf-8')
        f.attrs["model_config"] = text[-1].encode('utf-8')

    with open("/home/rblin/Documents/Image-Processing/Neural-networks/training_config101.txt", "r") as t:
        text = t.readlines()
        # f.attrs["training_config"] = text[-1][:-1].encode('utf-8')
        f.attrs["training_config"] = text[-1].encode('utf-8')

    with open("/home/rblin/Documents/Image-Processing/Neural-networks/layer_names101.txt", "r") as t:
        text = t.readlines()
        # np_array = np.array(text[-1][:-1].split(',')).astype(np.bytes_)
        np_array = np.array(text[-1].split(',')).astype(np.bytes_)
        f["model_weights"].attrs["layer_names"] = np_array

    f.close()

def rename_resnet152(filepath):

    f = h5py.File(filepath, "a")

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

    f.copy("/model_weights/bn_conv1/bn_conv1", "/model_weights/bn_conv12/bn_conv12")
    del f["model_weights"]['bn_conv1']

    f["model_weights"]["bn_conv12"].attrs["weight_names"] = b'bn_conv12/gamma:0', b'bn_conv12/beta:0', b'bn_conv12/moving_mean:0', b'bn_conv12/moving_variance:0'

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

    f.copy("/model_weights/res3b1_branch2a/res3b1_branch2a", "/model_weights/res3b1_branch2a2/res3b1_branch2a2")
    del f["model_weights"]['res3b1_branch2a']

    f["model_weights"]["res3b1_branch2a2"].attrs["weight_names"] = [b'res3b1_branch2a2/kernel:0']

    f.copy("/model_weights/bn3b1_branch2a/bn3b1_branch2a", "/model_weights/bn3b1_branch2a2/bn3b1_branch2a2")
    del f["model_weights"]['bn3b1_branch2a']

    f["model_weights"]["bn3b1_branch2a2"].attrs["weight_names"] = b'bn3b1_branch2a2/gamma:0', b'bn3b1_branch2a2/beta:0', b'bn3b1_branch2a2/moving_mean:0', b'bn3b1_branch2a2/moving_variance:0'

    f.copy("/model_weights/res3b1_branch2a_relu", "/model_weights/res3b1_branch2a_relu2")
    del f["model_weights"]['res3b1_branch2a_relu']

    f.copy("/model_weights/padding3b1_branch2b", "/model_weights/padding3b1_branch2b2")
    del f["model_weights"]['padding3b1_branch2b']

    f.copy("/model_weights/res3b1_branch2b/res3b1_branch2b", "/model_weights/res3b1_branch2b2/res3b1_branch2b2")
    del f["model_weights"]['res3b1_branch2b']

    f["model_weights"]["res3b1_branch2b2"].attrs["weight_names"] = [b'res3b1_branch2b2/kernel:0']

    f.copy("/model_weights/bn3b1_branch2b/bn3b1_branch2b", "/model_weights/bn3b1_branch2b2/bn3b1_branch2b2")
    del f["model_weights"]['bn3b1_branch2b']

    f["model_weights"]["bn3b1_branch2b2"].attrs["weight_names"] = b'bn3b1_branch2b2/gamma:0', b'bn3b1_branch2b2/beta:0', b'bn3b1_branch2b2/moving_mean:0', b'bn3b1_branch2b2/moving_variance:0'

    f.copy("/model_weights/res3b1_branch2b_relu", "/model_weights/res3b1_branch2b_relu2")
    del f["model_weights"]['res3b1_branch2b_relu']

    f.copy("/model_weights/res3b1_branch2c/res3b1_branch2c", "/model_weights/res3b1_branch2c2/res3b1_branch2c2")
    del f["model_weights"]['res3b1_branch2c']

    f["model_weights"]["res3b1_branch2c2"].attrs["weight_names"] = [b'res3b1_branch2c2/kernel:0']

    f.copy("/model_weights/bn3b1_branch2c/bn3b1_branch2c", "/model_weights/bn3b1_branch2c2/bn3b1_branch2c2")
    del f["model_weights"]['bn3b1_branch2c']

    f["model_weights"]["bn3b1_branch2c2"].attrs["weight_names"] = b'bn3b1_branch2c2/gamma:0', b'bn3b1_branch2c2/beta:0', b'bn3b1_branch2c2/moving_mean:0', b'bn3b1_branch2c2/moving_variance:0'

    f.copy("/model_weights/res3b1", "/model_weights/res3b12")
    del f["model_weights"]['res3b1']

    f.copy("/model_weights/res3b1_relu", "/model_weights/res3b1_relu2")
    del f["model_weights"]['res3b1_relu']

    f.copy("/model_weights/res3b2_branch2a/res3b2_branch2a", "/model_weights/res3b2_branch2a2/res3b2_branch2a2")
    del f["model_weights"]['res3b2_branch2a']

    f["model_weights"]["res3b2_branch2a2"].attrs["weight_names"] = [b'res3b2_branch2a2/kernel:0']

    f.copy("/model_weights/bn3b2_branch2a/bn3b2_branch2a", "/model_weights/bn3b2_branch2a2/bn3b2_branch2a2")
    del f["model_weights"]['bn3b2_branch2a']

    f["model_weights"]["bn3b2_branch2a2"].attrs["weight_names"] = b'bn3b2_branch2a2/gamma:0', b'bn3b2_branch2a2/beta:0', b'bn3b2_branch2a2/moving_mean:0', b'bn3b2_branch2a2/moving_variance:0'

    f.copy("/model_weights/res3b2_branch2a_relu", "/model_weights/res3b2_branch2a_relu2")
    del f["model_weights"]['res3b2_branch2a_relu']

    f.copy("/model_weights/padding3b2_branch2b", "/model_weights/padding3b2_branch2b2")
    del f["model_weights"]['padding3b2_branch2b']

    f.copy("/model_weights/res3b2_branch2b/res3b2_branch2b", "/model_weights/res3b2_branch2b2/res3b2_branch2b2")
    del f["model_weights"]['res3b2_branch2b']

    f["model_weights"]["res3b2_branch2b2"].attrs["weight_names"] = [b'res3b2_branch2b2/kernel:0']

    f.copy("/model_weights/bn3b2_branch2b/bn3b2_branch2b", "/model_weights/bn3b2_branch2b2/bn3b2_branch2b2")
    del f["model_weights"]['bn3b2_branch2b']

    f["model_weights"]["bn3b2_branch2b2"].attrs["weight_names"] = b'bn3b2_branch2b2/gamma:0', b'bn3b2_branch2b2/beta:0', b'bn3b2_branch2b2/moving_mean:0', b'bn3b2_branch2b2/moving_variance:0'

    f.copy("/model_weights/res3b2_branch2b_relu", "/model_weights/res3b2_branch2b_relu2")
    del f["model_weights"]['res3b2_branch2b_relu']

    f.copy("/model_weights/res3b2_branch2c/res3b2_branch2c", "/model_weights/res3b2_branch2c2/res3b2_branch2c2")
    del f["model_weights"]['res3b2_branch2c']

    f["model_weights"]["res3b2_branch2c2"].attrs["weight_names"] = [b'res3b2_branch2c2/kernel:0']

    f.copy("/model_weights/bn3b2_branch2c/bn3b2_branch2c", "/model_weights/bn3b2_branch2c2/bn3b2_branch2c2")
    del f["model_weights"]['bn3b2_branch2c']

    f["model_weights"]["bn3b2_branch2c2"].attrs["weight_names"] = b'bn3b2_branch2c2/gamma:0', b'bn3b2_branch2c2/beta:0', b'bn3b2_branch2c2/moving_mean:0', b'bn3b2_branch2c2/moving_variance:0'

    f.copy("/model_weights/res3b2", "/model_weights/res3b22")
    del f["model_weights"]['res3b2']

    f.copy("/model_weights/res3b2_relu", "/model_weights/res3b2_relu2")
    del f["model_weights"]['res3b2_relu']

    f.copy("/model_weights/res3b3_branch2a/res3b3_branch2a", "/model_weights/res3b3_branch2a2/res3b3_branch2a2")
    del f["model_weights"]['res3b3_branch2a']

    f["model_weights"]["res3b3_branch2a2"].attrs["weight_names"] = [b'res3b3_branch2a2/kernel:0']

    f.copy("/model_weights/bn3b3_branch2a/bn3b3_branch2a", "/model_weights/bn3b3_branch2a2/bn3b3_branch2a2")
    del f["model_weights"]['bn3b3_branch2a']

    f["model_weights"]["bn3b3_branch2a2"].attrs["weight_names"] = b'bn3b3_branch2a2/gamma:0', b'bn3b3_branch2a2/beta:0', b'bn3b3_branch2a2/moving_mean:0', b'bn3b3_branch2a2/moving_variance:0'

    f.copy("/model_weights/res3b3_branch2a_relu", "/model_weights/res3b3_branch2a_relu2")
    del f["model_weights"]['res3b3_branch2a_relu']

    f.copy("/model_weights/padding3b3_branch2b", "/model_weights/padding3b3_branch2b2")
    del f["model_weights"]['padding3b3_branch2b']

    f.copy("/model_weights/res3b3_branch2b/res3b3_branch2b", "/model_weights/res3b3_branch2b2/res3b3_branch2b2")
    del f["model_weights"]['res3b3_branch2b']

    f["model_weights"]["res3b3_branch2b2"].attrs["weight_names"] = [b'res3b3_branch2b2/kernel:0']

    f.copy("/model_weights/bn3b3_branch2b/bn3b3_branch2b", "/model_weights/bn3b3_branch2b2/bn3b3_branch2b2")
    del f["model_weights"]['bn3b3_branch2b']

    f["model_weights"]["bn3b3_branch2b2"].attrs["weight_names"] = b'bn3b3_branch2b2/gamma:0', b'bn3b3_branch2b2/beta:0', b'bn3b3_branch2b2/moving_mean:0', b'bn3b3_branch2b2/moving_variance:0'

    f.copy("/model_weights/res3b3_branch2b_relu", "/model_weights/res3b3_branch2b_relu2")
    del f["model_weights"]['res3b3_branch2b_relu']

    f.copy("/model_weights/res3b3_branch2c/res3b3_branch2c", "/model_weights/res3b3_branch2c2/res3b3_branch2c2")
    del f["model_weights"]['res3b3_branch2c']

    f["model_weights"]["res3b3_branch2c2"].attrs["weight_names"] = [b'res3b3_branch2c2/kernel:0']

    f.copy("/model_weights/bn3b3_branch2c/bn3b3_branch2c", "/model_weights/bn3b3_branch2c2/bn3b3_branch2c2")
    del f["model_weights"]['bn3b3_branch2c']

    f["model_weights"]["bn3b3_branch2c2"].attrs["weight_names"] = b'bn3b3_branch2c2/gamma:0', b'bn3b3_branch2c2/beta:0', b'bn3b3_branch2c2/moving_mean:0', b'bn3b3_branch2c2/moving_variance:0'

    f.copy("/model_weights/res3b3", "/model_weights/res3b32")
    del f["model_weights"]['res3b3']

    f.copy("/model_weights/res3b3_relu", "/model_weights/res3b3_relu2")
    del f["model_weights"]['res3b3_relu']

    f.copy("/model_weights/res3b4_branch2a/res3b4_branch2a", "/model_weights/res3b4_branch2a2/res3b4_branch2a2")
    del f["model_weights"]['res3b4_branch2a']

    f["model_weights"]["res3b4_branch2a2"].attrs["weight_names"] = [b'res3b4_branch2a2/kernel:0']

    f.copy("/model_weights/bn3b4_branch2a/bn3b4_branch2a", "/model_weights/bn3b4_branch2a2/bn3b4_branch2a2")
    del f["model_weights"]['bn3b4_branch2a']

    f["model_weights"]["bn3b4_branch2a2"].attrs["weight_names"] = b'bn3b4_branch2a2/gamma:0', b'bn3b4_branch2a2/beta:0', b'bn3b4_branch2a2/moving_mean:0', b'bn3b4_branch2a2/moving_variance:0'

    f.copy("/model_weights/res3b4_branch2a_relu", "/model_weights/res3b4_branch2a_relu2")
    del f["model_weights"]['res3b4_branch2a_relu']

    f.copy("/model_weights/padding3b4_branch2b", "/model_weights/padding3b4_branch2b2")
    del f["model_weights"]['padding3b4_branch2b']

    f.copy("/model_weights/res3b4_branch2b/res3b4_branch2b", "/model_weights/res3b4_branch2b2/res3b4_branch2b2")
    del f["model_weights"]['res3b4_branch2b']

    f["model_weights"]["res3b4_branch2b2"].attrs["weight_names"] = [b'res3b4_branch2b2/kernel:0']

    f.copy("/model_weights/bn3b4_branch2b/bn3b4_branch2b", "/model_weights/bn3b4_branch2b2/bn3b4_branch2b2")
    del f["model_weights"]['bn3b4_branch2b']

    f["model_weights"]["bn3b4_branch2b2"].attrs["weight_names"] = b'bn3b4_branch2b2/gamma:0', b'bn3b4_branch2b2/beta:0', b'bn3b4_branch2b2/moving_mean:0', b'bn3b4_branch2b2/moving_variance:0'

    f.copy("/model_weights/res3b4_branch2b_relu", "/model_weights/res3b4_branch2b_relu2")
    del f["model_weights"]['res3b4_branch2b_relu']

    f.copy("/model_weights/res3b4_branch2c/res3b4_branch2c", "/model_weights/res3b4_branch2c2/res3b4_branch2c2")
    del f["model_weights"]['res3b4_branch2c']

    f["model_weights"]["res3b4_branch2c2"].attrs["weight_names"] = [b'res3b4_branch2c2/kernel:0']

    f.copy("/model_weights/bn3b4_branch2c/bn3b4_branch2c", "/model_weights/bn3b4_branch2c2/bn3b4_branch2c2")
    del f["model_weights"]['bn3b4_branch2c']

    f["model_weights"]["bn3b4_branch2c2"].attrs["weight_names"] = b'bn3b4_branch2c2/gamma:0', b'bn3b4_branch2c2/beta:0', b'bn3b4_branch2c2/moving_mean:0', b'bn3b4_branch2c2/moving_variance:0'

    f.copy("/model_weights/res3b4", "/model_weights/res3b42")
    del f["model_weights"]['res3b4']

    f.copy("/model_weights/res3b4_relu", "/model_weights/res3b4_relu2")
    del f["model_weights"]['res3b4_relu']

    f.copy("/model_weights/res3b5_branch2a/res3b5_branch2a", "/model_weights/res3b5_branch2a2/res3b5_branch2a2")
    del f["model_weights"]['res3b5_branch2a']

    f["model_weights"]["res3b5_branch2a2"].attrs["weight_names"] = [b'res3b5_branch2a2/kernel:0']

    f.copy("/model_weights/bn3b5_branch2a/bn3b5_branch2a", "/model_weights/bn3b5_branch2a2/bn3b5_branch2a2")
    del f["model_weights"]['bn3b5_branch2a']

    f["model_weights"]["bn3b5_branch2a2"].attrs[
        "weight_names"] = b'bn3b5_branch2a2/gamma:0', b'bn3b5_branch2a2/beta:0', b'bn3b5_branch2a2/moving_mean:0', b'bn3b5_branch2a2/moving_variance:0'

    f.copy("/model_weights/res3b5_branch2a_relu", "/model_weights/res3b5_branch2a_relu2")
    del f["model_weights"]['res3b5_branch2a_relu']

    f.copy("/model_weights/padding3b5_branch2b", "/model_weights/padding3b5_branch2b2")
    del f["model_weights"]['padding3b5_branch2b']

    f.copy("/model_weights/res3b5_branch2b/res3b5_branch2b", "/model_weights/res3b5_branch2b2/res3b5_branch2b2")
    del f["model_weights"]['res3b5_branch2b']

    f["model_weights"]["res3b5_branch2b2"].attrs["weight_names"] = [b'res3b5_branch2b2/kernel:0']

    f.copy("/model_weights/bn3b5_branch2b/bn3b5_branch2b", "/model_weights/bn3b5_branch2b2/bn3b5_branch2b2")
    del f["model_weights"]['bn3b5_branch2b']

    f["model_weights"]["bn3b5_branch2b2"].attrs[
        "weight_names"] = b'bn3b5_branch2b2/gamma:0', b'bn3b5_branch2b2/beta:0', b'bn3b5_branch2b2/moving_mean:0', b'bn3b5_branch2b2/moving_variance:0'

    f.copy("/model_weights/res3b5_branch2b_relu", "/model_weights/res3b5_branch2b_relu2")
    del f["model_weights"]['res3b5_branch2b_relu']

    f.copy("/model_weights/res3b5_branch2c/res3b5_branch2c", "/model_weights/res3b5_branch2c2/res3b5_branch2c2")
    del f["model_weights"]['res3b5_branch2c']

    f["model_weights"]["res3b5_branch2c2"].attrs["weight_names"] = [b'res3b5_branch2c2/kernel:0']

    f.copy("/model_weights/bn3b5_branch2c/bn3b5_branch2c", "/model_weights/bn3b5_branch2c2/bn3b5_branch2c2")
    del f["model_weights"]['bn3b5_branch2c']

    f["model_weights"]["bn3b5_branch2c2"].attrs[
        "weight_names"] = b'bn3b5_branch2c2/gamma:0', b'bn3b5_branch2c2/beta:0', b'bn3b5_branch2c2/moving_mean:0', b'bn3b5_branch2c2/moving_variance:0'

    f.copy("/model_weights/res3b5", "/model_weights/res3b52")
    del f["model_weights"]['res3b5']

    f.copy("/model_weights/res3b5_relu", "/model_weights/res3b5_relu2")
    del f["model_weights"]['res3b5_relu']

    f.copy("/model_weights/res3b6_branch2a/res3b6_branch2a", "/model_weights/res3b6_branch2a2/res3b6_branch2a2")
    del f["model_weights"]['res3b6_branch2a']

    f["model_weights"]["res3b6_branch2a2"].attrs["weight_names"] = [b'res3b6_branch2a2/kernel:0']

    f.copy("/model_weights/bn3b6_branch2a/bn3b6_branch2a", "/model_weights/bn3b6_branch2a2/bn3b6_branch2a2")
    del f["model_weights"]['bn3b6_branch2a']

    f["model_weights"]["bn3b6_branch2a2"].attrs[
        "weight_names"] = b'bn3b6_branch2a2/gamma:0', b'bn3b6_branch2a2/beta:0', b'bn3b6_branch2a2/moving_mean:0', b'bn3b6_branch2a2/moving_variance:0'

    f.copy("/model_weights/res3b6_branch2a_relu", "/model_weights/res3b6_branch2a_relu2")
    del f["model_weights"]['res3b6_branch2a_relu']

    f.copy("/model_weights/padding3b6_branch2b", "/model_weights/padding3b6_branch2b2")
    del f["model_weights"]['padding3b6_branch2b']

    f.copy("/model_weights/res3b6_branch2b/res3b6_branch2b", "/model_weights/res3b6_branch2b2/res3b6_branch2b2")
    del f["model_weights"]['res3b6_branch2b']

    f["model_weights"]["res3b6_branch2b2"].attrs["weight_names"] = [b'res3b6_branch2b2/kernel:0']

    f.copy("/model_weights/bn3b6_branch2b/bn3b6_branch2b", "/model_weights/bn3b6_branch2b2/bn3b6_branch2b2")
    del f["model_weights"]['bn3b6_branch2b']

    f["model_weights"]["bn3b6_branch2b2"].attrs[
        "weight_names"] = b'bn3b6_branch2b2/gamma:0', b'bn3b6_branch2b2/beta:0', b'bn3b6_branch2b2/moving_mean:0', b'bn3b6_branch2b2/moving_variance:0'

    f.copy("/model_weights/res3b6_branch2b_relu", "/model_weights/res3b6_branch2b_relu2")
    del f["model_weights"]['res3b6_branch2b_relu']

    f.copy("/model_weights/res3b6_branch2c/res3b6_branch2c", "/model_weights/res3b6_branch2c2/res3b6_branch2c2")
    del f["model_weights"]['res3b6_branch2c']

    f["model_weights"]["res3b6_branch2c2"].attrs["weight_names"] = [b'res3b6_branch2c2/kernel:0']

    f.copy("/model_weights/bn3b6_branch2c/bn3b6_branch2c", "/model_weights/bn3b6_branch2c2/bn3b6_branch2c2")
    del f["model_weights"]['bn3b6_branch2c']

    f["model_weights"]["bn3b6_branch2c2"].attrs[
        "weight_names"] = b'bn3b6_branch2c2/gamma:0', b'bn3b6_branch2c2/beta:0', b'bn3b6_branch2c2/moving_mean:0', b'bn3b6_branch2c2/moving_variance:0'

    f.copy("/model_weights/res3b6", "/model_weights/res3b62")
    del f["model_weights"]['res3b6']

    f.copy("/model_weights/res3b6_relu", "/model_weights/res3b6_relu2")
    del f["model_weights"]['res3b6_relu']

    f.copy("/model_weights/res3b7_branch2a/res3b7_branch2a", "/model_weights/res3b7_branch2a2/res3b7_branch2a2")
    del f["model_weights"]['res3b7_branch2a']

    f["model_weights"]["res3b7_branch2a2"].attrs["weight_names"] = [b'res3b7_branch2a2/kernel:0']

    f.copy("/model_weights/bn3b7_branch2a/bn3b7_branch2a", "/model_weights/bn3b7_branch2a2/bn3b7_branch2a2")
    del f["model_weights"]['bn3b7_branch2a']

    f["model_weights"]["bn3b7_branch2a2"].attrs[
        "weight_names"] = b'bn3b7_branch2a2/gamma:0', b'bn3b7_branch2a2/beta:0', b'bn3b7_branch2a2/moving_mean:0', b'bn3b7_branch2a2/moving_variance:0'

    f.copy("/model_weights/res3b7_branch2a_relu", "/model_weights/res3b7_branch2a_relu2")
    del f["model_weights"]['res3b7_branch2a_relu']

    f.copy("/model_weights/padding3b7_branch2b", "/model_weights/padding3b7_branch2b2")
    del f["model_weights"]['padding3b7_branch2b']

    f.copy("/model_weights/res3b7_branch2b/res3b7_branch2b", "/model_weights/res3b7_branch2b2/res3b7_branch2b2")
    del f["model_weights"]['res3b7_branch2b']

    f["model_weights"]["res3b7_branch2b2"].attrs["weight_names"] = [b'res3b7_branch2b2/kernel:0']

    f.copy("/model_weights/bn3b7_branch2b/bn3b7_branch2b", "/model_weights/bn3b7_branch2b2/bn3b7_branch2b2")
    del f["model_weights"]['bn3b7_branch2b']

    f["model_weights"]["bn3b7_branch2b2"].attrs[
        "weight_names"] = b'bn3b7_branch2b2/gamma:0', b'bn3b7_branch2b2/beta:0', b'bn3b7_branch2b2/moving_mean:0', b'bn3b7_branch2b2/moving_variance:0'

    f.copy("/model_weights/res3b7_branch2b_relu", "/model_weights/res3b7_branch2b_relu2")
    del f["model_weights"]['res3b7_branch2b_relu']

    f.copy("/model_weights/res3b7_branch2c/res3b7_branch2c", "/model_weights/res3b7_branch2c2/res3b7_branch2c2")
    del f["model_weights"]['res3b7_branch2c']

    f["model_weights"]["res3b7_branch2c2"].attrs["weight_names"] = [b'res3b7_branch2c2/kernel:0']

    f.copy("/model_weights/bn3b7_branch2c/bn3b7_branch2c", "/model_weights/bn3b7_branch2c2/bn3b7_branch2c2")
    del f["model_weights"]['bn3b7_branch2c']

    f["model_weights"]["bn3b7_branch2c2"].attrs[
        "weight_names"] = b'bn3b7_branch2c2/gamma:0', b'bn3b7_branch2c2/beta:0', b'bn3b7_branch2c2/moving_mean:0', b'bn3b7_branch2c2/moving_variance:0'

    f.copy("/model_weights/res3b7", "/model_weights/res3b72")
    del f["model_weights"]['res3b7']

    f.copy("/model_weights/res3b7_relu", "/model_weights/res3b7_relu2")
    del f["model_weights"]['res3b7_relu']

    f.copy("/model_weights/res4a_branch2a/res4a_branch2a", "/model_weights/res4a_branch2a2/res4a_branch2a2")
    del f["model_weights"]['res4a_branch2a']

    f["model_weights"]["res4a_branch2a2"].attrs["weight_names"] = [b'res4a_branch2a2/kernel:0']

    f.copy("/model_weights/bn4a_branch2a/bn4a_branch2a", "/model_weights/bn4a_branch2a2/bn4a_branch2a2")
    del f["model_weights"]['bn4a_branch2a']

    f["model_weights"]["bn4a_branch2a2"].attrs[
        "weight_names"] = b'bn4a_branch2a2/gamma:0', b'bn4a_branch2a2/beta:0', b'bn4a_branch2a2/moving_mean:0', b'bn4a_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4a_branch2a_relu", "/model_weights/res4a_branch2a_relu2")
    del f["model_weights"]['res4a_branch2a_relu']

    f.copy("/model_weights/padding4a_branch2b", "/model_weights/padding4a_branch2b2")
    del f["model_weights"]['padding4a_branch2b']

    f.copy("/model_weights/res4a_branch2b/res4a_branch2b", "/model_weights/res4a_branch2b2/res4a_branch2b2")
    del f["model_weights"]['res4a_branch2b']

    f["model_weights"]["res4a_branch2b2"].attrs["weight_names"] = [b'res4a_branch2b2/kernel:0']

    f.copy("/model_weights/bn4a_branch2b/bn4a_branch2b", "/model_weights/bn4a_branch2b2/bn4a_branch2b2")
    del f["model_weights"]['bn4a_branch2b']

    f["model_weights"]["bn4a_branch2b2"].attrs[
        "weight_names"] = b'bn4a_branch2b2/gamma:0', b'bn4a_branch2b2/beta:0', b'bn4a_branch2b2/moving_mean:0', b'bn4a_branch2b2/moving_variance:0'

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

    f["model_weights"]["bn4a_branch2c2"].attrs[
        "weight_names"] = b'bn4a_branch2c2/gamma:0', b'bn4a_branch2c2/beta:0', b'bn4a_branch2c2/moving_mean:0', b'bn4a_branch2c2/moving_variance:0'

    f.copy("/model_weights/bn4a_branch1/bn4a_branch1", "/model_weights/bn4a_branch12/bn4a_branch12")
    del f["model_weights"]['bn4a_branch1']

    f["model_weights"]["bn4a_branch12"].attrs[
        "weight_names"] = b'bn4a_branch12/gamma:0', b'bn4a_branch12/beta:0', b'bn4a_branch12/moving_mean:0', b'bn4a_branch12/moving_variance:0'

    f.copy("/model_weights/res4a", "/model_weights/res4a2")
    del f["model_weights"]['res4a']

    f.copy("/model_weights/res4a_relu", "/model_weights/res4a_relu2")
    del f["model_weights"]['res4a_relu']

    f.copy("/model_weights/res4b1_branch2a/res4b1_branch2a", "/model_weights/res4b1_branch2a2/res4b1_branch2a2")
    del f["model_weights"]['res4b1_branch2a']

    f["model_weights"]["res4b1_branch2a2"].attrs["weight_names"] = [b'res4b1_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b1_branch2a/bn4b1_branch2a", "/model_weights/bn4b1_branch2a2/bn4b1_branch2a2")
    del f["model_weights"]['bn4b1_branch2a']

    f["model_weights"]["bn4b1_branch2a2"].attrs[
        "weight_names"] = b'bn4b1_branch2a2/gamma:0', b'bn4b1_branch2a2/beta:0', b'bn4b1_branch2a2/moving_mean:0', b'bn4b1_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b1_branch2a_relu", "/model_weights/res4b1_branch2a_relu2")
    del f["model_weights"]['res4b1_branch2a_relu']

    f.copy("/model_weights/padding4b1_branch2b", "/model_weights/padding4b1_branch2b2")
    del f["model_weights"]['padding4b1_branch2b']

    f.copy("/model_weights/res4b1_branch2b/res4b1_branch2b", "/model_weights/res4b1_branch2b2/res4b1_branch2b2")
    del f["model_weights"]['res4b1_branch2b']

    f["model_weights"]["res4b1_branch2b2"].attrs["weight_names"] = [b'res4b1_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b1_branch2b/bn4b1_branch2b", "/model_weights/bn4b1_branch2b2/bn4b1_branch2b2")
    del f["model_weights"]['bn4b1_branch2b']

    f["model_weights"]["bn4b1_branch2b2"].attrs[
        "weight_names"] = b'bn4b1_branch2b2/gamma:0', b'bn4b1_branch2b2/beta:0', b'bn4b1_branch2b2/moving_mean:0', b'bn4b1_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b1_branch2b_relu", "/model_weights/res4b1_branch2b_relu2")
    del f["model_weights"]['res4b1_branch2b_relu']

    f.copy("/model_weights/res4b1_branch2c/res4b1_branch2c", "/model_weights/res4b1_branch2c2/res4b1_branch2c2")
    del f["model_weights"]['res4b1_branch2c']

    f["model_weights"]["res4b1_branch2c2"].attrs["weight_names"] = [b'res4b1_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b1_branch2c/bn4b1_branch2c", "/model_weights/bn4b1_branch2c2/bn4b1_branch2c2")
    del f["model_weights"]['bn4b1_branch2c']

    f["model_weights"]["bn4b1_branch2c2"].attrs[
        "weight_names"] = b'bn4b1_branch2c2/gamma:0', b'bn4b1_branch2c2/beta:0', b'bn4b1_branch2c2/moving_mean:0', b'bn4b1_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b12", "/model_weights/res4b1222")
    del f["model_weights"]['res4b12']

    f.copy("/model_weights/res4b1", "/model_weights/res4b122")
    del f["model_weights"]['res4b1']

    f.copy("/model_weights/res4b1_relu", "/model_weights/res4b1_relu2")
    del f["model_weights"]['res4b1_relu']

    f.copy("/model_weights/res4b2_branch2a/res4b2_branch2a", "/model_weights/res4b2_branch2a2/res4b2_branch2a2")
    del f["model_weights"]['res4b2_branch2a']

    f["model_weights"]["res4b2_branch2a2"].attrs["weight_names"] = [b'res4b2_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b2_branch2a/bn4b2_branch2a", "/model_weights/bn4b2_branch2a2/bn4b2_branch2a2")
    del f["model_weights"]['bn4b2_branch2a']

    f["model_weights"]["bn4b2_branch2a2"].attrs[
        "weight_names"] = b'bn4b2_branch2a2/gamma:0', b'bn4b2_branch2a2/beta:0', b'bn4b2_branch2a2/moving_mean:0', b'bn4b2_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b2_branch2a_relu", "/model_weights/res4b2_branch2a_relu2")
    del f["model_weights"]['res4b2_branch2a_relu']

    f.copy("/model_weights/padding4b2_branch2b", "/model_weights/padding4b2_branch2b2")
    del f["model_weights"]['padding4b2_branch2b']

    f.copy("/model_weights/res4b2_branch2b/res4b2_branch2b", "/model_weights/res4b2_branch2b2/res4b2_branch2b2")
    del f["model_weights"]['res4b2_branch2b']

    f["model_weights"]["res4b2_branch2b2"].attrs["weight_names"] = [b'res4b2_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b2_branch2b/bn4b2_branch2b", "/model_weights/bn4b2_branch2b2/bn4b2_branch2b2")
    del f["model_weights"]['bn4b2_branch2b']

    f["model_weights"]["bn4b2_branch2b2"].attrs[
        "weight_names"] = b'bn4b2_branch2b2/gamma:0', b'bn4b2_branch2b2/beta:0', b'bn4b2_branch2b2/moving_mean:0', b'bn4b2_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b2_branch2b_relu", "/model_weights/res4b2_branch2b_relu2")
    del f["model_weights"]['res4b2_branch2b_relu']

    f.copy("/model_weights/res4b2_branch2c/res4b2_branch2c", "/model_weights/res4b2_branch2c2/res4b2_branch2c2")
    del f["model_weights"]['res4b2_branch2c']

    f["model_weights"]["res4b2_branch2c2"].attrs["weight_names"] = [b'res4b2_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b2_branch2c/bn4b2_branch2c", "/model_weights/bn4b2_branch2c2/bn4b2_branch2c2")
    del f["model_weights"]['bn4b2_branch2c']

    f["model_weights"]["bn4b2_branch2c2"].attrs[
        "weight_names"] = b'bn4b2_branch2c2/gamma:0', b'bn4b2_branch2c2/beta:0', b'bn4b2_branch2c2/moving_mean:0', b'bn4b2_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b22", "/model_weights/res4b2222")
    del f["model_weights"]['res4b22']

    f.copy("/model_weights/res4b2", "/model_weights/res4b222")
    del f["model_weights"]['res4b2']

    f.copy("/model_weights/res4b2_relu", "/model_weights/res4b2_relu2")
    del f["model_weights"]['res4b2_relu']

    f.copy("/model_weights/res4b3_branch2a/res4b3_branch2a", "/model_weights/res4b3_branch2a2/res4b3_branch2a2")
    del f["model_weights"]['res4b3_branch2a']

    f["model_weights"]["res4b3_branch2a2"].attrs["weight_names"] = [b'res4b3_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b3_branch2a/bn4b3_branch2a", "/model_weights/bn4b3_branch2a2/bn4b3_branch2a2")
    del f["model_weights"]['bn4b3_branch2a']

    f["model_weights"]["bn4b3_branch2a2"].attrs[
        "weight_names"] = b'bn4b3_branch2a2/gamma:0', b'bn4b3_branch2a2/beta:0', b'bn4b3_branch2a2/moving_mean:0', b'bn4b3_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b3_branch2a_relu", "/model_weights/res4b3_branch2a_relu2")
    del f["model_weights"]['res4b3_branch2a_relu']

    f.copy("/model_weights/padding4b3_branch2b", "/model_weights/padding4b3_branch2b2")
    del f["model_weights"]['padding4b3_branch2b']

    f.copy("/model_weights/res4b3_branch2b/res4b3_branch2b", "/model_weights/res4b3_branch2b2/res4b3_branch2b2")
    del f["model_weights"]['res4b3_branch2b']

    f["model_weights"]["res4b3_branch2b2"].attrs["weight_names"] = [b'res4b3_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b3_branch2b/bn4b3_branch2b", "/model_weights/bn4b3_branch2b2/bn4b3_branch2b2")
    del f["model_weights"]['bn4b3_branch2b']

    f["model_weights"]["bn4b3_branch2b2"].attrs[
        "weight_names"] = b'bn4b3_branch2b2/gamma:0', b'bn4b3_branch2b2/beta:0', b'bn4b3_branch2b2/moving_mean:0', b'bn4b3_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b3_branch2b_relu", "/model_weights/res4b3_branch2b_relu2")
    del f["model_weights"]['res4b3_branch2b_relu']

    f.copy("/model_weights/res4b3_branch2c/res4b3_branch2c", "/model_weights/res4b3_branch2c2/res4b3_branch2c2")
    del f["model_weights"]['res4b3_branch2c']

    f["model_weights"]["res4b3_branch2c2"].attrs["weight_names"] = [b'res4b3_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b3_branch2c/bn4b3_branch2c", "/model_weights/bn4b3_branch2c2/bn4b3_branch2c2")
    del f["model_weights"]['bn4b3_branch2c']

    f["model_weights"]["bn4b3_branch2c2"].attrs[
        "weight_names"] = b'bn4b3_branch2c2/gamma:0', b'bn4b3_branch2c2/beta:0', b'bn4b3_branch2c2/moving_mean:0', b'bn4b3_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b32", "/model_weights/res4b3222")
    del f["model_weights"]['res4b32']

    f.copy("/model_weights/res4b3", "/model_weights/res4b322")
    del f["model_weights"]['res4b3']

    f.copy("/model_weights/res4b3_relu", "/model_weights/res4b3_relu2")
    del f["model_weights"]['res4b3_relu']

    f.copy("/model_weights/res4b4_branch2a/res4b4_branch2a", "/model_weights/res4b4_branch2a2/res4b4_branch2a2")
    del f["model_weights"]['res4b4_branch2a']

    f["model_weights"]["res4b4_branch2a2"].attrs["weight_names"] = [b'res4b4_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b4_branch2a/bn4b4_branch2a", "/model_weights/bn4b4_branch2a2/bn4b4_branch2a2")
    del f["model_weights"]['bn4b4_branch2a']

    f["model_weights"]["bn4b4_branch2a2"].attrs[
        "weight_names"] = b'bn4b4_branch2a2/gamma:0', b'bn4b4_branch2a2/beta:0', b'bn4b4_branch2a2/moving_mean:0', b'bn4b4_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b4_branch2a_relu", "/model_weights/res4b4_branch2a_relu2")
    del f["model_weights"]['res4b4_branch2a_relu']

    f.copy("/model_weights/padding4b4_branch2b", "/model_weights/padding4b4_branch2b2")
    del f["model_weights"]['padding4b4_branch2b']

    f.copy("/model_weights/res4b4_branch2b/res4b4_branch2b", "/model_weights/res4b4_branch2b2/res4b4_branch2b2")
    del f["model_weights"]['res4b4_branch2b']

    f["model_weights"]["res4b4_branch2b2"].attrs["weight_names"] = [b'res4b4_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b4_branch2b/bn4b4_branch2b", "/model_weights/bn4b4_branch2b2/bn4b4_branch2b2")
    del f["model_weights"]['bn4b4_branch2b']

    f["model_weights"]["bn4b4_branch2b2"].attrs[
        "weight_names"] = b'bn4b4_branch2b2/gamma:0', b'bn4b4_branch2b2/beta:0', b'bn4b4_branch2b2/moving_mean:0', b'bn4b4_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b4_branch2b_relu", "/model_weights/res4b4_branch2b_relu2")
    del f["model_weights"]['res4b4_branch2b_relu']

    f.copy("/model_weights/res4b4_branch2c/res4b4_branch2c", "/model_weights/res4b4_branch2c2/res4b4_branch2c2")
    del f["model_weights"]['res4b4_branch2c']

    f["model_weights"]["res4b4_branch2c2"].attrs["weight_names"] = [b'res4b4_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b4_branch2c/bn4b4_branch2c", "/model_weights/bn4b4_branch2c2/bn4b4_branch2c2")
    del f["model_weights"]['bn4b4_branch2c']

    f["model_weights"]["bn4b4_branch2c2"].attrs[
        "weight_names"] = b'bn4b4_branch2c2/gamma:0', b'bn4b4_branch2c2/beta:0', b'bn4b4_branch2c2/moving_mean:0', b'bn4b4_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b4", "/model_weights/res4b42")
    del f["model_weights"]['res4b4']

    f.copy("/model_weights/res4b4_relu", "/model_weights/res4b4_relu2")
    del f["model_weights"]['res4b4_relu']

    f.copy("/model_weights/res4b5_branch2a/res4b5_branch2a", "/model_weights/res4b5_branch2a2/res4b5_branch2a2")
    del f["model_weights"]['res4b5_branch2a']

    f["model_weights"]["res4b5_branch2a2"].attrs["weight_names"] = [b'res4b5_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b5_branch2a/bn4b5_branch2a", "/model_weights/bn4b5_branch2a2/bn4b5_branch2a2")
    del f["model_weights"]['bn4b5_branch2a']

    f["model_weights"]["bn4b5_branch2a2"].attrs[
        "weight_names"] = b'bn4b5_branch2a2/gamma:0', b'bn4b5_branch2a2/beta:0', b'bn4b5_branch2a2/moving_mean:0', b'bn4b5_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b5_branch2a_relu", "/model_weights/res4b5_branch2a_relu2")
    del f["model_weights"]['res4b5_branch2a_relu']

    f.copy("/model_weights/padding4b5_branch2b", "/model_weights/padding4b5_branch2b2")
    del f["model_weights"]['padding4b5_branch2b']

    f.copy("/model_weights/res4b5_branch2b/res4b5_branch2b", "/model_weights/res4b5_branch2b2/res4b5_branch2b2")
    del f["model_weights"]['res4b5_branch2b']

    f["model_weights"]["res4b5_branch2b2"].attrs["weight_names"] = [b'res4b5_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b5_branch2b/bn4b5_branch2b", "/model_weights/bn4b5_branch2b2/bn4b5_branch2b2")
    del f["model_weights"]['bn4b5_branch2b']

    f["model_weights"]["bn4b5_branch2b2"].attrs[
        "weight_names"] = b'bn4b5_branch2b2/gamma:0', b'bn4b5_branch2b2/beta:0', b'bn4b5_branch2b2/moving_mean:0', b'bn4b5_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b5_branch2b_relu", "/model_weights/res4b5_branch2b_relu2")
    del f["model_weights"]['res4b5_branch2b_relu']

    f.copy("/model_weights/res4b5_branch2c/res4b5_branch2c", "/model_weights/res4b5_branch2c2/res4b5_branch2c2")
    del f["model_weights"]['res4b5_branch2c']

    f["model_weights"]["res4b5_branch2c2"].attrs["weight_names"] = [b'res4b5_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b5_branch2c/bn4b5_branch2c", "/model_weights/bn4b5_branch2c2/bn4b5_branch2c2")
    del f["model_weights"]['bn4b5_branch2c']

    f["model_weights"]["bn4b5_branch2c2"].attrs[
        "weight_names"] = b'bn4b5_branch2c2/gamma:0', b'bn4b5_branch2c2/beta:0', b'bn4b5_branch2c2/moving_mean:0', b'bn4b5_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b5", "/model_weights/res4b52")
    del f["model_weights"]['res4b5']

    f.copy("/model_weights/res4b5_relu", "/model_weights/res4b5_relu2")
    del f["model_weights"]['res4b5_relu']

    f.copy("/model_weights/res4b6_branch2a/res4b6_branch2a", "/model_weights/res4b6_branch2a2/res4b6_branch2a2")
    del f["model_weights"]['res4b6_branch2a']

    f["model_weights"]["res4b6_branch2a2"].attrs["weight_names"] = [b'res4b6_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b6_branch2a/bn4b6_branch2a", "/model_weights/bn4b6_branch2a2/bn4b6_branch2a2")
    del f["model_weights"]['bn4b6_branch2a']

    f["model_weights"]["bn4b6_branch2a2"].attrs[
        "weight_names"] = b'bn4b6_branch2a2/gamma:0', b'bn4b6_branch2a2/beta:0', b'bn4b6_branch2a2/moving_mean:0', b'bn4b6_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b6_branch2a_relu", "/model_weights/res4b6_branch2a_relu2")
    del f["model_weights"]['res4b6_branch2a_relu']

    f.copy("/model_weights/padding4b6_branch2b", "/model_weights/padding4b6_branch2b2")
    del f["model_weights"]['padding4b6_branch2b']

    f.copy("/model_weights/res4b6_branch2b/res4b6_branch2b", "/model_weights/res4b6_branch2b2/res4b6_branch2b2")
    del f["model_weights"]['res4b6_branch2b']

    f["model_weights"]["res4b6_branch2b2"].attrs["weight_names"] = [b'res4b6_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b6_branch2b/bn4b6_branch2b", "/model_weights/bn4b6_branch2b2/bn4b6_branch2b2")
    del f["model_weights"]['bn4b6_branch2b']

    f["model_weights"]["bn4b6_branch2b2"].attrs[
        "weight_names"] = b'bn4b6_branch2b2/gamma:0', b'bn4b6_branch2b2/beta:0', b'bn4b6_branch2b2/moving_mean:0', b'bn4b6_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b6_branch2b_relu", "/model_weights/res4b6_branch2b_relu2")
    del f["model_weights"]['res4b6_branch2b_relu']

    f.copy("/model_weights/res4b6_branch2c/res4b6_branch2c", "/model_weights/res4b6_branch2c2/res4b6_branch2c2")
    del f["model_weights"]['res4b6_branch2c']

    f["model_weights"]["res4b6_branch2c2"].attrs["weight_names"] = [b'res4b6_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b6_branch2c/bn4b6_branch2c", "/model_weights/bn4b6_branch2c2/bn4b6_branch2c2")
    del f["model_weights"]['bn4b6_branch2c']

    f["model_weights"]["bn4b6_branch2c2"].attrs[
        "weight_names"] = b'bn4b6_branch2c2/gamma:0', b'bn4b6_branch2c2/beta:0', b'bn4b6_branch2c2/moving_mean:0', b'bn4b6_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b6", "/model_weights/res4b62")
    del f["model_weights"]['res4b6']

    f.copy("/model_weights/res4b6_relu", "/model_weights/res4b6_relu2")
    del f["model_weights"]['res4b6_relu']

    f.copy("/model_weights/res4b7_branch2a/res4b7_branch2a", "/model_weights/res4b7_branch2a2/res4b7_branch2a2")
    del f["model_weights"]['res4b7_branch2a']

    f["model_weights"]["res4b7_branch2a2"].attrs["weight_names"] = [b'res4b7_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b7_branch2a/bn4b7_branch2a", "/model_weights/bn4b7_branch2a2/bn4b7_branch2a2")
    del f["model_weights"]['bn4b7_branch2a']

    f["model_weights"]["bn4b7_branch2a2"].attrs[
        "weight_names"] = b'bn4b7_branch2a2/gamma:0', b'bn4b7_branch2a2/beta:0', b'bn4b7_branch2a2/moving_mean:0', b'bn4b7_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b7_branch2a_relu", "/model_weights/res4b7_branch2a_relu2")
    del f["model_weights"]['res4b7_branch2a_relu']

    f.copy("/model_weights/padding4b7_branch2b", "/model_weights/padding4b7_branch2b2")
    del f["model_weights"]['padding4b7_branch2b']

    f.copy("/model_weights/res4b7_branch2b/res4b7_branch2b", "/model_weights/res4b7_branch2b2/res4b7_branch2b2")
    del f["model_weights"]['res4b7_branch2b']

    f["model_weights"]["res4b7_branch2b2"].attrs["weight_names"] = [b'res4b7_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b7_branch2b/bn4b7_branch2b", "/model_weights/bn4b7_branch2b2/bn4b7_branch2b2")
    del f["model_weights"]['bn4b7_branch2b']

    f["model_weights"]["bn4b7_branch2b2"].attrs[
        "weight_names"] = b'bn4b7_branch2b2/gamma:0', b'bn4b7_branch2b2/beta:0', b'bn4b7_branch2b2/moving_mean:0', b'bn4b7_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b7_branch2b_relu", "/model_weights/res4b7_branch2b_relu2")
    del f["model_weights"]['res4b7_branch2b_relu']

    f.copy("/model_weights/res4b7_branch2c/res4b7_branch2c", "/model_weights/res4b7_branch2c2/res4b7_branch2c2")
    del f["model_weights"]['res4b7_branch2c']
    
    f["model_weights"]["res4b7_branch2c2"].attrs["weight_names"] = [b'res4b7_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b7_branch2c/bn4b7_branch2c", "/model_weights/bn4b7_branch2c2/bn4b7_branch2c2")
    del f["model_weights"]['bn4b7_branch2c']

    f["model_weights"]["bn4b7_branch2c2"].attrs[
        "weight_names"] = b'bn4b7_branch2c2/gamma:0', b'bn4b7_branch2c2/beta:0', b'bn4b7_branch2c2/moving_mean:0', b'bn4b7_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b7", "/model_weights/res4b72")
    del f["model_weights"]['res4b7']

    f.copy("/model_weights/res4b7_relu", "/model_weights/res4b7_relu2")
    del f["model_weights"]['res4b7_relu']

    f.copy("/model_weights/res4b8_branch2a/res4b8_branch2a", "/model_weights/res4b8_branch2a2/res4b8_branch2a2")
    del f["model_weights"]['res4b8_branch2a']

    f["model_weights"]["res4b8_branch2a2"].attrs["weight_names"] = [b'res4b8_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b8_branch2a/bn4b8_branch2a", "/model_weights/bn4b8_branch2a2/bn4b8_branch2a2")
    del f["model_weights"]['bn4b8_branch2a']

    f["model_weights"]["bn4b8_branch2a2"].attrs[
        "weight_names"] = b'bn4b8_branch2a2/gamma:0', b'bn4b8_branch2a2/beta:0', b'bn4b8_branch2a2/moving_mean:0', b'bn4b8_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b8_branch2a_relu", "/model_weights/res4b8_branch2a_relu2")
    del f["model_weights"]['res4b8_branch2a_relu']

    f.copy("/model_weights/padding4b8_branch2b", "/model_weights/padding4b8_branch2b2")
    del f["model_weights"]['padding4b8_branch2b']

    f.copy("/model_weights/res4b8_branch2b/res4b8_branch2b", "/model_weights/res4b8_branch2b2/res4b8_branch2b2")
    del f["model_weights"]['res4b8_branch2b']

    f["model_weights"]["res4b8_branch2b2"].attrs["weight_names"] = [b'res4b8_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b8_branch2b/bn4b8_branch2b", "/model_weights/bn4b8_branch2b2/bn4b8_branch2b2")
    del f["model_weights"]['bn4b8_branch2b']

    f["model_weights"]["bn4b8_branch2b2"].attrs[
        "weight_names"] = b'bn4b8_branch2b2/gamma:0', b'bn4b8_branch2b2/beta:0', b'bn4b8_branch2b2/moving_mean:0', b'bn4b8_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b8_branch2b_relu", "/model_weights/res4b8_branch2b_relu2")
    del f["model_weights"]['res4b8_branch2b_relu']

    f.copy("/model_weights/res4b8_branch2c/res4b8_branch2c", "/model_weights/res4b8_branch2c2/res4b8_branch2c2")
    del f["model_weights"]['res4b8_branch2c']

    f["model_weights"]["res4b8_branch2c2"].attrs["weight_names"] = [b'res4b8_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b8_branch2c/bn4b8_branch2c", "/model_weights/bn4b8_branch2c2/bn4b8_branch2c2")
    del f["model_weights"]['bn4b8_branch2c']

    f["model_weights"]["bn4b8_branch2c2"].attrs[
        "weight_names"] = b'bn4b8_branch2c2/gamma:0', b'bn4b8_branch2c2/beta:0', b'bn4b8_branch2c2/moving_mean:0', b'bn4b8_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b8", "/model_weights/res4b82")
    del f["model_weights"]['res4b8']

    f.copy("/model_weights/res4b8_relu", "/model_weights/res4b8_relu2")
    del f["model_weights"]['res4b8_relu']

    f.copy("/model_weights/res4b9_branch2a/res4b9_branch2a", "/model_weights/res4b9_branch2a2/res4b9_branch2a2")
    del f["model_weights"]['res4b9_branch2a']

    f["model_weights"]["res4b9_branch2a2"].attrs["weight_names"] = [b'res4b9_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b9_branch2a/bn4b9_branch2a", "/model_weights/bn4b9_branch2a2/bn4b9_branch2a2")
    del f["model_weights"]['bn4b9_branch2a']

    f["model_weights"]["bn4b9_branch2a2"].attrs[
        "weight_names"] = b'bn4b9_branch2a2/gamma:0', b'bn4b9_branch2a2/beta:0', b'bn4b9_branch2a2/moving_mean:0', b'bn4b9_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b9_branch2a_relu", "/model_weights/res4b9_branch2a_relu2")
    del f["model_weights"]['res4b9_branch2a_relu']

    f.copy("/model_weights/padding4b9_branch2b", "/model_weights/padding4b9_branch2b2")
    del f["model_weights"]['padding4b9_branch2b']

    f.copy("/model_weights/res4b9_branch2b/res4b9_branch2b", "/model_weights/res4b9_branch2b2/res4b9_branch2b2")
    del f["model_weights"]['res4b9_branch2b']

    f["model_weights"]["res4b9_branch2b2"].attrs["weight_names"] = [b'res4b9_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b9_branch2b/bn4b9_branch2b", "/model_weights/bn4b9_branch2b2/bn4b9_branch2b2")
    del f["model_weights"]['bn4b9_branch2b']

    f["model_weights"]["bn4b9_branch2b2"].attrs[
        "weight_names"] = b'bn4b9_branch2b2/gamma:0', b'bn4b9_branch2b2/beta:0', b'bn4b9_branch2b2/moving_mean:0', b'bn4b9_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b9_branch2b_relu", "/model_weights/res4b9_branch2b_relu2")
    del f["model_weights"]['res4b9_branch2b_relu']

    f.copy("/model_weights/res4b9_branch2c/res4b9_branch2c", "/model_weights/res4b9_branch2c2/res4b9_branch2c2")
    del f["model_weights"]['res4b9_branch2c']

    f["model_weights"]["res4b9_branch2c2"].attrs["weight_names"] = [b'res4b9_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b9_branch2c/bn4b9_branch2c", "/model_weights/bn4b9_branch2c2/bn4b9_branch2c2")
    del f["model_weights"]['bn4b9_branch2c']

    f["model_weights"]["bn4b9_branch2c2"].attrs[
        "weight_names"] = b'bn4b9_branch2c2/gamma:0', b'bn4b9_branch2c2/beta:0', b'bn4b9_branch2c2/moving_mean:0', b'bn4b9_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b9", "/model_weights/res4b92")
    del f["model_weights"]['res4b9']

    f.copy("/model_weights/res4b9_relu", "/model_weights/res4b9_relu2")
    del f["model_weights"]['res4b9_relu']

    f.copy("/model_weights/res4b10_branch2a/res4b10_branch2a", "/model_weights/res4b10_branch2a2/res4b10_branch2a2")
    del f["model_weights"]['res4b10_branch2a']

    f["model_weights"]["res4b10_branch2a2"].attrs["weight_names"] = [b'res4b10_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b10_branch2a/bn4b10_branch2a", "/model_weights/bn4b10_branch2a2/bn4b10_branch2a2")
    del f["model_weights"]['bn4b10_branch2a']

    f["model_weights"]["bn4b10_branch2a2"].attrs[
        "weight_names"] = b'bn4b10_branch2a2/gamma:0', b'bn4b10_branch2a2/beta:0', b'bn4b10_branch2a2/moving_mean:0', b'bn4b10_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b10_branch2a_relu", "/model_weights/res4b10_branch2a_relu2")
    del f["model_weights"]['res4b10_branch2a_relu']

    f.copy("/model_weights/padding4b10_branch2b", "/model_weights/padding4b10_branch2b2")
    del f["model_weights"]['padding4b10_branch2b']

    f.copy("/model_weights/res4b10_branch2b/res4b10_branch2b", "/model_weights/res4b10_branch2b2/res4b10_branch2b2")
    del f["model_weights"]['res4b10_branch2b']

    f["model_weights"]["res4b10_branch2b2"].attrs["weight_names"] = [b'res4b10_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b10_branch2b/bn4b10_branch2b", "/model_weights/bn4b10_branch2b2/bn4b10_branch2b2")
    del f["model_weights"]['bn4b10_branch2b']

    f["model_weights"]["bn4b10_branch2b2"].attrs[
        "weight_names"] = b'bn4b10_branch2b2/gamma:0', b'bn4b10_branch2b2/beta:0', b'bn4b10_branch2b2/moving_mean:0', b'bn4b10_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b10_branch2b_relu", "/model_weights/res4b10_branch2b_relu2")
    del f["model_weights"]['res4b10_branch2b_relu']

    f.copy("/model_weights/res4b10_branch2c/res4b10_branch2c", "/model_weights/res4b10_branch2c2/res4b10_branch2c2")
    del f["model_weights"]['res4b10_branch2c']

    f["model_weights"]["res4b10_branch2c2"].attrs["weight_names"] = [b'res4b10_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b10_branch2c/bn4b10_branch2c", "/model_weights/bn4b10_branch2c2/bn4b10_branch2c2")
    del f["model_weights"]['bn4b10_branch2c']

    f["model_weights"]["bn4b10_branch2c2"].attrs[
        "weight_names"] = b'bn4b10_branch2c2/gamma:0', b'bn4b10_branch2c2/beta:0', b'bn4b10_branch2c2/moving_mean:0', b'bn4b10_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b10", "/model_weights/res4b102")
    del f["model_weights"]['res4b10']

    f.copy("/model_weights/res4b10_relu", "/model_weights/res4b10_relu2")
    del f["model_weights"]['res4b10_relu']

    f.copy("/model_weights/res4b11_branch2a/res4b11_branch2a", "/model_weights/res4b11_branch2a2/res4b11_branch2a2")
    del f["model_weights"]['res4b11_branch2a']

    f["model_weights"]["res4b11_branch2a2"].attrs["weight_names"] = [b'res4b11_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b11_branch2a/bn4b11_branch2a", "/model_weights/bn4b11_branch2a2/bn4b11_branch2a2")
    del f["model_weights"]['bn4b11_branch2a']

    f["model_weights"]["bn4b11_branch2a2"].attrs[
        "weight_names"] = b'bn4b11_branch2a2/gamma:0', b'bn4b11_branch2a2/beta:0', b'bn4b11_branch2a2/moving_mean:0', b'bn4b11_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b11_branch2a_relu", "/model_weights/res4b11_branch2a_relu2")
    del f["model_weights"]['res4b11_branch2a_relu']

    f.copy("/model_weights/padding4b11_branch2b", "/model_weights/padding4b11_branch2b2")
    del f["model_weights"]['padding4b11_branch2b']

    f.copy("/model_weights/res4b11_branch2b/res4b11_branch2b", "/model_weights/res4b11_branch2b2/res4b11_branch2b2")
    del f["model_weights"]['res4b11_branch2b']

    f["model_weights"]["res4b11_branch2b2"].attrs["weight_names"] = [b'res4b11_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b11_branch2b/bn4b11_branch2b", "/model_weights/bn4b11_branch2b2/bn4b11_branch2b2")
    del f["model_weights"]['bn4b11_branch2b']

    f["model_weights"]["bn4b11_branch2b2"].attrs[
        "weight_names"] = b'bn4b11_branch2b2/gamma:0', b'bn4b11_branch2b2/beta:0', b'bn4b11_branch2b2/moving_mean:0', b'bn4b11_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b11_branch2b_relu", "/model_weights/res4b11_branch2b_relu2")
    del f["model_weights"]['res4b11_branch2b_relu']

    f.copy("/model_weights/res4b11_branch2c/res4b11_branch2c", "/model_weights/res4b11_branch2c2/res4b11_branch2c2")
    del f["model_weights"]['res4b11_branch2c']

    f["model_weights"]["res4b11_branch2c2"].attrs["weight_names"] = [b'res4b11_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b11_branch2c/bn4b11_branch2c", "/model_weights/bn4b11_branch2c2/bn4b11_branch2c2")
    del f["model_weights"]['bn4b11_branch2c']

    f["model_weights"]["bn4b11_branch2c2"].attrs[
        "weight_names"] = b'bn4b11_branch2c2/gamma:0', b'bn4b11_branch2c2/beta:0', b'bn4b11_branch2c2/moving_mean:0', b'bn4b11_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b11", "/model_weights/res4b112")
    del f["model_weights"]['res4b11']

    f.copy("/model_weights/res4b11_relu", "/model_weights/res4b11_relu2")
    del f["model_weights"]['res4b11_relu']

    f.copy("/model_weights/res4b12_branch2a/res4b12_branch2a", "/model_weights/res4b12_branch2a2/res4b12_branch2a2")
    del f["model_weights"]['res4b12_branch2a']

    f["model_weights"]["res4b12_branch2a2"].attrs["weight_names"] = [b'res4b12_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b12_branch2a/bn4b12_branch2a", "/model_weights/bn4b12_branch2a2/bn4b12_branch2a2")
    del f["model_weights"]['bn4b12_branch2a']

    f["model_weights"]["bn4b12_branch2a2"].attrs[
        "weight_names"] = b'bn4b12_branch2a2/gamma:0', b'bn4b12_branch2a2/beta:0', b'bn4b12_branch2a2/moving_mean:0', b'bn4b12_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b12_branch2a_relu", "/model_weights/res4b12_branch2a_relu2")
    del f["model_weights"]['res4b12_branch2a_relu']

    f.copy("/model_weights/padding4b12_branch2b", "/model_weights/padding4b12_branch2b2")
    del f["model_weights"]['padding4b12_branch2b']

    f.copy("/model_weights/res4b12_branch2b/res4b12_branch2b", "/model_weights/res4b12_branch2b2/res4b12_branch2b2")
    del f["model_weights"]['res4b12_branch2b']

    f["model_weights"]["res4b12_branch2b2"].attrs["weight_names"] = [b'res4b12_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b12_branch2b/bn4b12_branch2b", "/model_weights/bn4b12_branch2b2/bn4b12_branch2b2")
    del f["model_weights"]['bn4b12_branch2b']

    f["model_weights"]["bn4b12_branch2b2"].attrs[
        "weight_names"] = b'bn4b12_branch2b2/gamma:0', b'bn4b12_branch2b2/beta:0', b'bn4b12_branch2b2/moving_mean:0', b'bn4b12_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b12_branch2b_relu", "/model_weights/res4b12_branch2b_relu2")
    del f["model_weights"]['res4b12_branch2b_relu']

    f.copy("/model_weights/res4b12_branch2c/res4b12_branch2c", "/model_weights/res4b12_branch2c2/res4b12_branch2c2")
    del f["model_weights"]['res4b12_branch2c']

    f["model_weights"]["res4b12_branch2c2"].attrs["weight_names"] = [b'res4b12_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b12_branch2c/bn4b12_branch2c", "/model_weights/bn4b12_branch2c2/bn4b12_branch2c2")
    del f["model_weights"]['bn4b12_branch2c']

    f["model_weights"]["bn4b12_branch2c2"].attrs[
        "weight_names"] = b'bn4b12_branch2c2/gamma:0', b'bn4b12_branch2c2/beta:0', b'bn4b12_branch2c2/moving_mean:0', b'bn4b12_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b12_relu", "/model_weights/res4b12_relu2")
    del f["model_weights"]['res4b12_relu']

    f.copy("/model_weights/res4b13_branch2a/res4b13_branch2a", "/model_weights/res4b13_branch2a2/res4b13_branch2a2")
    del f["model_weights"]['res4b13_branch2a']

    f["model_weights"]["res4b13_branch2a2"].attrs["weight_names"] = [b'res4b13_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b13_branch2a/bn4b13_branch2a", "/model_weights/bn4b13_branch2a2/bn4b13_branch2a2")
    del f["model_weights"]['bn4b13_branch2a']

    f["model_weights"]["bn4b13_branch2a2"].attrs[
        "weight_names"] = b'bn4b13_branch2a2/gamma:0', b'bn4b13_branch2a2/beta:0', b'bn4b13_branch2a2/moving_mean:0', b'bn4b13_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b13_branch2a_relu", "/model_weights/res4b13_branch2a_relu2")
    del f["model_weights"]['res4b13_branch2a_relu']

    f.copy("/model_weights/padding4b13_branch2b", "/model_weights/padding4b13_branch2b2")
    del f["model_weights"]['padding4b13_branch2b']

    f.copy("/model_weights/res4b13_branch2b/res4b13_branch2b", "/model_weights/res4b13_branch2b2/res4b13_branch2b2")
    del f["model_weights"]['res4b13_branch2b']

    f["model_weights"]["res4b13_branch2b2"].attrs["weight_names"] = [b'res4b13_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b13_branch2b/bn4b13_branch2b", "/model_weights/bn4b13_branch2b2/bn4b13_branch2b2")
    del f["model_weights"]['bn4b13_branch2b']

    f["model_weights"]["bn4b13_branch2b2"].attrs[
        "weight_names"] = b'bn4b13_branch2b2/gamma:0', b'bn4b13_branch2b2/beta:0', b'bn4b13_branch2b2/moving_mean:0', b'bn4b13_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b13_branch2b_relu", "/model_weights/res4b13_branch2b_relu2")
    del f["model_weights"]['res4b13_branch2b_relu']

    f.copy("/model_weights/res4b13_branch2c/res4b13_branch2c", "/model_weights/res4b13_branch2c2/res4b13_branch2c2")
    del f["model_weights"]['res4b13_branch2c']

    f["model_weights"]["res4b13_branch2c2"].attrs["weight_names"] = [b'res4b13_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b13_branch2c/bn4b13_branch2c", "/model_weights/bn4b13_branch2c2/bn4b13_branch2c2")
    del f["model_weights"]['bn4b13_branch2c']

    f["model_weights"]["bn4b13_branch2c2"].attrs[
        "weight_names"] = b'bn4b13_branch2c2/gamma:0', b'bn4b13_branch2c2/beta:0', b'bn4b13_branch2c2/moving_mean:0', b'bn4b13_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b13", "/model_weights/res4b132")
    del f["model_weights"]['res4b13']

    f.copy("/model_weights/res4b13_relu", "/model_weights/res4b13_relu2")
    del f["model_weights"]['res4b13_relu']

    f.copy("/model_weights/res4b14_branch2a/res4b14_branch2a", "/model_weights/res4b14_branch2a2/res4b14_branch2a2")
    del f["model_weights"]['res4b14_branch2a']

    f["model_weights"]["res4b14_branch2a2"].attrs["weight_names"] = [b'res4b14_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b14_branch2a/bn4b14_branch2a", "/model_weights/bn4b14_branch2a2/bn4b14_branch2a2")
    del f["model_weights"]['bn4b14_branch2a']

    f["model_weights"]["bn4b14_branch2a2"].attrs[
        "weight_names"] = b'bn4b14_branch2a2/gamma:0', b'bn4b14_branch2a2/beta:0', b'bn4b14_branch2a2/moving_mean:0', b'bn4b14_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b14_branch2a_relu", "/model_weights/res4b14_branch2a_relu2")
    del f["model_weights"]['res4b14_branch2a_relu']

    f.copy("/model_weights/padding4b14_branch2b", "/model_weights/padding4b14_branch2b2")
    del f["model_weights"]['padding4b14_branch2b']

    f.copy("/model_weights/res4b14_branch2b/res4b14_branch2b", "/model_weights/res4b14_branch2b2/res4b14_branch2b2")
    del f["model_weights"]['res4b14_branch2b']

    f["model_weights"]["res4b14_branch2b2"].attrs["weight_names"] = [b'res4b14_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b14_branch2b/bn4b14_branch2b", "/model_weights/bn4b14_branch2b2/bn4b14_branch2b2")
    del f["model_weights"]['bn4b14_branch2b']

    f["model_weights"]["bn4b14_branch2b2"].attrs[
        "weight_names"] = b'bn4b14_branch2b2/gamma:0', b'bn4b14_branch2b2/beta:0', b'bn4b14_branch2b2/moving_mean:0', b'bn4b14_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b14_branch2b_relu", "/model_weights/res4b14_branch2b_relu2")
    del f["model_weights"]['res4b14_branch2b_relu']

    f.copy("/model_weights/res4b14_branch2c/res4b14_branch2c", "/model_weights/res4b14_branch2c2/res4b14_branch2c2")
    del f["model_weights"]['res4b14_branch2c']

    f["model_weights"]["res4b14_branch2c2"].attrs["weight_names"] = [b'res4b14_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b14_branch2c/bn4b14_branch2c", "/model_weights/bn4b14_branch2c2/bn4b14_branch2c2")
    del f["model_weights"]['bn4b14_branch2c']

    f["model_weights"]["bn4b14_branch2c2"].attrs[
        "weight_names"] = b'bn4b14_branch2c2/gamma:0', b'bn4b14_branch2c2/beta:0', b'bn4b14_branch2c2/moving_mean:0', b'bn4b14_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b14", "/model_weights/res4b142")
    del f["model_weights"]['res4b14']

    f.copy("/model_weights/res4b14_relu", "/model_weights/res4b14_relu2")
    del f["model_weights"]['res4b14_relu']

    f.copy("/model_weights/res4b15_branch2a/res4b15_branch2a", "/model_weights/res4b15_branch2a2/res4b15_branch2a2")
    del f["model_weights"]['res4b15_branch2a']

    f["model_weights"]["res4b15_branch2a2"].attrs["weight_names"] = [b'res4b15_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b15_branch2a/bn4b15_branch2a", "/model_weights/bn4b15_branch2a2/bn4b15_branch2a2")
    del f["model_weights"]['bn4b15_branch2a']

    f["model_weights"]["bn4b15_branch2a2"].attrs[
        "weight_names"] = b'bn4b15_branch2a2/gamma:0', b'bn4b15_branch2a2/beta:0', b'bn4b15_branch2a2/moving_mean:0', b'bn4b15_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b15_branch2a_relu", "/model_weights/res4b15_branch2a_relu2")
    del f["model_weights"]['res4b15_branch2a_relu']

    f.copy("/model_weights/padding4b15_branch2b", "/model_weights/padding4b15_branch2b2")
    del f["model_weights"]['padding4b15_branch2b']

    f.copy("/model_weights/res4b15_branch2b/res4b15_branch2b", "/model_weights/res4b15_branch2b2/res4b15_branch2b2")
    del f["model_weights"]['res4b15_branch2b']

    f["model_weights"]["res4b15_branch2b2"].attrs["weight_names"] = [b'res4b15_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b15_branch2b/bn4b15_branch2b", "/model_weights/bn4b15_branch2b2/bn4b15_branch2b2")
    del f["model_weights"]['bn4b15_branch2b']

    f["model_weights"]["bn4b15_branch2b2"].attrs[
        "weight_names"] = b'bn4b15_branch2b2/gamma:0', b'bn4b15_branch2b2/beta:0', b'bn4b15_branch2b2/moving_mean:0', b'bn4b15_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b15_branch2b_relu", "/model_weights/res4b15_branch2b_relu2")
    del f["model_weights"]['res4b15_branch2b_relu']

    f.copy("/model_weights/res4b15_branch2c/res4b15_branch2c", "/model_weights/res4b15_branch2c2/res4b15_branch2c2")
    del f["model_weights"]['res4b15_branch2c']

    f["model_weights"]["res4b15_branch2c2"].attrs["weight_names"] = [b'res4b15_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b15_branch2c/bn4b15_branch2c", "/model_weights/bn4b15_branch2c2/bn4b15_branch2c2")
    del f["model_weights"]['bn4b15_branch2c']

    f["model_weights"]["bn4b15_branch2c2"].attrs[
        "weight_names"] = b'bn4b15_branch2c2/gamma:0', b'bn4b15_branch2c2/beta:0', b'bn4b15_branch2c2/moving_mean:0', b'bn4b15_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b15", "/model_weights/res4b152")
    del f["model_weights"]['res4b15']

    f.copy("/model_weights/res4b15_relu", "/model_weights/res4b15_relu2")
    del f["model_weights"]['res4b15_relu']

    f.copy("/model_weights/res4b16_branch2a/res4b16_branch2a", "/model_weights/res4b16_branch2a2/res4b16_branch2a2")
    del f["model_weights"]['res4b16_branch2a']

    f["model_weights"]["res4b16_branch2a2"].attrs["weight_names"] = [b'res4b16_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b16_branch2a/bn4b16_branch2a", "/model_weights/bn4b16_branch2a2/bn4b16_branch2a2")
    del f["model_weights"]['bn4b16_branch2a']

    f["model_weights"]["bn4b16_branch2a2"].attrs[
        "weight_names"] = b'bn4b16_branch2a2/gamma:0', b'bn4b16_branch2a2/beta:0', b'bn4b16_branch2a2/moving_mean:0', b'bn4b16_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b16_branch2a_relu", "/model_weights/res4b16_branch2a_relu2")
    del f["model_weights"]['res4b16_branch2a_relu']

    f.copy("/model_weights/padding4b16_branch2b", "/model_weights/padding4b16_branch2b2")
    del f["model_weights"]['padding4b16_branch2b']

    f.copy("/model_weights/res4b16_branch2b/res4b16_branch2b", "/model_weights/res4b16_branch2b2/res4b16_branch2b2")
    del f["model_weights"]['res4b16_branch2b']

    f["model_weights"]["res4b16_branch2b2"].attrs["weight_names"] = [b'res4b16_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b16_branch2b/bn4b16_branch2b", "/model_weights/bn4b16_branch2b2/bn4b16_branch2b2")
    del f["model_weights"]['bn4b16_branch2b']

    f["model_weights"]["bn4b16_branch2b2"].attrs[
        "weight_names"] = b'bn4b16_branch2b2/gamma:0', b'bn4b16_branch2b2/beta:0', b'bn4b16_branch2b2/moving_mean:0', b'bn4b16_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b16_branch2b_relu", "/model_weights/res4b16_branch2b_relu2")
    del f["model_weights"]['res4b16_branch2b_relu']

    f.copy("/model_weights/res4b16_branch2c/res4b16_branch2c", "/model_weights/res4b16_branch2c2/res4b16_branch2c2")
    del f["model_weights"]['res4b16_branch2c']

    f["model_weights"]["res4b16_branch2c2"].attrs["weight_names"] = [b'res4b16_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b16_branch2c/bn4b16_branch2c", "/model_weights/bn4b16_branch2c2/bn4b16_branch2c2")
    del f["model_weights"]['bn4b16_branch2c']

    f["model_weights"]["bn4b16_branch2c2"].attrs[
        "weight_names"] = b'bn4b16_branch2c2/gamma:0', b'bn4b16_branch2c2/beta:0', b'bn4b16_branch2c2/moving_mean:0', b'bn4b16_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b16", "/model_weights/res4b162")
    del f["model_weights"]['res4b16']

    f.copy("/model_weights/res4b16_relu", "/model_weights/res4b16_relu2")
    del f["model_weights"]['res4b16_relu']

    f.copy("/model_weights/res4b17_branch2a/res4b17_branch2a", "/model_weights/res4b17_branch2a2/res4b17_branch2a2")
    del f["model_weights"]['res4b17_branch2a']

    f["model_weights"]["res4b17_branch2a2"].attrs["weight_names"] = [b'res4b17_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b17_branch2a/bn4b17_branch2a", "/model_weights/bn4b17_branch2a2/bn4b17_branch2a2")
    del f["model_weights"]['bn4b17_branch2a']

    f["model_weights"]["bn4b17_branch2a2"].attrs["weight_names"] = b'bn4b17_branch2a2/gamma:0', b'bn4b17_branch2a2/beta:0', b'bn4b17_branch2a2/moving_mean:0', b'bn4b17_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b17_branch2a_relu", "/model_weights/res4b17_branch2a_relu2")
    del f["model_weights"]['res4b17_branch2a_relu']

    f.copy("/model_weights/padding4b17_branch2b", "/model_weights/padding4b17_branch2b2")
    del f["model_weights"]['padding4b17_branch2b']

    f.copy("/model_weights/res4b17_branch2b/res4b17_branch2b", "/model_weights/res4b17_branch2b2/res4b17_branch2b2")
    del f["model_weights"]['res4b17_branch2b']

    f["model_weights"]["res4b17_branch2b2"].attrs["weight_names"] = [b'res4b17_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b17_branch2b/bn4b17_branch2b", "/model_weights/bn4b17_branch2b2/bn4b17_branch2b2")
    del f["model_weights"]['bn4b17_branch2b']

    f["model_weights"]["bn4b17_branch2b2"].attrs["weight_names"] = b'bn4b17_branch2b2/gamma:0', b'bn4b17_branch2b2/beta:0', b'bn4b17_branch2b2/moving_mean:0', b'bn4b17_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b17_branch2b_relu", "/model_weights/res4b17_branch2b_relu2")
    del f["model_weights"]['res4b17_branch2b_relu']

    f.copy("/model_weights/res4b17_branch2c/res4b17_branch2c", "/model_weights/res4b17_branch2c2/res4b17_branch2c2")
    del f["model_weights"]['res4b17_branch2c']

    f["model_weights"]["res4b17_branch2c2"].attrs["weight_names"] = [b'res4b17_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b17_branch2c/bn4b17_branch2c", "/model_weights/bn4b17_branch2c2/bn4b17_branch2c2")
    del f["model_weights"]['bn4b17_branch2c']

    f["model_weights"]["bn4b17_branch2c2"].attrs["weight_names"] = b'bn4b17_branch2c2/gamma:0', b'bn4b17_branch2c2/beta:0', b'bn4b17_branch2c2/moving_mean:0', b'bn4b17_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b17", "/model_weights/res4b172")
    del f["model_weights"]['res4b17']

    f.copy("/model_weights/res4b17_relu", "/model_weights/res4b17_relu2")
    del f["model_weights"]['res4b17_relu']

    f.copy("/model_weights/res4b18_branch2a/res4b18_branch2a", "/model_weights/res4b18_branch2a2/res4b18_branch2a2")
    del f["model_weights"]['res4b18_branch2a']

    f["model_weights"]["res4b18_branch2a2"].attrs["weight_names"] = [b'res4b18_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b18_branch2a/bn4b18_branch2a", "/model_weights/bn4b18_branch2a2/bn4b18_branch2a2")
    del f["model_weights"]['bn4b18_branch2a']

    f["model_weights"]["bn4b18_branch2a2"].attrs["weight_names"] = b'bn4b18_branch2a2/gamma:0', b'bn4b18_branch2a2/beta:0', b'bn4b18_branch2a2/moving_mean:0', b'bn4b18_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b18_branch2a_relu", "/model_weights/res4b18_branch2a_relu2")
    del f["model_weights"]['res4b18_branch2a_relu']

    f.copy("/model_weights/padding4b18_branch2b", "/model_weights/padding4b18_branch2b2")
    del f["model_weights"]['padding4b18_branch2b']

    f.copy("/model_weights/res4b18_branch2b/res4b18_branch2b", "/model_weights/res4b18_branch2b2/res4b18_branch2b2")
    del f["model_weights"]['res4b18_branch2b']

    f["model_weights"]["res4b18_branch2b2"].attrs["weight_names"] = [b'res4b18_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b18_branch2b/bn4b18_branch2b", "/model_weights/bn4b18_branch2b2/bn4b18_branch2b2")
    del f["model_weights"]['bn4b18_branch2b']

    f["model_weights"]["bn4b18_branch2b2"].attrs["weight_names"] = b'bn4b18_branch2b2/gamma:0', b'bn4b18_branch2b2/beta:0', b'bn4b18_branch2b2/moving_mean:0', b'bn4b18_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b18_branch2b_relu", "/model_weights/res4b18_branch2b_relu2")
    del f["model_weights"]['res4b18_branch2b_relu']

    f.copy("/model_weights/res4b18_branch2c/res4b18_branch2c", "/model_weights/res4b18_branch2c2/res4b18_branch2c2")
    del f["model_weights"]['res4b18_branch2c']

    f["model_weights"]["res4b18_branch2c2"].attrs["weight_names"] = [b'res4b18_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b18_branch2c/bn4b18_branch2c", "/model_weights/bn4b18_branch2c2/bn4b18_branch2c2")
    del f["model_weights"]['bn4b18_branch2c']

    f["model_weights"]["bn4b18_branch2c2"].attrs["weight_names"] = b'bn4b18_branch2c2/gamma:0', b'bn4b18_branch2c2/beta:0', b'bn4b18_branch2c2/moving_mean:0', b'bn4b18_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b18", "/model_weights/res4b182")
    del f["model_weights"]['res4b18']

    f.copy("/model_weights/res4b18_relu", "/model_weights/res4b18_relu2")
    del f["model_weights"]['res4b18_relu']

    f.copy("/model_weights/res4b19_branch2a/res4b19_branch2a", "/model_weights/res4b19_branch2a2/res4b19_branch2a2")
    del f["model_weights"]['res4b19_branch2a']

    f["model_weights"]["res4b19_branch2a2"].attrs["weight_names"] = [b'res4b19_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b19_branch2a/bn4b19_branch2a", "/model_weights/bn4b19_branch2a2/bn4b19_branch2a2")
    del f["model_weights"]['bn4b19_branch2a']

    f["model_weights"]["bn4b19_branch2a2"].attrs["weight_names"] = b'bn4b19_branch2a2/gamma:0', b'bn4b19_branch2a2/beta:0', b'bn4b19_branch2a2/moving_mean:0', b'bn4b19_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b19_branch2a_relu", "/model_weights/res4b19_branch2a_relu2")
    del f["model_weights"]['res4b19_branch2a_relu']

    f.copy("/model_weights/padding4b19_branch2b", "/model_weights/padding4b19_branch2b2")
    del f["model_weights"]['padding4b19_branch2b']

    f.copy("/model_weights/res4b19_branch2b/res4b19_branch2b", "/model_weights/res4b19_branch2b2/res4b19_branch2b2")
    del f["model_weights"]['res4b19_branch2b']

    f["model_weights"]["res4b19_branch2b2"].attrs["weight_names"] = [b'res4b19_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b19_branch2b/bn4b19_branch2b", "/model_weights/bn4b19_branch2b2/bn4b19_branch2b2")
    del f["model_weights"]['bn4b19_branch2b']

    f["model_weights"]["bn4b19_branch2b2"].attrs["weight_names"] = b'bn4b19_branch2b2/gamma:0', b'bn4b19_branch2b2/beta:0', b'bn4b19_branch2b2/moving_mean:0', b'bn4b19_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b19_branch2b_relu", "/model_weights/res4b19_branch2b_relu2")
    del f["model_weights"]['res4b19_branch2b_relu']

    f.copy("/model_weights/res4b19_branch2c/res4b19_branch2c", "/model_weights/res4b19_branch2c2/res4b19_branch2c2")
    del f["model_weights"]['res4b19_branch2c']

    f["model_weights"]["res4b19_branch2c2"].attrs["weight_names"] = [b'res4b19_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b19_branch2c/bn4b19_branch2c", "/model_weights/bn4b19_branch2c2/bn4b19_branch2c2")
    del f["model_weights"]['bn4b19_branch2c']

    f["model_weights"]["bn4b19_branch2c2"].attrs["weight_names"] = b'bn4b19_branch2c2/gamma:0', b'bn4b19_branch2c2/beta:0', b'bn4b19_branch2c2/moving_mean:0', b'bn4b19_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b19", "/model_weights/res4b192")
    del f["model_weights"]['res4b19']

    f.copy("/model_weights/res4b19_relu", "/model_weights/res4b19_relu2")
    del f["model_weights"]['res4b19_relu']

    f.copy("/model_weights/res4b20_branch2a/res4b20_branch2a", "/model_weights/res4b20_branch2a2/res4b20_branch2a2")
    del f["model_weights"]['res4b20_branch2a']

    f["model_weights"]["res4b20_branch2a2"].attrs["weight_names"] = [b'res4b20_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b20_branch2a/bn4b20_branch2a", "/model_weights/bn4b20_branch2a2/bn4b20_branch2a2")
    del f["model_weights"]['bn4b20_branch2a']

    f["model_weights"]["bn4b20_branch2a2"].attrs["weight_names"] = b'bn4b20_branch2a2/gamma:0', b'bn4b20_branch2a2/beta:0', b'bn4b20_branch2a2/moving_mean:0', b'bn4b20_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b20_branch2a_relu", "/model_weights/res4b20_branch2a_relu2")
    del f["model_weights"]['res4b20_branch2a_relu']

    f.copy("/model_weights/padding4b20_branch2b", "/model_weights/padding4b20_branch2b2")
    del f["model_weights"]['padding4b20_branch2b']

    f.copy("/model_weights/res4b20_branch2b/res4b20_branch2b", "/model_weights/res4b20_branch2b2/res4b20_branch2b2")
    del f["model_weights"]['res4b20_branch2b']

    f["model_weights"]["res4b20_branch2b2"].attrs["weight_names"] = [b'res4b20_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b20_branch2b/bn4b20_branch2b", "/model_weights/bn4b20_branch2b2/bn4b20_branch2b2")
    del f["model_weights"]['bn4b20_branch2b']

    f["model_weights"]["bn4b20_branch2b2"].attrs["weight_names"] = b'bn4b20_branch2b2/gamma:0', b'bn4b20_branch2b2/beta:0', b'bn4b20_branch2b2/moving_mean:0', b'bn4b20_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b20_branch2b_relu", "/model_weights/res4b20_branch2b_relu2")
    del f["model_weights"]['res4b20_branch2b_relu']

    f.copy("/model_weights/res4b20_branch2c/res4b20_branch2c", "/model_weights/res4b20_branch2c2/res4b20_branch2c2")
    del f["model_weights"]['res4b20_branch2c']

    f["model_weights"]["res4b20_branch2c2"].attrs["weight_names"] = [b'res4b20_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b20_branch2c/bn4b20_branch2c", "/model_weights/bn4b20_branch2c2/bn4b20_branch2c2")
    del f["model_weights"]['bn4b20_branch2c']

    f["model_weights"]["bn4b20_branch2c2"].attrs["weight_names"] = b'bn4b20_branch2c2/gamma:0', b'bn4b20_branch2c2/beta:0', b'bn4b20_branch2c2/moving_mean:0', b'bn4b20_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b20", "/model_weights/res4b202")
    del f["model_weights"]['res4b20']

    f.copy("/model_weights/res4b20_relu", "/model_weights/res4b20_relu2")
    del f["model_weights"]['res4b20_relu']

    f.copy("/model_weights/res4b21_branch2a/res4b21_branch2a", "/model_weights/res4b21_branch2a2/res4b21_branch2a2")
    del f["model_weights"]['res4b21_branch2a']

    f["model_weights"]["res4b21_branch2a2"].attrs["weight_names"] = [b'res4b21_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b21_branch2a/bn4b21_branch2a", "/model_weights/bn4b21_branch2a2/bn4b21_branch2a2")
    del f["model_weights"]['bn4b21_branch2a']

    f["model_weights"]["bn4b21_branch2a2"].attrs["weight_names"] = b'bn4b21_branch2a2/gamma:0', b'bn4b21_branch2a2/beta:0', b'bn4b21_branch2a2/moving_mean:0', b'bn4b21_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b21_branch2a_relu", "/model_weights/res4b21_branch2a_relu2")
    del f["model_weights"]['res4b21_branch2a_relu']

    f.copy("/model_weights/padding4b21_branch2b", "/model_weights/padding4b21_branch2b2")
    del f["model_weights"]['padding4b21_branch2b']

    f.copy("/model_weights/res4b21_branch2b/res4b21_branch2b", "/model_weights/res4b21_branch2b2/res4b21_branch2b2")
    del f["model_weights"]['res4b21_branch2b']

    f["model_weights"]["res4b21_branch2b2"].attrs["weight_names"] = [b'res4b21_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b21_branch2b/bn4b21_branch2b", "/model_weights/bn4b21_branch2b2/bn4b21_branch2b2")
    del f["model_weights"]['bn4b21_branch2b']

    f["model_weights"]["bn4b21_branch2b2"].attrs["weight_names"] = b'bn4b21_branch2b2/gamma:0', b'bn4b21_branch2b2/beta:0', b'bn4b21_branch2b2/moving_mean:0', b'bn4b21_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b21_branch2b_relu", "/model_weights/res4b21_branch2b_relu2")
    del f["model_weights"]['res4b21_branch2b_relu']

    f.copy("/model_weights/res4b21_branch2c/res4b21_branch2c", "/model_weights/res4b21_branch2c2/res4b21_branch2c2")
    del f["model_weights"]['res4b21_branch2c']

    f["model_weights"]["res4b21_branch2c2"].attrs["weight_names"] = [b'res4b21_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b21_branch2c/bn4b21_branch2c", "/model_weights/bn4b21_branch2c2/bn4b21_branch2c2")
    del f["model_weights"]['bn4b21_branch2c']

    f["model_weights"]["bn4b21_branch2c2"].attrs["weight_names"] = b'bn4b21_branch2c2/gamma:0', b'bn4b21_branch2c2/beta:0', b'bn4b21_branch2c2/moving_mean:0', b'bn4b21_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b21", "/model_weights/res4b212")
    del f["model_weights"]['res4b21']

    f.copy("/model_weights/res4b21_relu", "/model_weights/res4b21_relu2")
    del f["model_weights"]['res4b21_relu']

    f.copy("/model_weights/res4b22_branch2a/res4b22_branch2a", "/model_weights/res4b22_branch2a2/res4b22_branch2a2")
    del f["model_weights"]['res4b22_branch2a']

    f["model_weights"]["res4b22_branch2a2"].attrs["weight_names"] = [b'res4b22_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b22_branch2a/bn4b22_branch2a", "/model_weights/bn4b22_branch2a2/bn4b22_branch2a2")
    del f["model_weights"]['bn4b22_branch2a']

    f["model_weights"]["bn4b22_branch2a2"].attrs["weight_names"] = b'bn4b22_branch2a2/gamma:0', b'bn4b22_branch2a2/beta:0', b'bn4b22_branch2a2/moving_mean:0', b'bn4b22_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b22_branch2a_relu", "/model_weights/res4b22_branch2a_relu2")
    del f["model_weights"]['res4b22_branch2a_relu']

    f.copy("/model_weights/padding4b22_branch2b", "/model_weights/padding4b22_branch2b2")
    del f["model_weights"]['padding4b22_branch2b']

    f.copy("/model_weights/res4b22_branch2b/res4b22_branch2b", "/model_weights/res4b22_branch2b2/res4b22_branch2b2")
    del f["model_weights"]['res4b22_branch2b']

    f["model_weights"]["res4b22_branch2b2"].attrs["weight_names"] = [b'res4b22_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b22_branch2b/bn4b22_branch2b", "/model_weights/bn4b22_branch2b2/bn4b22_branch2b2")
    del f["model_weights"]['bn4b22_branch2b']

    f["model_weights"]["bn4b22_branch2b2"].attrs["weight_names"] = b'bn4b22_branch2b2/gamma:0', b'bn4b22_branch2b2/beta:0', b'bn4b22_branch2b2/moving_mean:0', b'bn4b22_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b22_branch2b_relu", "/model_weights/res4b22_branch2b_relu2")
    del f["model_weights"]['res4b22_branch2b_relu']

    f.copy("/model_weights/res4b22_branch2c/res4b22_branch2c", "/model_weights/res4b22_branch2c2/res4b22_branch2c2")
    del f["model_weights"]['res4b22_branch2c']

    f["model_weights"]["res4b22_branch2c2"].attrs["weight_names"] = [b'res4b22_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b22_branch2c/bn4b22_branch2c", "/model_weights/bn4b22_branch2c2/bn4b22_branch2c2")
    del f["model_weights"]['bn4b22_branch2c']

    f["model_weights"]["bn4b22_branch2c2"].attrs["weight_names"] = b'bn4b22_branch2c2/gamma:0', b'bn4b22_branch2c2/beta:0', b'bn4b22_branch2c2/moving_mean:0', b'bn4b22_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b22_relu", "/model_weights/res4b22_relu2")
    del f["model_weights"]['res4b22_relu']

    f.copy("/model_weights/res4b23_branch2a/res4b23_branch2a", "/model_weights/res4b23_branch2a2/res4b23_branch2a2")
    del f["model_weights"]['res4b23_branch2a']

    f["model_weights"]["res4b23_branch2a2"].attrs["weight_names"] = [b'res4b23_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b23_branch2a/bn4b23_branch2a", "/model_weights/bn4b23_branch2a2/bn4b23_branch2a2")
    del f["model_weights"]['bn4b23_branch2a']

    f["model_weights"]["bn4b23_branch2a2"].attrs[
        "weight_names"] = b'bn4b23_branch2a2/gamma:0', b'bn4b23_branch2a2/beta:0', b'bn4b23_branch2a2/moving_mean:0', b'bn4b23_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b23_branch2a_relu", "/model_weights/res4b23_branch2a_relu2")
    del f["model_weights"]['res4b23_branch2a_relu']

    f.copy("/model_weights/padding4b23_branch2b", "/model_weights/padding4b23_branch2b2")
    del f["model_weights"]['padding4b23_branch2b']

    f.copy("/model_weights/res4b23_branch2b/res4b23_branch2b", "/model_weights/res4b23_branch2b2/res4b23_branch2b2")
    del f["model_weights"]['res4b23_branch2b']

    f["model_weights"]["res4b23_branch2b2"].attrs["weight_names"] = [b'res4b23_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b23_branch2b/bn4b23_branch2b", "/model_weights/bn4b23_branch2b2/bn4b23_branch2b2")
    del f["model_weights"]['bn4b23_branch2b']

    f["model_weights"]["bn4b23_branch2b2"].attrs[
        "weight_names"] = b'bn4b23_branch2b2/gamma:0', b'bn4b23_branch2b2/beta:0', b'bn4b23_branch2b2/moving_mean:0', b'bn4b23_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b23_branch2b_relu", "/model_weights/res4b23_branch2b_relu2")
    del f["model_weights"]['res4b23_branch2b_relu']

    f.copy("/model_weights/res4b23_branch2c/res4b23_branch2c", "/model_weights/res4b23_branch2c2/res4b23_branch2c2")
    del f["model_weights"]['res4b23_branch2c']

    f["model_weights"]["res4b23_branch2c2"].attrs["weight_names"] = [b'res4b23_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b23_branch2c/bn4b23_branch2c", "/model_weights/bn4b23_branch2c2/bn4b23_branch2c2")
    del f["model_weights"]['bn4b23_branch2c']

    f["model_weights"]["bn4b23_branch2c2"].attrs[
        "weight_names"] = b'bn4b23_branch2c2/gamma:0', b'bn4b23_branch2c2/beta:0', b'bn4b23_branch2c2/moving_mean:0', b'bn4b23_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b23", "/model_weights/res4b232")
    del f["model_weights"]['res4b23']

    f.copy("/model_weights/res4b23_relu", "/model_weights/res4b23_relu2")
    del f["model_weights"]['res4b23_relu']

    f.copy("/model_weights/res4b24_branch2a/res4b24_branch2a", "/model_weights/res4b24_branch2a2/res4b24_branch2a2")
    del f["model_weights"]['res4b24_branch2a']

    f["model_weights"]["res4b24_branch2a2"].attrs["weight_names"] = [b'res4b24_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b24_branch2a/bn4b24_branch2a", "/model_weights/bn4b24_branch2a2/bn4b24_branch2a2")
    del f["model_weights"]['bn4b24_branch2a']

    f["model_weights"]["bn4b24_branch2a2"].attrs[
        "weight_names"] = b'bn4b24_branch2a2/gamma:0', b'bn4b24_branch2a2/beta:0', b'bn4b24_branch2a2/moving_mean:0', b'bn4b24_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b24_branch2a_relu", "/model_weights/res4b24_branch2a_relu2")
    del f["model_weights"]['res4b24_branch2a_relu']

    f.copy("/model_weights/padding4b24_branch2b", "/model_weights/padding4b24_branch2b2")
    del f["model_weights"]['padding4b24_branch2b']

    f.copy("/model_weights/res4b24_branch2b/res4b24_branch2b", "/model_weights/res4b24_branch2b2/res4b24_branch2b2")
    del f["model_weights"]['res4b24_branch2b']

    f["model_weights"]["res4b24_branch2b2"].attrs["weight_names"] = [b'res4b24_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b24_branch2b/bn4b24_branch2b", "/model_weights/bn4b24_branch2b2/bn4b24_branch2b2")
    del f["model_weights"]['bn4b24_branch2b']

    f["model_weights"]["bn4b24_branch2b2"].attrs[
        "weight_names"] = b'bn4b24_branch2b2/gamma:0', b'bn4b24_branch2b2/beta:0', b'bn4b24_branch2b2/moving_mean:0', b'bn4b24_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b24_branch2b_relu", "/model_weights/res4b24_branch2b_relu2")
    del f["model_weights"]['res4b24_branch2b_relu']

    f.copy("/model_weights/res4b24_branch2c/res4b24_branch2c", "/model_weights/res4b24_branch2c2/res4b24_branch2c2")
    del f["model_weights"]['res4b24_branch2c']

    f["model_weights"]["res4b24_branch2c2"].attrs["weight_names"] = [b'res4b24_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b24_branch2c/bn4b24_branch2c", "/model_weights/bn4b24_branch2c2/bn4b24_branch2c2")
    del f["model_weights"]['bn4b24_branch2c']

    f["model_weights"]["bn4b24_branch2c2"].attrs[
        "weight_names"] = b'bn4b24_branch2c2/gamma:0', b'bn4b24_branch2c2/beta:0', b'bn4b24_branch2c2/moving_mean:0', b'bn4b24_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b24", "/model_weights/res4b242")
    del f["model_weights"]['res4b24']

    f.copy("/model_weights/res4b24_relu", "/model_weights/res4b24_relu2")
    del f["model_weights"]['res4b24_relu']

    f.copy("/model_weights/res4b25_branch2a/res4b25_branch2a", "/model_weights/res4b25_branch2a2/res4b25_branch2a2")
    del f["model_weights"]['res4b25_branch2a']

    f["model_weights"]["res4b25_branch2a2"].attrs["weight_names"] = [b'res4b25_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b25_branch2a/bn4b25_branch2a", "/model_weights/bn4b25_branch2a2/bn4b25_branch2a2")
    del f["model_weights"]['bn4b25_branch2a']

    f["model_weights"]["bn4b25_branch2a2"].attrs[
        "weight_names"] = b'bn4b25_branch2a2/gamma:0', b'bn4b25_branch2a2/beta:0', b'bn4b25_branch2a2/moving_mean:0', b'bn4b25_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b25_branch2a_relu", "/model_weights/res4b25_branch2a_relu2")
    del f["model_weights"]['res4b25_branch2a_relu']

    f.copy("/model_weights/padding4b25_branch2b", "/model_weights/padding4b25_branch2b2")
    del f["model_weights"]['padding4b25_branch2b']

    f.copy("/model_weights/res4b25_branch2b/res4b25_branch2b", "/model_weights/res4b25_branch2b2/res4b25_branch2b2")
    del f["model_weights"]['res4b25_branch2b']

    f["model_weights"]["res4b25_branch2b2"].attrs["weight_names"] = [b'res4b25_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b25_branch2b/bn4b25_branch2b", "/model_weights/bn4b25_branch2b2/bn4b25_branch2b2")
    del f["model_weights"]['bn4b25_branch2b']

    f["model_weights"]["bn4b25_branch2b2"].attrs[
        "weight_names"] = b'bn4b25_branch2b2/gamma:0', b'bn4b25_branch2b2/beta:0', b'bn4b25_branch2b2/moving_mean:0', b'bn4b25_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b25_branch2b_relu", "/model_weights/res4b25_branch2b_relu2")
    del f["model_weights"]['res4b25_branch2b_relu']

    f.copy("/model_weights/res4b25_branch2c/res4b25_branch2c", "/model_weights/res4b25_branch2c2/res4b25_branch2c2")
    del f["model_weights"]['res4b25_branch2c']

    f["model_weights"]["res4b25_branch2c2"].attrs["weight_names"] = [b'res4b25_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b25_branch2c/bn4b25_branch2c", "/model_weights/bn4b25_branch2c2/bn4b25_branch2c2")
    del f["model_weights"]['bn4b25_branch2c']

    f["model_weights"]["bn4b25_branch2c2"].attrs[
        "weight_names"] = b'bn4b25_branch2c2/gamma:0', b'bn4b25_branch2c2/beta:0', b'bn4b25_branch2c2/moving_mean:0', b'bn4b25_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b25", "/model_weights/res4b252")
    del f["model_weights"]['res4b25']

    f.copy("/model_weights/res4b25_relu", "/model_weights/res4b25_relu2")
    del f["model_weights"]['res4b25_relu']

    f.copy("/model_weights/res4b26_branch2a/res4b26_branch2a", "/model_weights/res4b26_branch2a2/res4b26_branch2a2")
    del f["model_weights"]['res4b26_branch2a']

    f["model_weights"]["res4b26_branch2a2"].attrs["weight_names"] = [b'res4b26_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b26_branch2a/bn4b26_branch2a", "/model_weights/bn4b26_branch2a2/bn4b26_branch2a2")
    del f["model_weights"]['bn4b26_branch2a']

    f["model_weights"]["bn4b26_branch2a2"].attrs[
        "weight_names"] = b'bn4b26_branch2a2/gamma:0', b'bn4b26_branch2a2/beta:0', b'bn4b26_branch2a2/moving_mean:0', b'bn4b26_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b26_branch2a_relu", "/model_weights/res4b26_branch2a_relu2")
    del f["model_weights"]['res4b26_branch2a_relu']

    f.copy("/model_weights/padding4b26_branch2b", "/model_weights/padding4b26_branch2b2")
    del f["model_weights"]['padding4b26_branch2b']

    f.copy("/model_weights/res4b26_branch2b/res4b26_branch2b", "/model_weights/res4b26_branch2b2/res4b26_branch2b2")
    del f["model_weights"]['res4b26_branch2b']

    f["model_weights"]["res4b26_branch2b2"].attrs["weight_names"] = [b'res4b26_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b26_branch2b/bn4b26_branch2b", "/model_weights/bn4b26_branch2b2/bn4b26_branch2b2")
    del f["model_weights"]['bn4b26_branch2b']

    f["model_weights"]["bn4b26_branch2b2"].attrs[
        "weight_names"] = b'bn4b26_branch2b2/gamma:0', b'bn4b26_branch2b2/beta:0', b'bn4b26_branch2b2/moving_mean:0', b'bn4b26_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b26_branch2b_relu", "/model_weights/res4b26_branch2b_relu2")
    del f["model_weights"]['res4b26_branch2b_relu']

    f.copy("/model_weights/res4b26_branch2c/res4b26_branch2c", "/model_weights/res4b26_branch2c2/res4b26_branch2c2")
    del f["model_weights"]['res4b26_branch2c']

    f["model_weights"]["res4b26_branch2c2"].attrs["weight_names"] = [b'res4b26_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b26_branch2c/bn4b26_branch2c", "/model_weights/bn4b26_branch2c2/bn4b26_branch2c2")
    del f["model_weights"]['bn4b26_branch2c']

    f["model_weights"]["bn4b26_branch2c2"].attrs[
        "weight_names"] = b'bn4b26_branch2c2/gamma:0', b'bn4b26_branch2c2/beta:0', b'bn4b26_branch2c2/moving_mean:0', b'bn4b26_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b26", "/model_weights/res4b262")
    del f["model_weights"]['res4b26']

    f.copy("/model_weights/res4b26_relu", "/model_weights/res4b26_relu2")
    del f["model_weights"]['res4b26_relu']

    f.copy("/model_weights/res4b27_branch2a/res4b27_branch2a", "/model_weights/res4b27_branch2a2/res4b27_branch2a2")
    del f["model_weights"]['res4b27_branch2a']

    f["model_weights"]["res4b27_branch2a2"].attrs["weight_names"] = [b'res4b27_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b27_branch2a/bn4b27_branch2a", "/model_weights/bn4b27_branch2a2/bn4b27_branch2a2")
    del f["model_weights"]['bn4b27_branch2a']

    f["model_weights"]["bn4b27_branch2a2"].attrs[
        "weight_names"] = b'bn4b27_branch2a2/gamma:0', b'bn4b27_branch2a2/beta:0', b'bn4b27_branch2a2/moving_mean:0', b'bn4b27_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b27_branch2a_relu", "/model_weights/res4b27_branch2a_relu2")
    del f["model_weights"]['res4b27_branch2a_relu']

    f.copy("/model_weights/padding4b27_branch2b", "/model_weights/padding4b27_branch2b2")
    del f["model_weights"]['padding4b27_branch2b']

    f.copy("/model_weights/res4b27_branch2b/res4b27_branch2b", "/model_weights/res4b27_branch2b2/res4b27_branch2b2")
    del f["model_weights"]['res4b27_branch2b']

    f["model_weights"]["res4b27_branch2b2"].attrs["weight_names"] = [b'res4b27_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b27_branch2b/bn4b27_branch2b", "/model_weights/bn4b27_branch2b2/bn4b27_branch2b2")
    del f["model_weights"]['bn4b27_branch2b']

    f["model_weights"]["bn4b27_branch2b2"].attrs[
        "weight_names"] = b'bn4b27_branch2b2/gamma:0', b'bn4b27_branch2b2/beta:0', b'bn4b27_branch2b2/moving_mean:0', b'bn4b27_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b27_branch2b_relu", "/model_weights/res4b27_branch2b_relu2")
    del f["model_weights"]['res4b27_branch2b_relu']

    f.copy("/model_weights/res4b27_branch2c/res4b27_branch2c", "/model_weights/res4b27_branch2c2/res4b27_branch2c2")
    del f["model_weights"]['res4b27_branch2c']

    f["model_weights"]["res4b27_branch2c2"].attrs["weight_names"] = [b'res4b27_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b27_branch2c/bn4b27_branch2c", "/model_weights/bn4b27_branch2c2/bn4b27_branch2c2")
    del f["model_weights"]['bn4b27_branch2c']

    f["model_weights"]["bn4b27_branch2c2"].attrs[
        "weight_names"] = b'bn4b27_branch2c2/gamma:0', b'bn4b27_branch2c2/beta:0', b'bn4b27_branch2c2/moving_mean:0', b'bn4b27_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b27", "/model_weights/res4b272")
    del f["model_weights"]['res4b27']

    f.copy("/model_weights/res4b27_relu", "/model_weights/res4b27_relu2")
    del f["model_weights"]['res4b27_relu']

    f.copy("/model_weights/res4b28_branch2a/res4b28_branch2a", "/model_weights/res4b28_branch2a2/res4b28_branch2a2")
    del f["model_weights"]['res4b28_branch2a']

    f["model_weights"]["res4b28_branch2a2"].attrs["weight_names"] = [b'res4b28_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b28_branch2a/bn4b28_branch2a", "/model_weights/bn4b28_branch2a2/bn4b28_branch2a2")
    del f["model_weights"]['bn4b28_branch2a']

    f["model_weights"]["bn4b28_branch2a2"].attrs[
        "weight_names"] = b'bn4b28_branch2a2/gamma:0', b'bn4b28_branch2a2/beta:0', b'bn4b28_branch2a2/moving_mean:0', b'bn4b28_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b28_branch2a_relu", "/model_weights/res4b28_branch2a_relu2")
    del f["model_weights"]['res4b28_branch2a_relu']

    f.copy("/model_weights/padding4b28_branch2b", "/model_weights/padding4b28_branch2b2")
    del f["model_weights"]['padding4b28_branch2b']

    f.copy("/model_weights/res4b28_branch2b/res4b28_branch2b", "/model_weights/res4b28_branch2b2/res4b28_branch2b2")
    del f["model_weights"]['res4b28_branch2b']

    f["model_weights"]["res4b28_branch2b2"].attrs["weight_names"] = [b'res4b28_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b28_branch2b/bn4b28_branch2b", "/model_weights/bn4b28_branch2b2/bn4b28_branch2b2")
    del f["model_weights"]['bn4b28_branch2b']

    f["model_weights"]["bn4b28_branch2b2"].attrs[
        "weight_names"] = b'bn4b28_branch2b2/gamma:0', b'bn4b28_branch2b2/beta:0', b'bn4b28_branch2b2/moving_mean:0', b'bn4b28_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b28_branch2b_relu", "/model_weights/res4b28_branch2b_relu2")
    del f["model_weights"]['res4b28_branch2b_relu']

    f.copy("/model_weights/res4b28_branch2c/res4b28_branch2c", "/model_weights/res4b28_branch2c2/res4b28_branch2c2")
    del f["model_weights"]['res4b28_branch2c']

    f["model_weights"]["res4b28_branch2c2"].attrs["weight_names"] = [b'res4b28_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b28_branch2c/bn4b28_branch2c", "/model_weights/bn4b28_branch2c2/bn4b28_branch2c2")
    del f["model_weights"]['bn4b28_branch2c']

    f["model_weights"]["bn4b28_branch2c2"].attrs[
        "weight_names"] = b'bn4b28_branch2c2/gamma:0', b'bn4b28_branch2c2/beta:0', b'bn4b28_branch2c2/moving_mean:0', b'bn4b28_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b28", "/model_weights/res4b282")
    del f["model_weights"]['res4b28']

    f.copy("/model_weights/res4b28_relu", "/model_weights/res4b28_relu2")
    del f["model_weights"]['res4b28_relu']

    f.copy("/model_weights/res4b29_branch2a/res4b29_branch2a", "/model_weights/res4b29_branch2a2/res4b29_branch2a2")
    del f["model_weights"]['res4b29_branch2a']

    f["model_weights"]["res4b29_branch2a2"].attrs["weight_names"] = [b'res4b29_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b29_branch2a/bn4b29_branch2a", "/model_weights/bn4b29_branch2a2/bn4b29_branch2a2")
    del f["model_weights"]['bn4b29_branch2a']

    f["model_weights"]["bn4b29_branch2a2"].attrs[
        "weight_names"] = b'bn4b29_branch2a2/gamma:0', b'bn4b29_branch2a2/beta:0', b'bn4b29_branch2a2/moving_mean:0', b'bn4b29_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b29_branch2a_relu", "/model_weights/res4b29_branch2a_relu2")
    del f["model_weights"]['res4b29_branch2a_relu']

    f.copy("/model_weights/padding4b29_branch2b", "/model_weights/padding4b29_branch2b2")
    del f["model_weights"]['padding4b29_branch2b']

    f.copy("/model_weights/res4b29_branch2b/res4b29_branch2b", "/model_weights/res4b29_branch2b2/res4b29_branch2b2")
    del f["model_weights"]['res4b29_branch2b']

    f["model_weights"]["res4b29_branch2b2"].attrs["weight_names"] = [b'res4b29_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b29_branch2b/bn4b29_branch2b", "/model_weights/bn4b29_branch2b2/bn4b29_branch2b2")
    del f["model_weights"]['bn4b29_branch2b']

    f["model_weights"]["bn4b29_branch2b2"].attrs[
        "weight_names"] = b'bn4b29_branch2b2/gamma:0', b'bn4b29_branch2b2/beta:0', b'bn4b29_branch2b2/moving_mean:0', b'bn4b29_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b29_branch2b_relu", "/model_weights/res4b29_branch2b_relu2")
    del f["model_weights"]['res4b29_branch2b_relu']

    f.copy("/model_weights/res4b29_branch2c/res4b29_branch2c", "/model_weights/res4b29_branch2c2/res4b29_branch2c2")
    del f["model_weights"]['res4b29_branch2c']

    f["model_weights"]["res4b29_branch2c2"].attrs["weight_names"] = [b'res4b29_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b29_branch2c/bn4b29_branch2c", "/model_weights/bn4b29_branch2c2/bn4b29_branch2c2")
    del f["model_weights"]['bn4b29_branch2c']

    f["model_weights"]["bn4b29_branch2c2"].attrs[
        "weight_names"] = b'bn4b29_branch2c2/gamma:0', b'bn4b29_branch2c2/beta:0', b'bn4b29_branch2c2/moving_mean:0', b'bn4b29_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b29", "/model_weights/res4b292")
    del f["model_weights"]['res4b29']

    f.copy("/model_weights/res4b29_relu", "/model_weights/res4b29_relu2")
    del f["model_weights"]['res4b29_relu']

    f.copy("/model_weights/res4b30_branch2a/res4b30_branch2a", "/model_weights/res4b30_branch2a2/res4b30_branch2a2")
    del f["model_weights"]['res4b30_branch2a']

    f["model_weights"]["res4b30_branch2a2"].attrs["weight_names"] = [b'res4b30_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b30_branch2a/bn4b30_branch2a", "/model_weights/bn4b30_branch2a2/bn4b30_branch2a2")
    del f["model_weights"]['bn4b30_branch2a']

    f["model_weights"]["bn4b30_branch2a2"].attrs[
        "weight_names"] = b'bn4b30_branch2a2/gamma:0', b'bn4b30_branch2a2/beta:0', b'bn4b30_branch2a2/moving_mean:0', b'bn4b30_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b30_branch2a_relu", "/model_weights/res4b30_branch2a_relu2")
    del f["model_weights"]['res4b30_branch2a_relu']

    f.copy("/model_weights/padding4b30_branch2b", "/model_weights/padding4b30_branch2b2")
    del f["model_weights"]['padding4b30_branch2b']

    f.copy("/model_weights/res4b30_branch2b/res4b30_branch2b", "/model_weights/res4b30_branch2b2/res4b30_branch2b2")
    del f["model_weights"]['res4b30_branch2b']

    f["model_weights"]["res4b30_branch2b2"].attrs["weight_names"] = [b'res4b30_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b30_branch2b/bn4b30_branch2b", "/model_weights/bn4b30_branch2b2/bn4b30_branch2b2")
    del f["model_weights"]['bn4b30_branch2b']

    f["model_weights"]["bn4b30_branch2b2"].attrs[
        "weight_names"] = b'bn4b30_branch2b2/gamma:0', b'bn4b30_branch2b2/beta:0', b'bn4b30_branch2b2/moving_mean:0', b'bn4b30_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b30_branch2b_relu", "/model_weights/res4b30_branch2b_relu2")
    del f["model_weights"]['res4b30_branch2b_relu']

    f.copy("/model_weights/res4b30_branch2c/res4b30_branch2c", "/model_weights/res4b30_branch2c2/res4b30_branch2c2")
    del f["model_weights"]['res4b30_branch2c']

    f["model_weights"]["res4b30_branch2c2"].attrs["weight_names"] = [b'res4b30_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b30_branch2c/bn4b30_branch2c", "/model_weights/bn4b30_branch2c2/bn4b30_branch2c2")
    del f["model_weights"]['bn4b30_branch2c']

    f["model_weights"]["bn4b30_branch2c2"].attrs[
        "weight_names"] = b'bn4b30_branch2c2/gamma:0', b'bn4b30_branch2c2/beta:0', b'bn4b30_branch2c2/moving_mean:0', b'bn4b30_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b30", "/model_weights/res4b302")
    del f["model_weights"]['res4b30']

    f.copy("/model_weights/res4b30_relu", "/model_weights/res4b30_relu2")
    del f["model_weights"]['res4b30_relu']

    f.copy("/model_weights/res4b31_branch2a/res4b31_branch2a", "/model_weights/res4b31_branch2a2/res4b31_branch2a2")
    del f["model_weights"]['res4b31_branch2a']

    f["model_weights"]["res4b31_branch2a2"].attrs["weight_names"] = [b'res4b31_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b31_branch2a/bn4b31_branch2a", "/model_weights/bn4b31_branch2a2/bn4b31_branch2a2")
    del f["model_weights"]['bn4b31_branch2a']

    f["model_weights"]["bn4b31_branch2a2"].attrs[
        "weight_names"] = b'bn4b31_branch2a2/gamma:0', b'bn4b31_branch2a2/beta:0', b'bn4b31_branch2a2/moving_mean:0', b'bn4b31_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b31_branch2a_relu", "/model_weights/res4b31_branch2a_relu2")
    del f["model_weights"]['res4b31_branch2a_relu']

    f.copy("/model_weights/padding4b31_branch2b", "/model_weights/padding4b31_branch2b2")
    del f["model_weights"]['padding4b31_branch2b']

    f.copy("/model_weights/res4b31_branch2b/res4b31_branch2b", "/model_weights/res4b31_branch2b2/res4b31_branch2b2")
    del f["model_weights"]['res4b31_branch2b']

    f["model_weights"]["res4b31_branch2b2"].attrs["weight_names"] = [b'res4b31_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b31_branch2b/bn4b31_branch2b", "/model_weights/bn4b31_branch2b2/bn4b31_branch2b2")
    del f["model_weights"]['bn4b31_branch2b']

    f["model_weights"]["bn4b31_branch2b2"].attrs[
        "weight_names"] = b'bn4b31_branch2b2/gamma:0', b'bn4b31_branch2b2/beta:0', b'bn4b31_branch2b2/moving_mean:0', b'bn4b31_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b31_branch2b_relu", "/model_weights/res4b31_branch2b_relu2")
    del f["model_weights"]['res4b31_branch2b_relu']

    f.copy("/model_weights/res4b31_branch2c/res4b31_branch2c", "/model_weights/res4b31_branch2c2/res4b31_branch2c2")
    del f["model_weights"]['res4b31_branch2c']

    f["model_weights"]["res4b31_branch2c2"].attrs["weight_names"] = [b'res4b31_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b31_branch2c/bn4b31_branch2c", "/model_weights/bn4b31_branch2c2/bn4b31_branch2c2")
    del f["model_weights"]['bn4b31_branch2c']

    f["model_weights"]["bn4b31_branch2c2"].attrs[
        "weight_names"] = b'bn4b31_branch2c2/gamma:0', b'bn4b31_branch2c2/beta:0', b'bn4b31_branch2c2/moving_mean:0', b'bn4b31_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b31", "/model_weights/res4b312")
    del f["model_weights"]['res4b31']

    f.copy("/model_weights/res4b31_relu", "/model_weights/res4b31_relu2")
    del f["model_weights"]['res4b31_relu']

    f.copy("/model_weights/res4b32_branch2a/res4b32_branch2a", "/model_weights/res4b32_branch2a2/res4b32_branch2a2")
    del f["model_weights"]['res4b32_branch2a']

    f["model_weights"]["res4b32_branch2a2"].attrs["weight_names"] = [b'res4b32_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b32_branch2a/bn4b32_branch2a", "/model_weights/bn4b32_branch2a2/bn4b32_branch2a2")
    del f["model_weights"]['bn4b32_branch2a']

    f["model_weights"]["bn4b32_branch2a2"].attrs[
        "weight_names"] = b'bn4b32_branch2a2/gamma:0', b'bn4b32_branch2a2/beta:0', b'bn4b32_branch2a2/moving_mean:0', b'bn4b32_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b32_branch2a_relu", "/model_weights/res4b32_branch2a_relu2")
    del f["model_weights"]['res4b32_branch2a_relu']

    f.copy("/model_weights/padding4b32_branch2b", "/model_weights/padding4b32_branch2b2")
    del f["model_weights"]['padding4b32_branch2b']

    f.copy("/model_weights/res4b32_branch2b/res4b32_branch2b", "/model_weights/res4b32_branch2b2/res4b32_branch2b2")
    del f["model_weights"]['res4b32_branch2b']

    f["model_weights"]["res4b32_branch2b2"].attrs["weight_names"] = [b'res4b32_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b32_branch2b/bn4b32_branch2b", "/model_weights/bn4b32_branch2b2/bn4b32_branch2b2")
    del f["model_weights"]['bn4b32_branch2b']

    f["model_weights"]["bn4b32_branch2b2"].attrs[
        "weight_names"] = b'bn4b32_branch2b2/gamma:0', b'bn4b32_branch2b2/beta:0', b'bn4b32_branch2b2/moving_mean:0', b'bn4b32_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b32_branch2b_relu", "/model_weights/res4b32_branch2b_relu2")
    del f["model_weights"]['res4b32_branch2b_relu']

    f.copy("/model_weights/res4b32_branch2c/res4b32_branch2c", "/model_weights/res4b32_branch2c2/res4b32_branch2c2")
    del f["model_weights"]['res4b32_branch2c']

    f["model_weights"]["res4b32_branch2c2"].attrs["weight_names"] = [b'res4b32_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b32_branch2c/bn4b32_branch2c", "/model_weights/bn4b32_branch2c2/bn4b32_branch2c2")
    del f["model_weights"]['bn4b32_branch2c']

    f["model_weights"]["bn4b32_branch2c2"].attrs[
        "weight_names"] = b'bn4b32_branch2c2/gamma:0', b'bn4b32_branch2c2/beta:0', b'bn4b32_branch2c2/moving_mean:0', b'bn4b32_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b32_relu", "/model_weights/res4b32_relu2")
    del f["model_weights"]['res4b32_relu']

    f.copy("/model_weights/res4b33_branch2a/res4b33_branch2a", "/model_weights/res4b33_branch2a2/res4b33_branch2a2")
    del f["model_weights"]['res4b33_branch2a']

    f["model_weights"]["res4b33_branch2a2"].attrs["weight_names"] = [b'res4b33_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b33_branch2a/bn4b33_branch2a", "/model_weights/bn4b33_branch2a2/bn4b33_branch2a2")
    del f["model_weights"]['bn4b33_branch2a']

    f["model_weights"]["bn4b33_branch2a2"].attrs["weight_names"] = b'bn4b33_branch2a2/gamma:0', b'bn4b33_branch2a2/beta:0', b'bn4b33_branch2a2/moving_mean:0', b'bn4b33_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b33_branch2a_relu", "/model_weights/res4b33_branch2a_relu2")
    del f["model_weights"]['res4b33_branch2a_relu']

    f.copy("/model_weights/padding4b33_branch2b", "/model_weights/padding4b33_branch2b2")
    del f["model_weights"]['padding4b33_branch2b']

    f.copy("/model_weights/res4b33_branch2b/res4b33_branch2b", "/model_weights/res4b33_branch2b2/res4b33_branch2b2")
    del f["model_weights"]['res4b33_branch2b']

    f["model_weights"]["res4b33_branch2b2"].attrs["weight_names"] = [b'res4b33_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b33_branch2b/bn4b33_branch2b", "/model_weights/bn4b33_branch2b2/bn4b33_branch2b2")
    del f["model_weights"]['bn4b33_branch2b']

    f["model_weights"]["bn4b33_branch2b2"].attrs["weight_names"] = b'bn4b33_branch2b2/gamma:0', b'bn4b33_branch2b2/beta:0', b'bn4b33_branch2b2/moving_mean:0', b'bn4b33_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b33_branch2b_relu", "/model_weights/res4b33_branch2b_relu2")
    del f["model_weights"]['res4b33_branch2b_relu']

    f.copy("/model_weights/res4b33_branch2c/res4b33_branch2c", "/model_weights/res4b33_branch2c2/res4b33_branch2c2")
    del f["model_weights"]['res4b33_branch2c']

    f["model_weights"]["res4b33_branch2c2"].attrs["weight_names"] = [b'res4b33_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b33_branch2c/bn4b33_branch2c", "/model_weights/bn4b33_branch2c2/bn4b33_branch2c2")
    del f["model_weights"]['bn4b33_branch2c']

    f["model_weights"]["bn4b33_branch2c2"].attrs["weight_names"] = b'bn4b33_branch2c2/gamma:0', b'bn4b33_branch2c2/beta:0', b'bn4b33_branch2c2/moving_mean:0', b'bn4b33_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b33", "/model_weights/res4b332")
    del f["model_weights"]['res4b33']

    f.copy("/model_weights/res4b33_relu", "/model_weights/res4b33_relu2")
    del f["model_weights"]['res4b33_relu']

    f.copy("/model_weights/res4b34_branch2a/res4b34_branch2a", "/model_weights/res4b34_branch2a2/res4b34_branch2a2")
    del f["model_weights"]['res4b34_branch2a']

    f["model_weights"]["res4b34_branch2a2"].attrs["weight_names"] = [b'res4b34_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b34_branch2a/bn4b34_branch2a", "/model_weights/bn4b34_branch2a2/bn4b34_branch2a2")
    del f["model_weights"]['bn4b34_branch2a']

    f["model_weights"]["bn4b34_branch2a2"].attrs["weight_names"] = b'bn4b34_branch2a2/gamma:0', b'bn4b34_branch2a2/beta:0', b'bn4b34_branch2a2/moving_mean:0', b'bn4b34_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b34_branch2a_relu", "/model_weights/res4b34_branch2a_relu2")
    del f["model_weights"]['res4b34_branch2a_relu']

    f.copy("/model_weights/padding4b34_branch2b", "/model_weights/padding4b34_branch2b2")
    del f["model_weights"]['padding4b34_branch2b']

    f.copy("/model_weights/res4b34_branch2b/res4b34_branch2b", "/model_weights/res4b34_branch2b2/res4b34_branch2b2")
    del f["model_weights"]['res4b34_branch2b']

    f["model_weights"]["res4b34_branch2b2"].attrs["weight_names"] = [b'res4b34_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b34_branch2b/bn4b34_branch2b", "/model_weights/bn4b34_branch2b2/bn4b34_branch2b2")
    del f["model_weights"]['bn4b34_branch2b']

    f["model_weights"]["bn4b34_branch2b2"].attrs["weight_names"] = b'bn4b34_branch2b2/gamma:0', b'bn4b34_branch2b2/beta:0', b'bn4b34_branch2b2/moving_mean:0', b'bn4b34_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b34_branch2b_relu", "/model_weights/res4b34_branch2b_relu2")
    del f["model_weights"]['res4b34_branch2b_relu']

    f.copy("/model_weights/res4b34_branch2c/res4b34_branch2c", "/model_weights/res4b34_branch2c2/res4b34_branch2c2")
    del f["model_weights"]['res4b34_branch2c']

    f["model_weights"]["res4b34_branch2c2"].attrs["weight_names"] = [b'res4b34_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b34_branch2c/bn4b34_branch2c", "/model_weights/bn4b34_branch2c2/bn4b34_branch2c2")
    del f["model_weights"]['bn4b34_branch2c']

    f["model_weights"]["bn4b34_branch2c2"].attrs["weight_names"] = b'bn4b34_branch2c2/gamma:0', b'bn4b34_branch2c2/beta:0', b'bn4b34_branch2c2/moving_mean:0', b'bn4b34_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b34", "/model_weights/res4b342")
    del f["model_weights"]['res4b34']

    f.copy("/model_weights/res4b34_relu", "/model_weights/res4b34_relu2")
    del f["model_weights"]['res4b34_relu']

    f.copy("/model_weights/res4b35_branch2a/res4b35_branch2a", "/model_weights/res4b35_branch2a2/res4b35_branch2a2")
    del f["model_weights"]['res4b35_branch2a']

    f["model_weights"]["res4b35_branch2a2"].attrs["weight_names"] = [b'res4b35_branch2a2/kernel:0']

    f.copy("/model_weights/bn4b35_branch2a/bn4b35_branch2a", "/model_weights/bn4b35_branch2a2/bn4b35_branch2a2")
    del f["model_weights"]['bn4b35_branch2a']

    f["model_weights"]["bn4b35_branch2a2"].attrs["weight_names"] = b'bn4b35_branch2a2/gamma:0', b'bn4b35_branch2a2/beta:0', b'bn4b35_branch2a2/moving_mean:0', b'bn4b35_branch2a2/moving_variance:0'

    f.copy("/model_weights/res4b35_branch2a_relu", "/model_weights/res4b35_branch2a_relu2")
    del f["model_weights"]['res4b35_branch2a_relu']

    f.copy("/model_weights/padding4b35_branch2b", "/model_weights/padding4b35_branch2b2")
    del f["model_weights"]['padding4b35_branch2b']

    f.copy("/model_weights/res4b35_branch2b/res4b35_branch2b", "/model_weights/res4b35_branch2b2/res4b35_branch2b2")
    del f["model_weights"]['res4b35_branch2b']

    f["model_weights"]["res4b35_branch2b2"].attrs["weight_names"] = [b'res4b35_branch2b2/kernel:0']

    f.copy("/model_weights/bn4b35_branch2b/bn4b35_branch2b", "/model_weights/bn4b35_branch2b2/bn4b35_branch2b2")
    del f["model_weights"]['bn4b35_branch2b']

    f["model_weights"]["bn4b35_branch2b2"].attrs["weight_names"] = b'bn4b35_branch2b2/gamma:0', b'bn4b35_branch2b2/beta:0', b'bn4b35_branch2b2/moving_mean:0', b'bn4b35_branch2b2/moving_variance:0'

    f.copy("/model_weights/res4b35_branch2b_relu", "/model_weights/res4b35_branch2b_relu2")
    del f["model_weights"]['res4b35_branch2b_relu']

    f.copy("/model_weights/res4b35_branch2c/res4b35_branch2c", "/model_weights/res4b35_branch2c2/res4b35_branch2c2")
    del f["model_weights"]['res4b35_branch2c']

    f["model_weights"]["res4b35_branch2c2"].attrs["weight_names"] = [b'res4b35_branch2c2/kernel:0']

    f.copy("/model_weights/bn4b35_branch2c/bn4b35_branch2c", "/model_weights/bn4b35_branch2c2/bn4b35_branch2c2")
    del f["model_weights"]['bn4b35_branch2c']

    f["model_weights"]["bn4b35_branch2c2"].attrs["weight_names"] = b'bn4b35_branch2c2/gamma:0', b'bn4b35_branch2c2/beta:0', b'bn4b35_branch2c2/moving_mean:0', b'bn4b35_branch2c2/moving_variance:0'

    f.copy("/model_weights/res4b35", "/model_weights/res4b352")
    del f["model_weights"]['res4b35']

    f.copy("/model_weights/res4b35_relu", "/model_weights/res4b35_relu2")
    del f["model_weights"]['res4b35_relu']

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

    f["model_weights"]["regression_submodel2"].attrs["weight_names"] = b'pyramid_regression_0/kernel:0', b'pyramid_regression_0/bias:0', b'pyramid_regression_1/kernel:0', b'pyramid_regression_1/bias:0', b'pyramid_regression_2/kernel:0', b'pyramid_regression_2/bias:0', b'pyramid_regression_3/kernel:0', b'pyramid_regression_3/bias:0', b'pyramid_regression/kernel:0', b'pyramid_regression/bias:0'

    f.copy("/model_weights/regression", "/model_weights/regression2")
    del f["model_weights"]['regression']

    f.copy("/model_weights/classification_submodel", "/model_weights/classification_submodel2")
    del f["model_weights"]['classification_submodel']

    f.copy("/model_weights/classification", "/model_weights/classification2")
    del f["model_weights"]['classification']

    with open("/home/rblin/Documents/Image-Processing/Neural-networks/model_config152.txt", "r") as t:
        text = t.readlines()
        # f.attrs["model_config"] = text[-1][:-1].encode('utf-8')
        f.attrs["model_config"] = text[-1].encode('utf-8')

    with open("/home/rblin/Documents/Image-Processing/Neural-networks/training_config152.txt", "r") as t:
        text = t.readlines()
        # f.attrs["training_config"] = text[-1][:-1].encode('utf-8')
        f.attrs["training_config"] = text[-1].encode('utf-8')

    with open("/home/rblin/Documents/Image-Processing/Neural-networks/layer_names152.txt", "r") as t:
        text = t.readlines()
        # np_array = np.array(text[-1][:-1].split(',')).astype(np.bytes_)
        np_array = np.array(text[-1].split(',')).astype(np.bytes_)
        f["model_weights"].attrs["layer_names"] = np_array

#filepath = "/home/rblin/Documents/weights/test_rename/resnet50_pascal_08.h5"
#rename_resnet50(filepath)

#filepath = "/home/rblin/Documents/weights/test_rename/resnet101_pascal_02.h5"
#rename_resnet101(filepath)

filepath = "/home/rblin/Documents/weights/test_rename/resnet152_pascal_08.h5"
rename_resnet152(filepath)

