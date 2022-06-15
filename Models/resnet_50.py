import tensorflow as tf
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.models import Model
from keras.applications.resnet import ResNet50, preprocess_input

from Models.parent_model import ParentModel


class ResNetTuned(ParentModel):
    def __init__(self, input_dims=(224, 224, 3), n_outputs=18, pretrained=True, pooling=True):
        super().__init__()
        self.name = "resnet50"

        # Load the mobilenet architecture
        if pretrained:
            weights = "imagenet"
        else:
            weights = None
        if pooling:
            pooling = "avg"
        else:
            pooling = None
        self.base_model = ResNet50(
            input_shape=input_dims,
            include_top=False,
            weights=weights,
            pooling=pooling
        )

        # Preprocessing required for MobileNet
        i = Input([input_dims[1], input_dims[0], 3], dtype=tf.uint8)
        x = tf.cast(i, tf.float32)
        x = preprocess_input(x)
        x = self.base_model(x)

        # Add prediction layer to the model. If pooling is used, the result of global average pooling is used as the
        # input to this layer (1280 features). This means we only need a dense layer. If the pooling is not used, we
        # perform convolution that gets us a feature map of depth 1 and we provide that to the dense layer.
        if pooling is None:
            x = Conv2D(filters=1, kernel_size=1)(x)
            x = Flatten()(x)

        x = Dense(n_outputs)(x)
        self.model = Model(inputs=[i], outputs=[x])

        print(self.model.summary())
