import os


def load_net_structure(net_name):

    if net_name == "EfficientNetB2":
        from tensorflow.python.keras.applications.efficientnet import (
            # EfficientNetB2,
            preprocess_input,
            decode_predictions,
        )

        # net = EfficientNetB2()


    elif net_name == "EfficientNetB0":
        from tensorflow.python.keras.applications.efficientnet import (
            # EfficientNetB4,
            preprocess_input,
            decode_predictions,
        )


    elif net_name == "Xception":
        from tensorflow.keras.applications.xception import (
            # Xception,
            preprocess_input,
            decode_predictions,
        )

        # net = Xception()

    elif net_name == "VGG16":
        from tensorflow.keras.applications.vgg16 import (
            # VGG16,
            preprocess_input,
            decode_predictions,
        )

        # net = VGG16()


    elif net_name == "ResNet50":
        from tensorflow.keras.applications.resnet import (
            # ResNet50,
            preprocess_input,
            decode_predictions,
        )

        # net = ResNet50()


    elif net_name == "ResNet101":
        from tensorflow.keras.applications.resnet import (
            # ResNet101,
            preprocess_input,
            decode_predictions,
        )

        # net = ResNet101()


    elif net_name == "DenseNet121":
        from tensorflow.keras.applications.densenet import (
            # DenseNet121,
            preprocess_input,
            decode_predictions,
        )

        # net = DenseNet121()

    elif net_name == "DenseNet169":
        from tensorflow.keras.applications.densenet import (
            # DenseNet169,
            preprocess_input,
            decode_predictions,
        )

        # net = DenseNet169()

    elif net_name == "DenseNet201":
        from tensorflow.keras.applications.densenet import (
            # DenseNet201,
            preprocess_input,
            decode_predictions,
        )

        # net = DenseNet201()

    elif net_name == "InceptionV3":
        from tensorflow.keras.applications.inception_v3 import (
            # InceptionV3,
            preprocess_input,
            decode_predictions,
        )


    else:
        raise NameError("You should input a correct net name")
    

    return preprocess_input, decode_predictions