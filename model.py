from tensorflow.python.keras.layers import Conv2D,Activation,Dropout,Flatten,Dense,MaxPooling2D,Concatenate
class CNN(object):
    def __init__(self,keep_drop):
        self.keep_drop = keep_drop
    def run_model(self,inputs):
        # 第1层卷积
        conv1 = Conv2D(32, (3, 3), name='conv1')(inputs)
        relu1 = Activation('relu', name="relu1")(conv1)

        # 第2 层卷积
        conv2 = Conv2D(32, (3, 3), name="conv2")(relu1)
        relu2 = Activation('relu', name="relu2")(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2), padding='same', name="pool2")(relu2)

        # 第3层卷积
        conv3 = Conv2D(64, (3, 3), name="conv3")(pool2)
        relu3 = Activation("relu", name="relu3")(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2), padding="same", name="pool3")(relu3)

        # 将Pooled feature map 摊平后输入全连接网络
        x = Flatten()(pool3)

        # Dropout
        x = Dropout(self.keep_drop)(x)

        # 4个全连接层分别做36分类，分别对应四个字符
        x = [Dense(36, activation='softmax', name="fc%d" % (i + 1))(x) for i in range(4)]
        outs = Concatenate()(x)
        return outs
