import numpy as np
import random,os,shutil,glob,time
from captcha.image import ImageCaptcha
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input
from model import CNN


class train_model(CNN):
    def __init__(self, keep_drop, BATCH_SIZE, EPOCHS, OPT, LOSS, img_width, img_height, train_num,
                            test_num, data_root_path, train_path, train_txt, x_train_savepath, y_train_savepath,
                            test_path,
                            test_txt, x_test_savepath, y_test_savepath, y_list):
        # 生成/导入训练集
        if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and os.path.exists(
                x_test_savepath) and os.path.exists(y_test_savepath):
            print('-------------Load Datasets-----------------')
            self.x_train = np.load(x_train_savepath)
            self.y_train = np.load(y_train_savepath)
            self.x_test = np.load(x_test_savepath)
            self.y_test = np.load(y_test_savepath)
            print('x_train.shape=', self.x_train.shape)
            print('y_train.shape=', self.y_train.shape)
            self.y_train = self.label_one_hot(self.y_train)
            self.y_test = self.label_one_hot(self.y_test)
        else:
            print('-------------Generate Datasets-----------------')
            # 生成图片
            self.datasets_generate(train_path, train_txt, train_num)
            self.datasets_generate(test_path, test_txt, test_num)
            # 生成标签
            self.delete_illegal_data(train_path, train_txt)
            self.delete_illegal_data(train_path, train_txt)
            # 保存为np数组
            self.x_train, self.y_train = self.data_arr_generate(self, train_path, train_txt)
            self.x_test, self.y_test = self.data_arr_generate(self, test_path, test_txt)
            print('-------------Save Datasets-----------------')
            # 保存为npy文件
            np.save(x_train_savepath, self.x_train)
            np.save(y_train_savepath, self.y_train)
            np.save(x_test_savepath, self.x_test)
            np.save(y_test_savepath, self.y_test)
            self.y_train = self.label_one_hot(self.y_train)
            self.y_test = self.label_one_hot(self.y_test)

        super(train_model, self).__init__(keep_drop)

        self.inputs = Input(shape=(self.x_train.shape[1], self.x_train.shape[2], 1), name="inputs")
        self.keep_drop = keep_drop
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.OPT = OPT
        self.LOSS = LOSS
        self.img_width = img_width
        self.img_height = img_height
        self.train_num = train_num
        self.test_num = test_num
        self.keep_drop = keep_drop
        self.data_root_path = data_root_path
        self.train_path = train_path
        self.train_txt = train_txt
        self.x_train_savepath = x_train_savepath
        self.y_train_savepath = y_train_savepath
        self.test_path = test_path
        self.test_txt = test_txt
        self.x_test_savepath = x_test_savepath
        self.y_test_savepath = y_test_savepath
        self.y_list = y_list

    # 生成训练集原始图片
    @staticmethod
    def datasets_generate(img_path,txt_path,n):
        captcha_array = list('0123456789abcdefghijkemnopqrstuvwxyz')
        captcha_size = 4
        try:
            shutil.rmtree(img_path)
            os.remove(txt_path)
        except:
            pass
        os.mkdir(img_path)

        for i in range(n):
            image = ImageCaptcha()
            image_text = ''.join(random.sample(captcha_array,captcha_size))
            image_path = f'{img_path}/{i}_{image_text}.png'
            image.write(image_text,image_path)

    # 重写文本文件，删除非法图片
    @staticmethod
    def delete_illegal_data(train_path, train_txt):
        print('----------------------  开始删除非法数据  --------------------------')
        folder_path = train_path  # 文件夹路径
        extensions = ('*.jpg', '*.jpeg', '*.png')  # 支持的图片格式

        # 清空文本
        file = open(train_txt, 'w')
        file.write('')
        file.close()

        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))

        # 重写文本文件，删除非法图片
        for image_file in image_files:
            value = image_file.split('\\')
            xxx = value[1].split('_')
            yyy = xxx[1].split('.')
            if len(yyy[0]) == 4:
                file = open(train_txt, 'a')
                file.write(f'{value[1]} {yyy[0]}\n')
                file.close()
            else:
                print('value[1]=', value[1])
                print('xxx=', xxx)
                print('yyy=', yyy)
                os.remove(image_file)
        print('----------------------  结束删除非法数据  --------------------------')

    # 生成输入特征和标签
    @staticmethod
    def data_arr_generate(self,img_path, txt_path):
        f = open(txt_path, 'r')  # 以只读形式打开txt文件
        contents = f.readlines()  # 读取文件中所有行
        f.close()  # 关闭txt文件
        x, y_ = [], []  # 建立空列表
        for content in contents:  # 逐行取出
            value = content.split()  # 以空格分开，图片路径为value[0] , 标签为value[1] , 存入列表
            img_one_path = img_path +'/'+ value[0]  # 拼出图片路径和文件名
            img = Image.open(img_one_path)  # 读入图片
            # 灰度化
            img = tf.image.rgb_to_grayscale(img)
            img = np.array(img)  # 图片变为8位宽灰度值的np.array格式
            # 二值化
            for i in img:
                for j in i:
                    if j[0] < i.mean():
                        j[0] = 0
                    else:
                        j[0] = 255
            # 归一化
            for i in img:
                for j in i:
                    if j[0] == 255:
                        j[0] = 1
            x.append(img)  # 归一化后的数据，贴到列表x
            yy = []
            for i in value[1]:
                yy.append(self.y_list[i])
            y_.append(yy)   # 标签贴到列表y_
        x = np.array(x)  # 变为np.array格式
        y_ = np.array(y_)
        return x, y_  # 返回输入特征x，返回标签y_

    # 对验证码中每个字符进行one-hot编码
    @staticmethod
    def label_one_hot(labels):
        hoted_label = []
        i = 0
        for label in labels:
            hoted_label.append(tf.one_hot(label, depth=36))
            i += 1
        hoted_label = np.array(hoted_label)
        hoted_label = hoted_label.reshape(hoted_label.shape[0], hoted_label.shape[1] * hoted_label.shape[2])
        return hoted_label

    # 训练模型并保存模型
    def train(self,load=True):
        # 打乱数据集
        seed = random.randint(0,100)
        np.random.seed(seed)
        self.x_train = np.random.permutation(self.x_train)
        np.random.seed(seed)
        self.y_train = np.random.permutation(self.y_train)

        inputs = Input(shape = (self.x_train.shape[1],self.x_train.shape[2],1), name = "inputs")
        outs = self.run_model(inputs)
        model = Model(inputs, outs)
        self.model = model
        model.compile(optimizer=self.OPT, loss=self.LOSS, metrics=['accuracy'])

        # 载入之前保存的参数
        checkpoint_save_path = "./checkpoint/Baseline.ckpt"
        if load:
            if os.path.exists(checkpoint_save_path + '.index'):
                print('---------------- 开始加载参数 ------------------')
                model.load_weights(checkpoint_save_path)
                print('---------------- 参数加载完毕 ------------------')
        else:
            print('------------- not load model ----------------')
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path, save_weights_only=True,
                                                         save_best_only=True)

        history = model.fit(self.x_train, self.y_train, batch_size=self.BATCH_SIZE, epochs=self.EPOCHS, validation_data=(self.x_test, self.y_test),
                            validation_freq=1, callbacks=[cp_callback])
        self.history = history

        # 保存模型
        print('---------------- 开始保存模型 ------------------')
        model.save('./checkpoint/model.h5')
        print('---------------- 模型保存成功 ------------------')


        model.summary()

        '''
        这里记录参数为文本形式，让我们可以看到具体数值
        全部写出来需要时间太久，需要看参数的时候启用，作用仅使参数可见
        参数和模型，默认保存在checkpoint文件夹内
        
        print('--------------- 开始记录参数 ------------------')
        # 记录参数
        file = open('./weights.txt', 'w')
        for v in model.trainable_variables:
            file.write(str(v.name) + '\n')
            file.write(str(v.shape) + '\n')
            file.write(str(v.numpy()) + '\n')
        file.close()
        print('--------------- 参数记录完成 ------------------')
        '''


        return history

    # acc、loss可视化记
    def show_acc_loss(self):
        try:
            # 显示训练集和验证集的acc和loss曲线
            acc = self.history.history['accuracy']
            val_acc = self.history.history['val_accuracy']
            loss = self.history.history['loss']
            val_loss = self.history.history['val_loss']

            plt.subplot(1, 2, 1)
            plt.plot(acc, label='Training Accuracy')
            plt.plot(val_acc, label='Validation Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(loss, label='Training Loss')
            plt.plot(val_loss, label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.show()
        except:
            pass

    # 随机测试
    def predict_one(self):
        print('--------------predict part start--------------')
        n = random.randint(0,self.y_test.shape[0])
        x = self.x_test[n].reshape(1, self.img_height, self.img_width, 1)
        y = self.model.predict(x)

        def hoted_to_text(hoted):
            hoted = np.array(hoted)
            if hoted.shape[0] == 1:
                hoted = hoted.reshape(4, int(hoted.shape[1] / 4))
            else:
                hoted = hoted.reshape(4, int(hoted.shape[0] / 4))
            text = []
            for i in hoted:
                index = np.argmax(i)
                for k, v in self.y_list.items():
                    if v == index:
                        text.append(k)
            return text

        y = hoted_to_text(y)
        label_y = hoted_to_text(self.y_test[n])
        print('predict_y=', y)
        print('label_y=', label_y)
        print('--------------predict part end--------------')



def main():
    # 配置参数
    start_time = time.time()

    np.set_printoptions(threshold=np.inf)

    BATCH_SIZE = 128
    EPOCHS = 1
    OPT = 'adam'
    LOSS = 'binary_crossentropy'
    img_width = 200
    img_height = 50
    train_num = 4400
    test_num = 990
    keep_drop = 0.2

    data_root_path = './datasets'
    train_path = f'{data_root_path}/train_img'
    train_txt = f'{data_root_path}/train_label.txt'
    x_train_savepath = f'{data_root_path}/train_img.npy'
    y_train_savepath = f'{data_root_path}/train_label.npy'

    test_path = f'{data_root_path}/test_img'
    test_txt = f'{data_root_path}/test_label.txt'
    x_test_savepath = f'{data_root_path}/test_img.npy'
    y_test_savepath = f'{data_root_path}/test_label.npy'

    y_list = {'0': 1, '1': 2, '2': 3, '3': 4, '4': 5, '5': 6, '6': 7, '7': 8, '8': 9, '9': 10, 'a': 11, 'b': 12,
              'c': 13, 'd': 14, 'e': 15, 'f': 16, 'g': 17, 'h': 18, 'i': 19, 'j': 20, 'k': 21, 'l': 22, 'm': 23,
              'n': 24, 'o': 25, 'p': 26, 'q': 27, 'r': 28, 's': 29, 't': 30, 'u': 31, 'v': 32, 'w': 33, 'x': 34,
              'y': 35, 'z': 36}

    # 开始训练
    train_obj = train_model(keep_drop, BATCH_SIZE, EPOCHS, OPT, LOSS, img_width, img_height, train_num,
                            test_num, data_root_path, train_path, train_txt, x_train_savepath, y_train_savepath,
                            test_path,
                            test_txt, x_test_savepath, y_test_savepath, y_list)
    history = train_obj.train()
    train_obj.predict_one()

    # 记录训练时间等参数，保存
    end_time = time.time()
    file = open('./time_record.txt', 'a')
    file.write(f'\n=====================================================\n此次运行的时间为：{int(end_time-start_time)}s\n运行开始时间：{start_time}\n运行结束时间：{end_time}\n总数据量为：{train_num}\n循环次数为：{EPOCHS}\n准确率为：{history.history["accuracy"]}\n=====================================================\n')
    file.close()
    plt.show()



if __name__ == '__main__':
    main()










