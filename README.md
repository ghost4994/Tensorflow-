# Tensorflow-
卷积神经网络数字字母验证码识别

自行创建目录./datasets
自行创建目录./checkpoint

1.训练模型运行train_model.py即可

2.没有数据集会自动从captcha库获取

3.导入自己的数据集请将numpy数组文件train_img.npy、train_label.npy、test_img.npy、test_label.npy放入./datasets中

  数据格式为
  
  img:(n,height,width,1) 只有0，1组成

  label:(n,4)  比如：[ 8 17 24 24]，0-36代表0-z 
  
