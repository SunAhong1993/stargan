## StarGAN - PaddlePaddle实现
<p align="center"><img width="100%" src="jpg/main.jpg" /></p>


该代码提供了如下论文的官方实现：    
StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation

## 依赖
* [Python 3.5+](https://www.continuum.io/downloads)
* [PaddlePaddle 2.0.1+](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/windows-pip.html)
* [X2Paddle 1.2+](https://github.com/PaddlePaddle/X2Paddle)

## 下载数据集
下载CelebA数据集:
```bash
git clone https://github.com/yunjey/StarGAN.git
cd StarGAN/
bash download.sh celeba
```

为了下载RaFD数据集, 你必须从[the Radboud Faces Database website](http://www.socsci.ru.nl:8180/RaFD2/RaFD?p=main)请求访问数据集，而创建的文件夹构造必须像[这里](./jpg/RaFD.md)所描述的一样。

## 训练网络
为了在CelebA数据集上训练，可以运行如下脚本，可以在[这里](./jpg/CelebA.md)查看CelebA数据集可供选择的属性，如果修改了 `selected_attrs`这一参数, 对应地，应该修改`c_dim` 这一参数。

```bash
# 在CelebA数据集上训练StarGAN
python main.py --mode train --dataset CelebA --image_size 128 --c_dim 5 \
               --sample_dir stargan_celeba/samples --log_dir stargan_celeba/logs \
               --model_save_dir stargan_celeba/models --result_dir stargan_celeba/results \
               --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young

# 在CelebA数据集上测试StarGAN
python main.py --mode test --dataset CelebA --image_size 128 --c_dim 5 \
               --sample_dir stargan_celeba/samples --log_dir stargan_celeba/logs \
               --model_save_dir stargan_celeba/models --result_dir stargan_celeba/results \
               --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young
```

在RaFD数据集上训练StarGAN：

```bash
# 在RaFD数据集上训练StarGAN
python main.py --mode train --dataset RaFD --image_size 128 \
               --c_dim 8 --rafd_image_dir data/RaFD/train \
               --sample_dir stargan_rafd/samples --log_dir stargan_rafd/logs \
               --model_save_dir stargan_rafd/models --result_dir stargan_rafd/results

# 在RaFD数据集上测试tarGAN
python main.py --mode test --dataset RaFD --image_size 128 \
               --c_dim 8 --rafd_image_dir data/RaFD/test \
               --sample_dir stargan_rafd/samples --log_dir stargan_rafd/logs \
               --model_save_dir stargan_rafd/models --result_dir stargan_rafd/results
```

在CelebA和RaFD数据集上训练StarGAN：

```bash
# 在CelebA和RaFD数据集上训练StarGAN
python main.py --mode=train --dataset Both --image_size 256 --c_dim 5 --c2_dim 8 \
               --sample_dir stargan_both/samples --log_dir stargan_both/logs \
               --model_save_dir stargan_both/models --result_dir stargan_both/results

# 在CelebA和RaFD数据集上测试StarGAN
python main.py --mode test --dataset Both --image_size 256 --c_dim 5 --c2_dim 8 \
               --sample_dir stargan_both/samples --log_dir stargan_both/logs \
               --model_save_dir stargan_both/models --result_dir stargan_both/results
```

To train StarGAN on your own dataset, create a folder structure in the same format as [RaFD](https://github.com/yunjey/StarGAN/blob/master/jpg/RaFD.md) and run the command:

```bash
# Train StarGAN on custom datasets
python main.py --mode train --dataset RaFD --rafd_crop_size CROP_SIZE --image_size IMG_SIZE \
               --c_dim LABEL_DIM --rafd_image_dir TRAIN_IMG_DIR \
               --sample_dir stargan_custom/samples --log_dir stargan_custom/logs \
               --model_save_dir stargan_custom/models --result_dir stargan_custom/results

# Test StarGAN on custom datasets
python main.py --mode test --dataset RaFD --rafd_crop_size CROP_SIZE --image_size IMG_SIZE \
               --c_dim LABEL_DIM --rafd_image_dir TEST_IMG_DIR \
               --sample_dir stargan_custom/samples --log_dir stargan_custom/logs \
               --model_save_dir stargan_custom/models --result_dir stargan_custom/results
```


## 使用预训练模型
下载预训练模型：
```bash
mkdir ./stargan_celeba_128/models
cd ./stargan_celeba_128/models
wget https://x2paddle.bj.bcebos.com/vision/StatGAN/200000-G.pdiparams
wget https://x2paddle.bj.bcebos.com/vision/StatGAN/200000-D.pdiparams
```
为了使用预训练模型转换图像，可以运行如下脚本进行评估，转换结果将保存至`./stargan_celeba_128/results`。

```bash
python main.py --mode test --dataset CelebA --image_size 128 --c_dim 5 \
                 --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young \
                 --model_save_dir='stargan_celeba_128/models' \
                 --result_dir='stargan_celeba_128/results'
```
