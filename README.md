## StarGAN - Official PyTorch Implementation
<p align="center"><img width="100%" src="jpg/main.jpg" /></p>
该代码提供了如下论文的官方实现：
> **StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation**<br>
> [Yunjey Choi](https://github.com/yunjey)<sup>1,2</sup>, [Minje Choi](https://github.com/mjc92)<sup>1,2</sup>, [Munyoung Kim](https://www.facebook.com/munyoung.kim.1291)<sup>2,3</sup>, [Jung-Woo Ha](https://www.facebook.com/jungwoo.ha.921)<sup>2</sup>, [Sung Kim](https://www.cse.ust.hk/~hunkim/)<sup>2,4</sup>, [Jaegul Choo](https://sites.google.com/site/jaegulchoo/)<sup>1,2</sup>    <br/>
> <sup>1</sup>Korea University, <sup>2</sup>Clova AI Research, NAVER Corp. <br>
> <sup>3</sup>The College of New Jersey, <sup>4</sup>Hong Kong University of Science and Technology <br/>
> https://arxiv.org/abs/1711.09020 <br>
>
> **Abstract:** *Recent studies have shown remarkable success in image-to-image translation for two domains. However, existing approaches have limited scalability and robustness in handling more than two domains, since different models should be built independently for every pair of image domains. To address this limitation, we propose StarGAN, a novel and scalable approach that can perform image-to-image translations for multiple domains using only a single model. Such a unified model architecture of StarGAN allows simultaneous training of multiple datasets with different domains within a single network. This leads to StarGAN's superior quality of translated images compared to existing models as well as the novel capability of flexibly translating an input image to any desired target domain. We empirically demonstrate the effectiveness of our approach on a facial attribute transfer and a facial expression synthesis tasks.*

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
$ python main.py --mode test --dataset CelebA --image_size 128 --c_dim 5 \
                 --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young \
                 --model_save_dir='stargan_celeba_128/models' \
                 --result_dir='stargan_celeba_128/results'
```