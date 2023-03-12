## Note 

### 数据
    ./utils.py line23 改成存储data文件夹的根目录。
### 模型
    ./config.py “model_root”中存储4个数据集、2个模型的模型文件（共8个）。

### 运行命令
    cifar10 vgg16: ./attack_reproduce_k=50_vgg16_cifar10.sh
    cifar100 vgg16: ./attack_reproduce_k=50_vgg16_cifar100.sh
    stl10 vgg16: ./attack_reproduce_k=50_vgg16_stl10.sh
    tinyimagenet vgg16: ./attack_reproduce_k=50_vgg16_tinyimagenet.sh
    cifar10 resnet32: attack_reproduce_k=50_resnet32_cifar10.sh
    cifar100 resnet32: attack_reproduce_k=50_resnet32_cifar100.sh
    stl10 resnet32: attack_reproduce_k=50_resnet32_stl10.sh
    tinyimagenet resnet32: attack_reproduce_k=50_resnet32_tinyimagenet.sh

### 输出样例
    cifar10 vgg16的输出样例: 
    TA-LBF_OUR_AD_c10_vgg16_k=50.out
