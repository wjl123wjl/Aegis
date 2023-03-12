## TA-LBF Attacks

### DATA
    ./utils.py modify line23 to specify positions to save data. When you first run attack scripts. We will automatically download CIFAR-10, CIFAR-100, STL-10. But you need to download tiny-ImageNet by yourself.
    

### Run non-adaptove attacks:
    For example, sh ./attack_reproduce_k=50_vgg16_cifar10.sh
    All sh files are listed as follows.
    cifar10 resnet32: attack_reproduce_k=50_resnet32_cifar10.sh
    cifar100 resnet32: attack_reproduce_k=50_resnet32_cifar100.sh
    stl10 resnet32: attack_reproduce_k=50_resnet32_stl10.sh
    tinyimagenet resnet32: attack_reproduce_k=50_resnet32_tinyimagenet.sh
    cifar10 vgg16: ./attack_reproduce_k=50_vgg16_cifar10.sh
    cifar100 vgg16: ./attack_reproduce_k=50_vgg16_cifar100.sh
    stl10 vgg16: ./attack_reproduce_k=50_vgg16_stl10.sh
    tinyimagenet vgg16: ./attack_reproduce_k=50_vgg16_tinyimagenet.sh
    

### Results
    Results are shown on the screen.
