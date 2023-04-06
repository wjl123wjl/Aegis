## Package required

    install python 3.6.9, pytorch 1.7.0, torchvision 0.8.1, tensorboardX 2.5, matplotlib 3.3.4, tqdm 4.60.0, pandas 1.1.5, numpy 1.18.5.


## Train Models
In each dataset and model, first train the base model, than the enhanced model.

### CIFAR10-resne32
    cd cifar10/resnet32
    mkdir data
    sh train_CIFAR.sh
    sh train_finetune_branch.sh
    
### CIFAR10-vgg16
    cd cifar10/vgg16
    mkdir data
    sh train_CIFAR.sh
    sh train_finetune_branch.sh
    
### CIFAR100-resne32
    cd cifar100/resnet32
    mkdir data
    sh train_CIFAR.sh
    sh train_finetune_branch.sh
    
### CIFAR100-vgg16
    cd cifar100/vgg16
    mkdir data
    sh train_CIFAR.sh
    sh train_finetune_branch.sh

### stl10-resne32
    cd stl10/resnet32
    mkdir data
    sh train_STL.sh
    sh train_finetune_branch.sh
    
### stl10-vgg16
    cd stl10/vgg16
    mkdir data
    sh train_STL.sh
    sh train_finetune_branch.sh
    
### tinyimagenet-resne32
    cd tinyimagenet/resnet32
    mkdir data
    sh train_tinyimagenet.sh
    sh train_finetune_branch.sh
    
### tinyimagenet-vgg16
    cd tinyimagenet/vgg16
    mkdir data
    sh train_tinyimagenet.sh
    sh train_finetune_branch.sh


## TBT Attacks

First enter a folder to attack the target model, e.g., cd ./Aegis/TBT/resnet32-cifar10/

### Run non-adaptove attacks
    python3 TBT_noadaptive.py
   
### Run adaptive attacks
    python3 TBT_adaptive.py
    

### Results
    The attack success rate under our defense are shown on the screen.


## TA-LBF Attacks

### DATA
    ./utils.py modify line23 to specify positions to save data. When you first run attack scripts. We will automatically download CIFAR-10, CIFAR-100, STL-10. But you need to download tiny-ImageNet by yourself.



### Run non-adaptove attacks:
    cd TA-LBF/non-adaptive
    Then run attacks. for example, sh ./attack_reproduce_k=50_vgg16_cifar10.sh
    All sh files are listed as follows.
    cifar10 resnet32: attack_reproduce_k=50_resnet32_cifar10.sh
    cifar100 resnet32: attack_reproduce_k=50_resnet32_cifar100.sh
    stl10 resnet32: attack_reproduce_k=50_resnet32_stl10.sh
    tinyimagenet resnet32: attack_reproduce_k=50_resnet32_tinyimagenet.sh
    cifar10 vgg16: ./attack_reproduce_k=50_vgg16_cifar10.sh
    cifar100 vgg16: ./attack_reproduce_k=50_vgg16_cifar100.sh
    stl10 vgg16: ./attack_reproduce_k=50_vgg16_stl10.sh
    tinyimagenet vgg16: ./attack_reproduce_k=50_vgg16_tinyimagenet.sh

### Run adaptove attacks:
    cd TA-LBF/adaptive
    Then run attacks. for example, sh ./attack_reproduce_k=50_vgg16_cifar10.sh
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
    The attack success rate under our defense are shown on the screen.
    
## ProFlip Attacks
    First enter a folder to attack the target model, e.g., cd ./Aegis/ProFlip/resnet32-cifar10/

### Run non-adaptove attacks
    Run the instruction to generate a trigger: python3 trigger_nonadaptive.py
    Then, attack: python3 CSB_nonadaptive.py
   
### Run adaptive attacks
    Run the instruction to generate a trigger: python3 trigger_adaptive.py
    Then, attack: python3 CSB_adaptive.py
    

### Results
    The attack success rate under our defense are shown on the screen.

