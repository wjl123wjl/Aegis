
# path of cifar10 data on yout device
# cifar_root = "./data/cifar10"


# path of the quantized model
model_root =  { "cifar10": { "vgg16": "../../cifar10/vgg16/safe_finetune", "resnet32": "../../cifar10/resnet32/safe_finetune", "resnet20":"./save/cifar10_resnet20_quan_200_SGD_binarized"},
                "cifar100": { "vgg16": "../../cifar100/vgg16/safe_finetune","resnet32": "../../cifar100/resnet32/safe_finetune"},
                "stl10": { "vgg16": "../../stl10/vgg16/safe_finetune","resnet32": "../../stl10/resnet32/safe_finetune"},
                "tinyimagenet": { "vgg16": "../../tinyimagenet/vgg16/safe_finetune","resnet32": "../../tinyimagenet/resnet32/safe_finetune"},
                }

num_classes = {
                "cifar10": 10,
                "cifar100": 100,
                "stl10":10,
                "tinyimagenet": 200,
}
# The 1,000 attacked samples used in our experiments
# Format: [ [target-class, sample-index],
# 	      [target-class, sample-index],
# 	       ...
# 	      [target-class, sample-index] ]
info_root = { "cifar10": "cifar10_attack_info.txt",
            "cifar100": "cifar100_attack_info.txt",
            "tinyimagenet": "tinyimagenet_attack_info.txt",
            "stl10": "stl10_attack_info.txt"
            }

# result_root = {"cifar10" : {"vgg16": "Result_TA-LBF_c10.txt", "resnet32": "Result_TA-LBF_c10_resnet32.txt"},
#                 "cifar100" :{"vgg16": "Result_TA-LBF_c100.txt", "resnet32": "Result_TA-LBF_c100_resnet32.txt"},
#                 "tinyimagenet" : {"vgg16": "Result_TA-LBF_ti.txt", "resnet32": "Result_TA-LBF_ti_resnet32.txt"},
#                 "stl10" : {"vgg16": "Result_TA-LBF_stl10.txt", "resnet32": "Result_TA-LBF_c100_stl10.txt"},
# }

num_branch = {"vgg16": 15,
                "resnet32": 16,
} 

is_modify = {
    'cifar10':{'vgg16': 0, 'resnet32': 1},
    'cifar100':{'vgg16': 0, 'resnet32': 1},
    'stl10':{'vgg16': 1, 'resnet32': 1},
    'tinyimagenet':{'vgg16': 0, 'resnet32': 1}
}

escape_num = {
    'cifar10':{'vgg16': 0, 'resnet32': 4},
    'cifar100':{'vgg16': 0, 'resnet32': 0},
    'stl10':{'vgg16': 0, 'resnet32': 2},
    'tinyimagenet':{'vgg16': 0, 'resnet32': 0}
}

mask_num = {
    'cifar10':{'vgg16': 3, 'resnet32': 3},
    'cifar100':{'vgg16': 3, 'resnet32': 5},
    'stl10':{'vgg16': 6, 'resnet32': 5},
    'tinyimagenet':{'vgg16': 6, 'resnet32': 8}
}

confidence_threshold = {
    'cifar10':{'vgg16': 0.95, 'resnet32': 0.95},
    'cifar100':{'vgg16': 0.8, 'resnet32': 0.9},
    'stl10':{'vgg16': 0.95, 'resnet32': 0.9},
    'tinyimagenet':{'vgg16': 0.95, 'resnet32': 0.9}

}

useful_linear = {
    'vgg16': [1,3,5,7,9,11,13,15,16,17,18,19,20,21,22],
    'vgg161':[1,3,5,7,9,11,13,15,17,19,21,23,24,25,26],   # only for stl10
    'resnet32':[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,30],
    # 'cifar10':{'vgg16': tuple([1,3,5,7,9,11,13,15,16,17,18,19,20,21,22]),},
    # 'cifar100':{'vgg16': tuple([1,3,5,7,9,11,13,15,16,17,18,19,20,21,22]),},
    # 'stl10':{'vgg16': '',},
    # 'tinyimagenet':{'vgg16': ''}
}
choose_branch={
    'cifar100':{'vgg16':[[7,12,13],[6,7,12,13,14],[6,7,8,11,12,13,14],[5,6,7,8,9,11,12,13,14],[7],[4,5,6,7,8,9,10,11,12,13,14],[2,3,4,5,6,7,8,9,10,11,12,13,14]]},
    'cifar10':{'vgg16':[[4,5,6],[2,3,4,5,6],[2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8,9]]},

}