#!/usr/bin/env sh

############### Host   ##############################
HOST=$(hostname)
echo "Current host is: $HOST"

# Automatic check the host and configure
case $HOST in
"alpha")
    PYTHON="/home/elliot/anaconda3/envs/pytorch041/bin/python" # python environment path
    TENSORBOARD='/home/elliot/anaconda3/envs/pytorch041/bin/tensorboard' # tensorboard environment path
    data_path='/home/elliot/data/pytorch/cifar10'
    ;;
esac

DATE=`date +%Y-%m-%d`



############### Configurations ########################
enable_tb_display=false # enable tensorboard display
model=resnet32_quan
dataset=cifar100
test_batch_size=128

label_info=BFA_defense_test

attack_sample_size=128 # number of data used for BFA
n_iter=1000 # number of iteration to perform BFA
k_top=100 # only check k_top weights with top gradient ranking in each layer

save_path=./save/${dataset}_${model}_${label_info}
tb_path=${save_path}/tb_log  #tensorboard log path

# set the pretrained model path
pretrained_model=/home/wangjialai/copy_for_use/flip_attack/BFA/save/cifar10_resnet20_quan_200_SGD_binarized/model_best.pth.tar
PYTHON="/usr/bin/python3.6"
data_path='/home/wangjialai/copy_for_use/flip_attack/TBT-CVPR2020-master/data'
    


############### Neural network ############################
COUNTER=0
{
while [ $COUNTER -lt 1 ]; do
    $PYTHON main.py --dataset ${dataset} \
        --data_path ${data_path}   \
        --arch ${model} --save_path ${save_path}  \
        --test_batch_size ${test_batch_size} --workers 1 --ngpu 1 --gpu_id 3 \
        --print_freq 50 \
        --evaluate --resume ${pretrained_model} --fine_tune\
        --reset_weight --bfa --n_iter ${n_iter} \
        --attack_sample_size ${attack_sample_size} \
        --bfa_mydefense \

    #let COUNTER=COUNTER+1
    COUNTER=$((COUNTER+1))
done
} &
############## Tensorboard logging ##########################
{
if [ "$enable_tb_display" = true ]; then 
    sleep 30 
    wait
    $TENSORBOARD --logdir $tb_path  --port=6006
fi
} &
{
if [ "$enable_tb_display" = true ]; then
    sleep 45
    wait
    case $HOST in
    "Hydrogen")
        firefox http://0.0.0.0:6006/
        ;;
    "alpha")
        google-chrome http://0.0.0.0:6006/
        ;;
    esac
fi 
} &
wait