data_root=PATH/TO/DATASET
DESC_PATH=./descriptions/image_datasets
gpu=$1
testset=$2

# number of descriptions per class
N_DESC=50

CUDA_VISIBLE_DEVICES=$gpu python -u clip.py \
        --test_set ${testset} \
        --arch ViT-B/16 \
        --descriptor_path ${DESC_PATH} \
        --num_descriptor ${N_DESC} \
        --data ${data_root} \
        --resolution 224
