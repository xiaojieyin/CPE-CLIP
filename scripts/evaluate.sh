data_root=PATH/TO/DATASET
DESC_PATH=./descriptions/image_datasets
gpu=$1
testsets=oxford_flowers/dtd/oxford_pets/stanford_cars/ucf101/caltech101/food101/sun397/fgvc_aircraft/eurosat

# number of descriptions per class
N_DESC=50
BS=100

seed=0

CUDA_VISIBLE_DEVICES=$gpu python -u evaluate.py \
        --test_set ${testsets} \
        --arch ViT-B/16 \
        --descriptor_path ${DESC_PATH} \
        --num_descriptor ${N_DESC} \
        --data ${data_root}  \
        --batch-size ${BS}  \
        --seed ${seed}
