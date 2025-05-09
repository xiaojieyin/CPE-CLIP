data_root=PATH/TO/DATASET

gpu=$1
testset=$2

CUDA_VISIBLE_DEVICES=$gpu python -u ./eva02-clip.py ${data_root} \
        --test_set ${testset} \
        --descriptor_path ./descriptions/image_datasets \
        --batch-size 100 \
        --num_descriptor 50