
CKPT=AlexNet_0526_norm

CUDA_VISIBLE_DEVICES=2 python3 Train.py \
    --checkpoint_dir=checkpoints/${CKPT} \
    --num_epochs=30 \
    --checkpoint_interval=6 \
    --model_name=alexnet_norm \
    --scheduler=none \
    | tee logs/${CKPT}.txt
