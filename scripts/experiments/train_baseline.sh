
CKPT=AlexNet_0526_baseline

CUDA_VISIBLE_DEVICES=0 python3 Train.py \
    --checkpoint_dir=checkpoints/${CKPT} \
    --num_epochs=30 \
    --checkpoint_interval=6 \
    --model_name=alexnet \
    --scheduler=none \
    | tee logs/${CKPT}.txt
