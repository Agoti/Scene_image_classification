
CKPT=AlexNet_0526_norm_step

CUDA_VISIBLE_DEVICES=1 python3 Train.py \
    --checkpoint_dir=checkpoints/${CKPT} \
    --num_epochs=30 \
    --checkpoint_interval=6 \
    --model_name=alexnet_norm \
    --scheduler=step \
    --step_size=6 \
    --gamma=0.5 \
    | tee logs/${CKPT}.txt
