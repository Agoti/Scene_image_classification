
CKPT=AlexNet_0527_decay2

python3 Train.py \
    --checkpoint_dir=checkpoints/${CKPT} \
    --num_epochs=30 \
    --checkpoint_interval=6 \
    --model_name=alexnet_norm_v2 \
    --scheduler=step \
    --step_size=6 \
    --gamma=0.5 \
    --weight_decay=0.01 \
    | tee logs/${CKPT}.txt
