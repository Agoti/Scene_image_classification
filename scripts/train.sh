
CKPT=AlexNetNorm_step_decay5e-4_0526_30

python3 Train.py \
    --checkpoint_dir=checkpoints/${CKPT} \
    --checkpoint_interval=6 \
    --num_epochs=30 \
    --scheduler=step \
    --step_size=10 \
    --gamma=0.3 \
    --weight_decay=0.0005 \
    --model_name=alexnet_norm \
    | tee logs/${CKPT}.txt
