
CKPT=AlexNet_0526_30

python3 Train.py \
    --checkpoint_dir=checkpoints/${CKPT} \
    --num_epochs=30 \
    --checkpoint_interval=6 \
    --model_name=alexnet \
    | tee logs/${CKPT}.txt
