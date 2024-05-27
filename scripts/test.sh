
CKPT="AlexNet_0527_aug"

python3 Test.py --checkpoint_path=checkpoints/${CKPT} \
                --result_path=checkpoints/${CKPT} \
                # --checkpoint_epoch=12 \
