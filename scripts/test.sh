
CKPT="AlexNet0526_50"

python3 Test.py --checkpoint_path=checkpoints/${CKPT} \
                --result_path=checkpoints/${CKPT} \
                # --checkpoint_epoch=20 \
