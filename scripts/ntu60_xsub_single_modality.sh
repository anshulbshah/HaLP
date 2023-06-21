python pretrain_moco_single_modality.py --batch-size=64 --contrast-k=16384 \
        --contrast-t=0.07 --epochs=451 --lr=0.01 --method=single_modality_halp \
        --num_closest_to_ignore_positives=20 --num_positives=100 --num_prototypes=20 \
        --mu=2 --pre-dataset=ntu60 --protocol=cross_subject \
        --queue_els_for_prototypes=256 \
        --resume=pretrained_checkpoints/ntu60_xsub_sm/checkpoint_0200.pth.tar \
        --save_every=100 --schedule=351 --skeleton-representation=graph-based \
        --skip_closest_positives=1 --student-t=0.1 --teacher-t=0.05 \
        --topk=8192 --update_prototypes_every=10 --mlp

python action_classification.py --batch-size=64 --finetune-dataset=ntu60 \
        --finetune-skeleton-representation=graph-based --lr=0.005 \
        --pretrained=checkpoint_0450.pth.tar \
        --protocol=cross_subject