python pretrain_moco_single_modality.py --batch-size=64 --contrast-k=16384 \
        --contrast-t=0.07 --epochs=1001 --lambda_pos=type4_auto-0.8 \
        --lr=0.01 --method=single_modality_halp --num_closest_to_ignore_positives=20 \
        --num_positives=100 --num_prototypes=20 --mu=1 \
        --pre-dataset=pku_v2 --protocol=cross_subject \
        --queue_els_for_prototypes=256 \
        --resume=pretrained_checkpoints/pku_xsub_sm/checkpoint_0500.pth.tar \
        --save_every=100 --schedule=801 --skeleton-representation=graph-based \
        --skip_closest_positives=1 --student-t=0.1 --teacher-t=0.05 --topk=8192 \
        --update_prototypes_every=10 --mlp

python action_classification.py --batch-size=64 --finetune-dataset=pku_v2 \
        --finetune-skeleton-representation=graph-based --lr=0.001 \
        --pretrained=checkpoint_1000.pth.tar \
        --protocol=cross_subject