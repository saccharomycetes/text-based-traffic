# CUDA_VISIBLE_DEVICES=0 python train.py \
# --data bdd \
# --test_type e \
# --eval_batch_size 16 \
# --train_batch_size 8 \
# --acc_step 4 \
# --num_epoches 10 &

# CUDA_VISIBLE_DEVICES=1 python train.py \
# --data bdd \
# --test_type c \
# --eval_batch_size 16 \
# --train_batch_size 8 \
# --acc_step 4 \
# --num_epoches 10 &

# CUDA_VISIBLE_DEVICES=2 python train.py \
# --data hdt \
# --test_type 2 \
# --eval_batch_size 16 \
# --train_batch_size 4 \
# --acc_step 8 \
# --num_epoches 10

CUDA_VISIBLE_DEVICES=0 python train.py \
--data hdt \
--test_type 3 \
--eval_batch_size 16 \
--train_batch_size 4 \
--acc_step 8 \
--num_epoches 10

CUDA_VISIBLE_DEVICES=0 python train.py \
--data hdt \
--test_type 4 \
--eval_batch_size 16 \
--train_batch_size 4 \
--acc_step 8 \
--num_epoches 10

# CUDA_VISIBLE_DEVICES=0 python train.py \
# --data hdt \
# --test_type 5 \
# --eval_batch_size 16 \
# --train_batch_size 4 \
# --acc_step 8 \
# --num_epoches 10