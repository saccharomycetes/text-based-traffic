kgs=(roberta_large roberta_base t5_small t5_large t5_3b)
nlis=(roberta-large-mnli facebook/bart-large-mnli microsoft/deberta-large-mnli microsoft/deberta-v2-xlarge-mnli microsoft/deberta-v2-xxlarge-mnli)
bdds=(e c)
hdts=(2 3 4 5)
unifys=(0 1)

for model in ${kgs[@]}
    do
    for type in ${bdds[@]}
        do
        echo $model
        echo $type
        echo "bdd"
        python eval_kg.py --data bdd \
        --eval_batch_size 8 \
        --model_type $model \
        --test_type $type
        done
    done

for model in ${kgs[@]}
    do
    for type in ${hdts[@]}
        do
        echo $model
        echo $type
        echo "hdt"
        python eval_kg.py --data hdt \
        --eval_batch_size 8 \
        --model_type $model \
        --test_type $type
        done
    done

for model in ${nlis[@]}
    do
    for type in ${bdds[@]}
        do
        echo $model
        echo $type
        echo "bdd"
        python eval_nli.py --data bdd \
        --eval_batch_size 8 \
        --model_type $model \
        --test_type $type
        done
    done

for model in ${nlis[@]}
    do
    for type in ${hdts[@]}
        do
        echo $model
        echo $type
        echo "hdt"
        python eval_nli.py --data hdt \
        --eval_batch_size 8 \
        --model_type $model \
        --test_type $type
        done
    done

for model in ${unifys[@]}
    do
    for type in ${bdds[@]}
        do
        echo $model
        echo $type
        echo "bdd"
        python eval_unifiy.py --data bdd \
        --num_related $model \
        --test_type $type
        done
    done

for model in ${unifys[@]}
    do
    for type in ${hdts[@]}
        do
        echo $model
        echo $type
        echo "hdt"
        python eval_unifiy.py --data hdt \
        --num_related $model \
        --test_type $type
        done
    done