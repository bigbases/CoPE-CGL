seed=1042
repeat=3
epoch=500
device=cuda:0

# CoraFull
dataset=corafull

for method in cope_gl tpp taam
do
python train_transductive.py \
--seed $seed \
--repeat $repeat \
--cls-epoch $epoch \
--cgl-method $method \
--data-dir ./data \
--device $device \
--result-path ./results \
--dataset-name $dataset
done

dataset=arxiv

for method in cope_gl tpp taam
do
python train_transductive.py \
--seed $seed \
--repeat $repeat \
--cls-epoch $epoch \
--cgl-method $method \
--data-dir ./data \
--device $device \
--result-path ./results \
--dataset-name $dataset
done


dataset=reddit

for method in cope_gl tpp taam
do
python train_transductive.py \
--seed $seed \
--repeat $repeat \
--cls-epoch $epoch \
--cgl-method $method \
--data-dir ./data \
--device $device \
--result-path ./results \
--dataset-name $dataset
done


dataset=products

for method in cope_gl tpp taam
do
python train_transductive.py \
--seed $seed \
--repeat $repeat \
--cls-epoch $epoch \
--cgl-method $method \
--data-dir ./data \
--device $device \
--result-path ./results \
--dataset-name $dataset
done
