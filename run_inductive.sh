seed=1024
repeat=3
epoch=500
device=cuda:0

# CoraFull
dataset=kindle

for method in cope_gl tpp taam
do
python train_inductive.py \
--seed $seed \
--repeat $repeat \
--cls-epoch $epoch \
--cgl-method $method \
--data-dir ./data \
--device $device \
--result-path ./results \
--dataset-name $dataset
done

dataset=DBLP

for method in cope_gl tpp taam
do
python train_inductive.py \
--seed $seed \
--repeat $repeat \
--cls-epoch $epoch \
--cgl-method $method \
--data-dir ./data \
--device $device \
--result-path ./results \
--dataset-name $dataset
done


dataset=ACM

for method in cope_gl tpp taam
do
python train_inductive.py \
--seed $seed \
--repeat $repeat \
--cls-epoch $epoch \
--cgl-method $method \
--data-dir ./data \
--device $device \
--result-path ./results \
--dataset-name $dataset
done


