seed=1024
repeat=3
epoch=500
device=cuda:0
ratio=0.1
# CoraFull
dataset=corafull

for method in cope_gl tpp taam
do
python train_mixed.py \
--seed $seed \
--repeat $repeat \
--cls-epoch $epoch \
--cgl-method $method \
--data-dir ./data \
--device $device \
--result-path ./results \
--ind_task_ratio $ratio \
--dataset-name $dataset
done

dataset=arxiv

for method in cope_gl tpp taam
do
python train_mixed.py \
--seed $seed \
--repeat $repeat \
--cls-epoch $epoch \
--cgl-method $method \
--data-dir ./data \
--device $device \
--result-path ./results \
--ind_task_ratio $ratio \
--dataset-name $dataset
done


dataset=reddit

for method in cope_gl tpp taam
do
python train_mixed.py \
--seed $seed \
--repeat $repeat \
--cls-epoch $epoch \
--cgl-method $method \
--data-dir ./data \
--device $device \
--result-path ./results \
--ind_task_ratio $ratio \
--dataset-name $dataset
done


dataset=products

for method in cope_gl tpp taam
do
python train_mixed.py \
--seed $seed \
--repeat $repeat \
--cls-epoch $epoch \
--cgl-method $method \
--data-dir ./data \
--device $device \
--result-path ./results \
--ind_task_ratio $ratio \
--dataset-name $dataset
done
