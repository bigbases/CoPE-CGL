seed=1024
repeat=1
epoch=500
device=cuda:1
run=1

# CoraFull
dataset=corafull
budget=100

for method in sage_cog
do
python train_tpp.py \
--seed $seed \
--repeat $repeat \
--cls-epoch $epoch \
--cgl-method $method \
--data-dir ./data \
--device $device \
--result-path ./results \
--dataset-name $dataset > ./logs_sage_new/${dataset}_${method}_${run}_new.log
done
