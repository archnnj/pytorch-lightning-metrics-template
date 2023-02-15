#!/bin/bash

GPU=7
CPU=104-111
LOG_FOLDER=~/storage/cl_road_pavement/outputs

trap 'for pid in $pids; do kill $pid; done; exit' INT
run () {
	i=$2
	c_name=cudrano_pytorch-lightning-template_arg${i}
	run-docker --container_name ${c_name} GPU CPU python train.py &
	pids[${i}]=$!
	sleep 5
	docker logs -f ${c_name} > ${LOG_FOLDER}/run_${c_name}.log &
	echo "Started exp perm $i"
}

for i in {1..5}; do
	run $i
done

for pid in $pids; do
	wait $pid
done

