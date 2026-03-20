#!/bin/bash
export PF_SITE=local
DATA_DIR=$(python3 scripts/get_param.py particleflow_spec.yaml productions.cms_run3.workspace_dir)/tfds/

#./scripts/local/wrapper.sh python mlpf/pipeline.py --spec-file particleflow_spec.yaml --model-name pyg-cms-v1 --production cms_run3 --data-dir $DATA_DIR train --gpu_batch_multiplier 4
#./scripts/local/wrapper.sh python mlpf/pipeline.py --spec-file particleflow_spec.yaml --model-name pyg-cms-v1 --production cms_run3 --data-dir $DATA_DIR --prefix asdf ray-train --gpu_batch_multiplier 4 --ray-local --ray-cpus 4 --ray-gpus 1

./scripts/local/wrapper.sh python3 mlpf/standalone/eval.py --data-dir $DATA_DIR --dsl "i(55,128,256,default)|s(16,128,512)*4|o(8,256,default)" > log_standard.txt
mv plots/attention_matrix.png plots/attention_standard.png
./scripts/local/wrapper.sh python3 mlpf/standalone/eval.py --data-dir $DATA_DIR --dsl "i(55,128,256,default)|h(16,128,512)*4|o(8,256,default)" > log_hept.txt
mv plots/attention_matrix.png plots/attention_hept.png
./scripts/local/wrapper.sh python3 mlpf/standalone/eval.py --data-dir $DATA_DIR --dsl "i(55,128,256,default)|f(16,128,512)*4|o(8,256,default)" > log_fast.txt
mv plots/attention_matrix.png plots/attention_fast.png
./scripts/local/wrapper.sh python3 mlpf/standalone/eval.py --data-dir $DATA_DIR --dsl "i(55,128,256,default)|g(16,128,512)*4|o(8,256,default)" > log_global.txt
mv plots/attention_matrix.png plots/attention_global.png

#./scripts/local/wrapper.sh python3 mlpf/standalone/run_evolution.py --pop-size 2 --generations 1 --data-dir $DATA_DIR
