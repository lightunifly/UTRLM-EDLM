#!/bin/bash

#cd ${path}
ls

#export HYDRA_FULL_ERROR=1

#parameterization=arebm
#exp_name=${exp_names[$parameterization]}
#backbone=${backbones[$parameterization]}

#is_start=1.0

#output_file=${parameterization}_${eval_model_name}_${num_step}_${is_size}_${is_start}_${is_end}
#export PYTHONPATH="/data/home/scxj534/run/wu/quantum/Energy-Diffusion-LLM-main/:$PYTHONPATH"
#export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

#python -u -m main \
#python main.py \
#    mode=sample_eval \
#    hydra.run.dir=outputs/$exp_name \
#    eval.gen_ppl_eval_model_name_or_path=$eval_model \
#    data=openwebtext-split  \
#    model.length=1024  \
#    sampling.predictor=ddpm_cache  \
#    sampling.steps=128 \
#    sampling.is_size=2 \
#    sampling.is_start=0.6 \
#    sampling.is_end=0.4 \
#    loader.eval_batch_size=1 \
#    sampling.num_sample_batches=2 \
#    ebm_backbone=ar
python try.py
