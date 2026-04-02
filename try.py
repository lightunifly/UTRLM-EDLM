import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import transformers
import hydra

@hydra.main(version_base=None, config_path='configs',config_name='config')
def main(config):
    print(config.seed)
    print(config)
#main()

#path = "/data/home/scxj534/.cache/huggingface/hub/checkpoints/mdlm.ckpt"
path = "/data/home/scxj534/.cache/huggingface/hub/models--kuleshov-group--mdlm-owt/snapshots/9e6829bb908d241a074146e4c5c095238bb5e316"
#amod = transformers.AutoModelForMaskedLM.from_pretrained("Synthyra/DPLM2-150M", trust_remote_code=True)
amod = transformers.AutoModelForMaskedLM.from_pretrained(path, trust_remote_code=True)
print(amod)

import diffusion
path = "/data/home/scxj534/.cache/huggingface/hub/checkpoints/mdlm.ckpt"
print('diffusion:')
diffusion.EBM.load_from_checkpoint(path)#,tokenizer=tokenizer,config=config)
