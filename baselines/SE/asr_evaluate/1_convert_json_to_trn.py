import os
import json
import sys
import warnings
import subprocess

# scenes_path: the direcotry of the test scenes
scenes_path = '/RealMAN/test/dp_speech'
scenes = [name for name in os.listdir(scenes_path) if os.path.isdir(os.path.join(scenes_path, name))]
print(scenes)
print(len(scenes))

# json_file: the path of the json file generated by ./0_inference.py
json_file = f'/asr_results/wenetspeech_asr_model/dataset_results/transcript_lowSNR_SpatialNet_real_real.json'

for mode in ['static', 'moving']:
    results_path = f'/asr_results/wenetspeech_asr_model/dataset_results/{mode}'
    for scene in scenes:
        if not os.path.exists(f'{scenes_path}/{scene}/{mode}'):
            continue
        scene_path = os.path.join(results_path, scene)
        if not os.path.exists(scene_path):
            os.makedirs(scene_path)
        
        trn_file = f'{scene_path}/{json_file.split("/")[-1].split(".")[0]}_{scene}.trn'

        # read '*.json' file
        with open(json_file, 'r') as file:
            data = json.load(file)

        # write transcripts to '*.trn' file
        with open(trn_file, 'w') as file:
            for item in data:
                if scene in item[0] and mode in item[0] and 'tar'in item[0]:
                    # The format of each line is: unsigned Chinese text insert space (spk_id-wav_id)
                    s_spaced = " ".join(item[1])
                    file.write(f"{s_spaced}  (S{item[0].split('/')[-2]}-{item[0].split('_')[-1].split('.')[0]}) \n")
                elif scene in item[0] and mode in item[0] and ('rec' in item[0] or 'epoch' in item[0]):
                    s_spaced = " ".join(item[1])
                    file.write(f"{s_spaced}  (S{item[0].split('/')[-2]}-{item[0].split('_')[-2]}) \n")

