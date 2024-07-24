import os
import time
import resampy
import soundfile
import json
from typing import List
import multiprocessing as mp
import numpy as np


# asr_process: function, the function to process the enhanced speech using ESPNet toolkit and get the ASR results
def asr_process(files):
    # input sample rate of the ASR model 
    fs = 16000 

    def text_normalizer(text):
        text = text.upper()
        return text.translate(str.maketrans('', '', string.punctuation))

    try:
        from espnet2.bin.asr_inference import Speech2Text
        import string
        
        # using the ASR model trained on the WenetSpeech dataset (https://arxiv.org/abs/2110.03370)
        speech2text = Speech2Text(
            asr_train_config='/espnet/egs2/wenetspeech/asr1/exp_ctc_44_32_60/espnet/pengcheng_guo_wenetspeech_asr_train_asr_raw_zh_char/config.yaml',
            # here you need to specify the path of the ASR model
            asr_model_file='/espnet/egs2/wenetspeech/asr1/exp_ctc_44_32_60/espnet/pengcheng_guo_wenetspeech_asr_train_asr_raw_zh_char/valid.acc.ave_10best.pth',
            device="cuda",
            minlenratio=0.0,
            maxlenratio=0.0,
            ctc_weight=0.6,
            beam_size=10,
            batch_size=0,
            nbest=1
        )
        for index, file in enumerate(files):            
            wav_path = file
            speech, rate = soundfile.read(wav_path)
            ma = np.max(np.abs(speech))
            speech = speech / ma
            speech = resampy.resample(speech, rate, fs, axis=0)
            nbests = speech2text(speech)
            text, *_ = nbests[0]

            path_parts = wav_path.split(os.sep)
            if len(path_parts) > 7:
                path_parts = path_parts[-7:]
            wav_path = os.sep.join(path_parts)

            json_obj = json.dumps([wav_path, text_normalizer(text)], ensure_ascii=False)
            excep=True
            while excep:
                try :
                    file_output = open(json_file, 'a', encoding='utf-8')
                    excep=False
                except:
                    time.sleep(1)
                
            file_output.write(json_obj + ',\n')
            file_output.close()

            print(f"{wav_path}: {text_normalizer(text)}")
            print("*" * 50)
    except Exception as e:
        print(f"{e}")


if __name__ == '__main__':
    # test_recs: list, directories of the test results enhanced by the SE model
    test_recs = ['/logs/SpatialNet/final/version_7/epoch47_test_set/version_2/examples', '/logs/SpatialNet/final/version_7/epoch47_test_set/version_3/examples']
    # json_file: str, the path of the output json file
    json_file = '/asr_results/wenetspeech_asr_model/dataset_results/transcript_highSNR_SpatialNet_real_real.json'

    all_files = []
    for test_rec in test_recs:
        for root, dirs, files in os.walk(test_rec):
            for index, file in enumerate(files):
                # channel 0 is the reference channel
                if file.endswith('_CH0.flac'):
                    all_files.append(os.path.join(root, file))

    files = all_files
    files.sort()

    asr_process(files)
