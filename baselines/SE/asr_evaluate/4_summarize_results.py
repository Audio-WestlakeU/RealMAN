import os
import re

# Get cer results for each scene
def extract_lines(source_path, destination_path, keyword):
    with open(source_path, 'r') as f:
        lines = f.readlines()[:50]
    
    filtered_lines = [re.sub(' +', '\t', line) for line in lines if keyword in line]
    
    with open(destination_path, 'a') as f:
        f.writelines(f'{os.path.basename(source_path)}\n{filtered_lines[0]}')
    with open(summary_path, 'a') as f:
        f.write(f'{os.path.basename(source_path)}\n{filtered_lines[0]}')


for mode in ['static', 'moving']:
    scenes_path = f'/asr_results/wenetspeech_asr_model/dataset_results/{mode}'
    scenes = [name for name in os.listdir(scenes_path) if os.path.isdir(os.path.join(scenes_path, name))]
    scenes.sort()
    print(scenes)
    print(len(scenes))

    summary_path = f'{scenes_path}/summary.txt'
    with open(summary_path, 'w') as f:
            f.write(f'\n')

    for scene in scenes:
        scene_path = f'{scenes_path}/{scene}'
        with open(summary_path, 'a') as f:
            f.write(f'# {scene}\n')

        output_path = f'{scene_path}/results_{scene}'
        if os.path.exists(output_path):
            os.remove(output_path)

        for root, dirs, files in os.walk(scene_path):
            files.sort() 
            for index, file in enumerate(files):
                if file.endswith('.txt'):
                    extract_lines(os.path.join(root, file), output_path, 'Sum/Avg') 
        
        with open(summary_path, 'a') as f:
            f.write(f'\n# end of {scene}\n\n')



# write summary to markdown
import pandas as pd
import re

for mode in ['moving','static']:
    txt_path = f'/asr_results/wenetspeech_asr_model/dataset_results/{mode}/summary.txt'
    md_path = txt_path.replace('.txt', '.md')

    with open(txt_path, 'r') as file:
        lines = file.readlines()

    with open(md_path, 'w') as f:
        f.write(f'# {mode.capitalize()} Results\n\n')
        f.write('ASR Model was Tained on WenetSpeech Dataset: https://arxiv.org/abs/2110.03370\n\n')

    # Initializes the list to store data
    data = {'dataset': [], 'Snt': [], 'Wrd': [], 'Corr': [], 'Sub': [], 'Del': [], 'Ins': [], 'Err': [], 'S.Err': []}
    summary_data = {'dataset': [], 'Snt': [], 'Wrd': [], 'Corr': [], 'Sub': [], 'Del': [], 'Ins': [], 'Err': [], 'S.Err': []}

    # Extract the required data and store it in a dictionary
    current_dataset = ''
    current_scene = ''
    for line in lines:
        if line.startswith('#'):
            if 'end' in line:
                df = pd.DataFrame(data)
                markdown_content = df.to_markdown(index=False, floatfmt='.1f')

                with open(md_path, 'a') as f:
                    f.write(markdown_content)
            else:
                current_scene = line.split('# ')[1].strip()
                with open(md_path, 'a') as f:
                    f.write('\n\n## {}\n'.format(current_scene))
                # Reinitializes the list
                data = {'dataset': [], 'Snt': [], 'Wrd': [], 'Corr': [], 'Sub': [], 'Del': [], 'Ins': [], 'Err': [], 'S.Err': []}
        elif line.startswith('transcript') or line.startswith('clean'):
            ss = line.strip().split('_')[-1]
            current_dataset = line.strip().replace(f'_{ss}', '')
            # print(current_dataset)
        else:
            numbers = re.findall(r"\d+\.?\d*", line)  # Extract numbers from the line
            if numbers:
                data['dataset'].append(current_dataset)
                data['Snt'].append(int(numbers[0]))
                data['Wrd'].append(int(numbers[1]))
                data['Corr'].append(float(numbers[2]))
                data['Sub'].append(float(numbers[3]))
                data['Del'].append(float(numbers[4]))
                data['Ins'].append(float(numbers[5]))
                data['Err'].append(float(numbers[6]))
                data['S.Err'].append(float(numbers[7]))
                # Calculate summary data
                if current_dataset not in summary_data['dataset']:
                    summary_data['dataset'].append(current_dataset)
                    summary_data['Snt'].append(int(numbers[0]))
                    summary_data['Wrd'].append(int(numbers[1]))
                    summary_data['Corr'].append(float(numbers[2])*int(numbers[1]))
                    summary_data['Sub'].append(float(numbers[3])*int(numbers[1]))
                    summary_data['Del'].append(float(numbers[4])*int(numbers[1]))
                    summary_data['Ins'].append(float(numbers[5])*int(numbers[1]))
                    summary_data['Err'].append(float(numbers[6])*int(numbers[1]))
                    summary_data['S.Err'].append(float(numbers[7])*int(numbers[0]))
                else:
                    summary_data['Snt'][summary_data['dataset'].index(current_dataset)] += int(numbers[0])
                    summary_data['Wrd'][summary_data['dataset'].index(current_dataset)] += int(numbers[1])
                    summary_data['Corr'][summary_data['dataset'].index(current_dataset)] += float(numbers[2])*int(numbers[1])
                    summary_data['Sub'][summary_data['dataset'].index(current_dataset)] += float(numbers[3])*int(numbers[1])
                    summary_data['Del'][summary_data['dataset'].index(current_dataset)] += float(numbers[4])*int(numbers[1])
                    summary_data['Ins'][summary_data['dataset'].index(current_dataset)] += float(numbers[5])*int(numbers[1])
                    summary_data['Err'][summary_data['dataset'].index(current_dataset)] += float(numbers[6])*int(numbers[1])
                    summary_data['S.Err'][summary_data['dataset'].index(current_dataset)] += float(numbers[7])*int(numbers[0])
                    

    for i in range(len(summary_data['dataset'])):
        summary_data['Corr'][i] /= summary_data['Wrd'][i]
        summary_data['Sub'][i] /= summary_data['Wrd'][i]
        summary_data['Del'][i] /= summary_data['Wrd'][i]
        summary_data['Ins'][i] /= summary_data['Wrd'][i]
        summary_data['Err'][i] /= summary_data['Wrd'][i]
        summary_data['S.Err'][i] /= summary_data['Snt'][i]

    df = pd.DataFrame(summary_data)
    markdown_content = df.to_markdown(index=False, floatfmt='.1f')

    with open(md_path, 'a') as f:
        f.write('\n\n## Summary\n\n')
        f.write(markdown_content)
                

    print(f'Done! {md_path}')