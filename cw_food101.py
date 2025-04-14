import os, time
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time, shutil
from tools import get_num_value, write_singe_json, lora_product_image
from diffusers import DiffusionPipeline, StableDiffusionPipeline
import torch

from product_image import run_product_check_point_image

# run dreambooth
def run_cp_food(input_data_path, food_name, output_weight_path, epoch=30000):
    print('=======input_data_path======')
    print(input_data_path)
    print(output_weight_path)
    os.system(
        'accelerate launch self_train_CW_food.py '
        '--pretrained_model_name_or_path="/home/image1325_user/ssd_disk1/your_path/diffsuers_file" '
        '--instance_data_dir={} '
        '--output_dir={} '
        '--instance_prompt="a photo of {}" '
        '--resolution=512 '
        '--train_batch_size=1   '
        '--gradient_accumulation_steps=1   '
        '--checkpointing_steps=5000   '
        '--learning_rate=5e-5   '
        '--lr_scheduler="constant"   '
        '--lr_warmup_steps=0   '
        '--max_train_steps={}   '
        '--seed="0"   '
        '--train_text_encoder'.format(input_data_path, output_weight_path,food_name, epoch)

    )



input_data_path = '/home/image1325_user/ssd_disk1/your_path/Data/Food101'

output_weight_path = '/home/image1325_user/ssd_disk2/your_path/Experiments/output_cp_weight_101_66666'
sample_path = '/home/image1325_user/ssd_disk2/your_path/Experiments/output_cp_sample_101_66666'



label_list = os.listdir(input_data_path)
label_list.sort()
label_list = label_list[0:1]

for label in label_list:

    label = str(label)

    temp_input_data_path = os.path.join(input_data_path, label)
    temp_output_weight_path = os.path.join(output_weight_path, label)

    #
    print('=======product image======')
    print(label)
    print(temp_input_data_path)

    # 
    run_cp_flag = True
    run_cp_flag = False
    if run_cp_flag:
        print('========================')
        print(temp_input_data_path)
        run_cp_food(temp_input_data_path, label, temp_output_weight_path, epoch=25000)

        import time
        print('wait')
        time.sleep(10)
    
    # train
    temp_sample_path = os.path.join(sample_path, label) 
    # product image
    run_product_check_point_image(temp_output_weight_path, temp_input_data_path, label, temp_sample_path, check_point=25000, sample_num=1000)


