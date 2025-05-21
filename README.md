# 数据集

## 数据处理pipeline

cd data_process

### Step1:Lang_sam
得到图片中人物的分割

入口 ./lang_sam.py

启动示例：

python lang_sam.py --dataset_root_path xxx --gpu_list 0,1,2,3,4,5,6,7

注意：dataset_root_path的xxx路径下必须有名为img的文件夹，里面放置需要处理的图片（可以有多级目录）

其余参数解析：

--total_segments 将整个数据集切分为几片

--which_segment 当前处理第几片数据集

### Step2:Detect_skeleton

入口 ./skeleton_detect.py

启动示例：

python skeleton_detect.py --dataset_root_path xxx --gpu_list 0,1,2,3,4,5,6,7

注意：dataset_root_path的xxx路径下必须有名为img的文件夹，里面放置需要处理的图片（可以有多级目录）

其余参数解析：

--total_segments 将整个数据集切分为几片

--which_segment 当前处理第几片数据集

### Step3:Detect_skeleton

入口 ./skeleton_detect.py

启动示例：

python skeleton_detect.py --dataset_root_path xxx --gpu_list 0,1,2,3,4,5,6,7

注意：dataset_root_path的xxx路径下必须有名为img的文件夹，里面放置需要处理的图片（可以有多级目录）

其余参数解析：

--total_segments 将整个数据集切分为几片

--which_segment 当前处理第几片数据集

### Step3:face_parser(真人)

入口 ./face_parse_real.py

启动示例：

python face_parse_real.py --dataset_root_path xxx 

注意：dataset_root_path的xxx路径下必须有名为img的文件夹，里面放置需要处理的图片（可以有多级目录）

其余参数解析：

--total_segments 将整个数据集切分为几片

--which_segment 当前处理第几片数据集

### Step4:sapiens_seg

入口 ./sapiens/seg/seg.sh

需要修改其中的INPUT、OUTPUT、VALID_GPU_IDS

启动示例：

cd sapiens/seg && bash seg.sh

### Step4:sapiens_skeleton

入口 ./sapiens/pose/keypoints308.sh

需要修改其中的INPUT、OUTPUT、VALID_GPU_IDS

启动示例：

cd sapiens/pose && bash keypoints308.sh

## 训练数据
- vcg_122w [不含图片], 467725
  - https://pan.baidu.com/s/1tuy5hYuYh-k3FDtgp4Vrmw?pwd=yhew 提取码: yhew 
- vcg_images_80w [不含图片], 123529
  - https://pan.baidu.com/s/12wY4t3EnihINTtIhMCSsWg?pwd=ujb9 提取码: ujb9 
- vcg_reelshort_0_200 [不含图片], 129446
  - https://pan.baidu.com/s/1NtSTz9LENzCrFWn1Ce9f-A?pwd=jqsr 提取码: jqsr 
- vcg_chat_history_0_300_two_people  [不含图片], 117416
  - https://pan.baidu.com/s/1_KgaiTc0G9WFVPk93hIpIw?pwd=kcgr 提取码: kcgr 
- flux_img_78_all [含图片], 95798
  - https://pan.baidu.com/s/1zP5ZIrjzd76yWmPgCxgbLg?pwd=gmse 提取码: gmse 
- flux_anime_1character [含图片], 61847
  - https://pan.baidu.com/s/1Y-jA0PhwEpUeNXDeMXn7_A?pwd=nani 提取码: nani 
- flux_anime_2characters [含图片], 41357
  - https://pan.baidu.com/s/1yM2KkJTQuFODNJN3QGxiIg?pwd=c253 提取码: c253 
- anime_pictures_46w [不含图片], 423535
  - https://pan.baidu.com/s/1U61GOFYUtZemqJz7_9WEaQ?pwd=wa9x 提取码: wa9x 
- danbooru_anime [不含图片], 82947
  - https://pan.baidu.com/s/1U61GOFYUtZemqJz7_9WEaQ?pwd=wa9x 提取码: wa9x 
- tuchong

注意：
其中，vcg_122w, vcg_images_80w, vcg_reelshort_0_200, vcg_chat_history_0_300_two_people, anime_pictures_46w, danbooru_anime，如果需要图片，请发送邮件到wxktongji@163.com，说明身份/用途

另外，数据集解压之后，请按照以下结构去除多余目录，以下文件目录格式为准：
```|-- anime_pictures_46w
|   |-- caption
|   |-- img
|   |-- json_w_face_body
|   |-- mask
|   |-- mask_sapiens
|   `-- skeleton_sapiens_308
|-- danbooru_anime
|   |-- caption
|   |-- img
|   |-- json_w_face
|   |-- mask
|   `-- skeleton_sapiens_308
|-- flux_anime_1character
|   |-- caption
|   |-- img
|   |-- json_w_face
|   |-- mask_sapiens
|   |-- skeleton
|   `-- skeleton_sapiens_308
|-- flux_anime_2characters
|   |-- caption
|   |-- img
|   |-- json_w_face
|   |-- mask
|   |-- mask_sapiens
|   `-- skeleton_sapiens_308
|-- flux_img_78_all
|   |-- caption
|   |-- face_embedding
|   |-- img
|   |-- json_final
|   |-- mask
|   |-- mask_sapiens
|   |-- skeleton_sapiens
|   `-- skeleton_sapiens_308
|-- vcg_122w
|   |-- caption
|   |-- face_embedding
|   |-- img
|   |-- mask
|   |-- mask_sapiens
|   |-- skeleton_sapiens
|   `-- json_final
|-- vcg_chat_history_0_300_two_people
|   |-- caption
|   |-- face_embedding
|   |-- img
|   |-- json_final
|   |-- mask
|   |-- mask_sapiens
|   `-- skeleton_sapiens
|-- vcg_images_80W
|   |-- caption
|   |-- face_embedding
|   |-- img
|   |-- json_final
|   |-- mask
|   |-- mask_sapiens
|   |-- origin
|   `-- skeleton_sapiens
`-- vcg_reelshort_0_200
    |-- caption
    |-- face_embedding
    |-- img
    |-- json_final
    |-- mask
    |-- mask_sapiens
    `-- skeleton_sapiens```

## 验证数据 
我们准备了真人/动漫，换脸 or 换装的几种case，数据存放地址：./valid_data
```|-- mask
|   |-- clothes
|   `-- face
|-- origin
|   |-- clothes
|   `-- face
|-- process_img
|   |-- face_bbox
|   |-- img
|   |-- json
|   |-- mask
|   |-- mask_sapiens
|   |-- skeleton
|   `-- skeleton_sapiens_308
|-- process_img_reference
|   |-- img
|   |-- json
|   |-- mask
|   |-- mask_sapiens
|   |-- skeleton
|   `-- skeleton_sapiens_308
`-- reference
    |-- clothes
    `-- face```

# 模型

换脸模型：FLUX-Inpainting-IPA-face： https://pan.baidu.com/s/1N5gY0GfjXroMyv6DwYw7bw?pwd=p23s 提取码: p23s 

换装模型：FLUX-Inpainting-IPA-cloth： https://pan.baidu.com/s/1BKQHrN1Irocs5iHSm43yfA?pwd=8bst 提取码: 8bst 

antelopev2：https://pan.baidu.com/s/1rDxuSv9FSYiB-sQJm4tmlA?pwd=8p9x 提取码: 8p9x 

CurricularFace：https://pan.baidu.com/s/15AIykSXedKYe9qOqdAHcMQ?pwd=nawi 提取码: nawi 

dinov2_vitg14：https://pan.baidu.com/s/1cl1YPyvUFh24lL7P-ESZfg?pwd=wdrm 提取码: wdrm 

模型目录结构：
```|-- CurricularFace
|   `-- CurricularFace_Backbone.pth
|-- antelopev2
|   `-- glintr100.onnx
|-- checkpoint-cloth
|   |-- ip_adaptor.safetensors
|   |-- ip_adaptor_controlnet.safetensors
|   `-- ip_adaptor_project.safetensors
`-- checkpoint-face
    |-- ip_adaptor.safetensors
    |-- ip_adaptor_controlnet.safetensors
    `-- ip_adaptor_project.safetensors
```
# 环境
conda env create -f environment.yml

# 代码说明
## 训练

```python3 train_flux_deepspeed_inpainting_ipa.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --config_file accelerate_deepspeed_bf16.yaml \
    --main_process_port 30090 \
    --num_processes 8 \
    train_flux_deepspeed_inpainting_ipa.py \
    --config "train_configs/inpaint_cloth.yaml"
```

## 评测

```python3 valid_training.py```

# 效果
