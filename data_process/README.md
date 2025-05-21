## Step1:Lang_sam
得到图片中人物的分割

入口 ./lang_sam.py

启动示例：

python lang_sam.py --dataset_root_path xxx --gpu_list 0,1,2,3,4,5,6,7

注意：dataset_root_path的xxx路径下必须有名为img的文件夹，里面放置需要处理的图片（可以有多级目录）

其余参数解析：

--total_segments 将整个数据集切分为几片

--which_segment 当前处理第几片数据集

## Step2:Detect_skeleton

入口 ./skeleton_detect.py

启动示例：

python skeleton_detect.py --dataset_root_path xxx --gpu_list 0,1,2,3,4,5,6,7

注意：dataset_root_path的xxx路径下必须有名为img的文件夹，里面放置需要处理的图片（可以有多级目录）

其余参数解析：

--total_segments 将整个数据集切分为几片

--which_segment 当前处理第几片数据集

## Step3:Detect_skeleton

入口 ./skeleton_detect.py

启动示例：

python skeleton_detect.py --dataset_root_path xxx --gpu_list 0,1,2,3,4,5,6,7

注意：dataset_root_path的xxx路径下必须有名为img的文件夹，里面放置需要处理的图片（可以有多级目录）

其余参数解析：

--total_segments 将整个数据集切分为几片

--which_segment 当前处理第几片数据集

## Step3:face_parser(真人)

入口 ./face_parse_real.py

启动示例：

python face_parse_real.py --dataset_root_path xxx 

注意：dataset_root_path的xxx路径下必须有名为img的文件夹，里面放置需要处理的图片（可以有多级目录）

其余参数解析：

--total_segments 将整个数据集切分为几片

--which_segment 当前处理第几片数据集

## Step4:sapiens_seg

入口 ./sapiens/seg/seg.sh

需要修改其中的INPUT、OUTPUT、VALID_GPU_IDS

启动示例：

cd sapiens/seg && bash seg.sh

## Step4:sapiens_skeleton

入口 ./sapiens/pose/keypoints308.sh

需要修改其中的INPUT、OUTPUT、VALID_GPU_IDS

启动示例：

cd sapiens/pose && bash keypoints308.sh
