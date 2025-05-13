import os
from image_datasets.utils import get_new_size, c_cropv2, draw_handpose, draw_bodypose, convert_open_to_mmpose, canny_processor, get_skeleton_keypoints_from_facebook308, combine_canny_skeleton
from image_datasets.utils import fill_mask, get_sapiens_sam_path, check_bbox_overlap, calculate_mask_external_rectangle, pad_bbox, canny_processor, c_crop, c_pad, expand_bbox, c_crop_1344x768, get_sapiens_path


plist = [
    # "/mnt/wangxuekuan/code/layout/dataset_split/danbooru_anime.txt",
    # "/mnt/wangxuekuan/code/layout/dataset_split/flux_anime_1character.txt",
    # "/mnt/wangxuekuan/code/layout/dataset_split/flux_anime_2characters.txt",
    "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/anime_pictures_46w.txt",
    "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/train_ip_vcg_80w.txt",
    "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/train_ip_flux_1.txt",
    "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/train_ip_vcg_122w.txt",
    "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/train_ip_vcg_chat_history_0_300_two_people.txt",
    "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/train_ip_vcg_reelshort_0_200.txt",
]

# plist += [
#     "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/tuchong.txt",
# ]

for path in plist:
    data = open(path, "r").readlines()
    print(len(data))
    for item in data:
        item = item.split(", ")
        json_path = item[1].strip()
        image_path = item[2].strip()
        # caption_path = item[3]
        ftype = image_path.split(".")[-1]
        caption_path = "/mnt2/wangxuekuan/data/ip_adapter_gpt4o__desc/" + image_path[:-len(ftype)] + "txt"

        mask_sapiens_path, skeleton_sapiens_path, mask_sam_path = get_sapiens_sam_path(json_path)

        # print(mask_sapiens_path, skeleton_sapiens_path, mask_sam_path)
        # if os.path.exists(json_path) and os.path.exists(image_path) and os.path.exists(caption_path):
        #     print(json_path)
        #     print(image_path)
        #     print(caption_path)
        # if mask_sapiens_path and skeleton_sapiens_path and mask_sam_path:
        #     if os.path.exists(mask_sapiens_path) and os.path.exists(skeleton_sapiens_path) and os.path.exists(mask_sam_path):
        #         print(json_path)
        #         print(image_path)
        #         print(caption_path)

