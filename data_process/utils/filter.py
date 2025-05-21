import numpy as np


def is_contained(bbox1, bbox2):
    """
    判断bbox1是否包含bbox2
    :param bbox1: (x1, y1, x2, y2)
    :param bbox2: (x1, y1, x2, y2)
    :return: True 如果bbox1包含bbox2，否则False
    """
    return bbox1[0] <= bbox2[0] and bbox1[1] <= bbox2[1] and bbox1[2] >= bbox2[2] and bbox1[3] >= bbox2[3]

def is_contained_with_threshold(A, B, threshold=0.8):
    """
    判断边界框A是否被边界框B包含，基于面积比例的阈值。
    
    参数:
    A: 子框,边界框A的坐标，格式为(x_min, y_min, x_max, y_max)。
    B: 父框,边界框B的坐标，格式为(x_min, y_min, x_max, y_max)。
    threshold: 包含的阈值，表示A在B内部的面积比例。
    
    返回:
    如果A在B内部的面积比例超过阈值，则返回True；否则返回False。
    """
    # 解包边界框A和B的坐标
    Ax_min, Ay_min, Ax_max, Ay_max = A
    Bx_min, By_min, Bx_max, By_max = B
    
    # 计算A和B的交集区域的坐标
    Ix_min = max(Ax_min, Bx_min)
    Iy_min = max(Ay_min, By_min)
    Ix_max = min(Ax_max, Bx_max)
    Iy_max = min(Ay_max, By_max)
    
    # 确保交集区域是有效的（即存在交集）
    if Ix_min < Ix_max and Iy_min < Iy_max:
        # 计算A的面积和交集区域的面积
        A_area = (Ax_max - Ax_min) * (Ay_max - Ay_min)
        I_area = (Ix_max - Ix_min) * (Iy_max - Iy_min)
        
        # 计算交集区域面积占A面积的比例
        overlap_ratio = I_area / A_area
        
        # 判断是否满足阈值条件
        return overlap_ratio >= threshold
    else:
        # 如果没有交集，显然不满足包含条件
        return False

def get_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def find_sub_bboxes(bboxes):
    """
    查找每个bbox及其包含的子bbox
    :param bboxes: bbox列表，每个bbox表示为(x1, y1, x2, y2)
    :return: 包含子bbox的字典
    """
    sub_bboxes = {i: [] for i in range(len(bboxes))}

    for i in range(0, len(bboxes)):
        for j in range(i+1, len(bboxes)):
            area_bbox1 = get_area(bboxes[i])
            area_bbox2 = get_area(bboxes[j])
            if area_bbox1 >= area_bbox2:
                if is_contained_with_threshold(bboxes[j], bboxes[i], threshold=0.8):
                    sub_bboxes[i].append(j)
            else:
                if is_contained_with_threshold(bboxes[i], bboxes[j], threshold=0.8):
                    sub_bboxes[j].append(i)
    return sub_bboxes

def mask_area_filter(masks):
    index = []
    if len(masks):
        height, width = masks[0].shape
        total_area = height * width
        for i, mask in enumerate(masks):
            mask_area = mask.sum()
            percent = mask_area / total_area
            if percent > 0.01:
                index.append(1)
            else:
                index.append(0)
    return index

def subtract_masks(parent_mask, child_mask):
    """
    从父掩模中减去子掩模，返回区分后的父掩模。
    """
    return np.where(child_mask == 1, 0, parent_mask)
    