"""在数据集的目录下创建pytorch文件夹，把数据集转换为ImageFolder
每个文件夹名是人id代码
query -> query
gt_bbox -> mulit-query
bounding_box_test -> gallery
bounding_box_train -> train_all
bounding_box_train -> train + val

每个身份的第一张会放到val中

数据集介绍看readme

实际转换后train有751个身份,query有750个身份,gallery有752个身份（多一个0和一个-1）
"""

import os
from shutil import copyfile


dataset_path = "./datasets/Market-1501"  # Please not change.
# download_path2 = "./data/Market"  # You only need to change this line to your dataset download path


# if not os.path.isdir(download_path):
#     if os.path.isdir(download_path2):
#         os.system("mv %s %s" % (download_path2, download_path))  # rename
#     else:
#         print("please change the download_path")

save_path = dataset_path + "/pytorch"
if not os.path.isdir(save_path):
    os.mkdir(save_path)
# -----------------------------------------
# query
query_path = dataset_path + "/query"
query_save_path = dataset_path + "/pytorch/query"
if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)

for root, dirs, files in os.walk(query_path, topdown=True):
    for name in files:
        if not name[-3:] == "jpg":
            continue
        ID = name.split("_")
        src_path = query_path + "/" + name
        dst_path = query_save_path + "/" + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + "/" + name)

# -----------------------------------------
# multi-query
query_path = dataset_path + "/gt_bbox"
# for dukemtmc-reid, we do not need multi-query
if os.path.isdir(query_path):
    query_save_path = dataset_path + "/pytorch/multi-query"
    if not os.path.isdir(query_save_path):
        os.mkdir(query_save_path)

    for root, dirs, files in os.walk(query_path, topdown=True):
        for name in files:
            if not name[-3:] == "jpg":
                continue
            ID = name.split("_")
            src_path = query_path + "/" + name
            dst_path = query_save_path + "/" + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + "/" + name)

# -----------------------------------------
# gallery
gallery_path = dataset_path + "/bounding_box_test"
gallery_save_path = dataset_path + "/pytorch/gallery"
if not os.path.isdir(gallery_save_path):
    os.mkdir(gallery_save_path)

for root, dirs, files in os.walk(gallery_path, topdown=True):
    for name in files:
        if not name[-3:] == "jpg":
            continue
        ID = name.split("_")
        src_path = gallery_path + "/" + name
        dst_path = gallery_save_path + "/" + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + "/" + name)

# ---------------------------------------
# train_all
train_path = dataset_path + "/bounding_box_train"
train_save_path = dataset_path + "/pytorch/train_all"
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:] == "jpg":
            continue
        ID = name.split("_")
        src_path = train_path + "/" + name
        dst_path = train_save_path + "/" + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + "/" + name)


# ---------------------------------------
# train_val
train_path = dataset_path + "/bounding_box_train"
train_save_path = dataset_path + "/pytorch/train"
val_save_path = dataset_path + "/pytorch/val"
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
    os.mkdir(val_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:] == "jpg":
            continue
        ID = name.split("_")
        src_path = train_path + "/" + name
        dst_path = train_save_path + "/" + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
            dst_path = val_save_path + "/" + ID[0]  # first image is used as val image
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + "/" + name)
