import os
import shutil

from scipy.fft import dst


def copydir_inc_suffix(src_dir: str, dst_dir: str) -> str:
    """
    复制文件夹到目标路径，如果目标已存在则添加递增数字后缀

    Args:
        src_dir: 源文件夹路径
        dest_dir: 目标文件夹路径（如果存在则添加后缀）


    Returns:
        最终复制的目标路径
    """

    counter: int = 0
    new_dst_dir: str = dst_dir
    # 如果目标存在，尝试添加 _1, _2, ... 直到找到不存在的目录
    while os.path.exists(new_dst_dir):
        counter += 1
        new_dst_dir = dst_dir + f"_{counter}"

    # 使用 shutil.copytree 复制（确保目标不存在）
    shutil.copytree(src_dir, new_dst_dir)
    return new_dst_dir


# 创建一个文件夹，检查本地是否有，如果有在文件夹名最后递增追加数字
def makedirs_inc_suffix(base_dir: str, *, use_existing: bool = False) -> str:
    """
    创建新目录，如果目录存在则添加递增数字后缀

    参数:
        base_path: 要创建的基本目录路径
        use_existing: 如果为True，当目录存在时直接返回现有目录路径
                     如果为False，则添加数字后缀创建新目录

    返回:
        创建的目录路径或现有目录路径
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    elif not use_existing:
        i = 1
        while os.path.exists(base_dir + f"_{i}"):
            i += 1
        base_dir = base_dir + f"_{i}"
        os.makedirs(base_dir)
    return base_dir
