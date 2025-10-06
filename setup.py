"""
快速设置脚本
创建必要的目录和数据模板
"""

import os


def setup_project():
    """设置项目结构"""
    
    # 创建必要的目录
    directories = [
        "data",
        "checkpoints",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ 创建目录: {directory}/")
    
    print("\n项目结构设置完成！")
    print("\n下一步：")
    print("1. 运行 'python utils.py' 创建数据目录模板")
    print("2. 将图像文件放置在 data/<app_name>/images/ 目录下")
    print("3. 编辑各应用的 privacy_qa.json 和 normal_qa.json")
    print("4. 运行 'python train.py' 开始训练")


if __name__ == "__main__":
    setup_project()

