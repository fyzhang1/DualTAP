import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class PrivacyProtectionDataset(Dataset):
    """
    隐私保护数据集
    适配现有数据格式：
    - 图像文件：.png 格式
    - privacy_qa.json：answer 是字典格式，包含多种隐私类型
    - normal_qa.json：answers 是列表格式
    """
    
    def __init__(self, data_root, image_size=224, transform=None, app_filter=None):
        """
        Args:
            data_root: 数据根目录
            image_size: 图像尺寸
            transform: 图像变换
            app_filter: 只加载指定的app，例如 "amazon"
        """
        self.data_root = data_root
        self.image_size = image_size
        self.app_filter = app_filter
        
        # 默认的图像变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
        
        # 加载所有数据
        self.data_items = []
        self._load_data()
    
    def _parse_privacy_answer(self, answer_data):
        """
        解析隐私答案，支持字符串和字典两种格式
        
        Args:
            answer_data: 可以是字符串 "emailAddress: xxx" 
                        或字典 {"name": ["John"], "emailAddress": ["john@example.com"]}
        
        Returns:
            str: "name: John, emailAddress: john@example.com"
        """
        # 如果已经是字符串，直接返回
        if isinstance(answer_data, str):
            return answer_data
        
        # 如果是字典，解析为字符串
        if isinstance(answer_data, dict):
            parts = []
            for key, values in answer_data.items():
                if isinstance(values, list) and len(values) > 0:
                    # 将值列表合并
                    value_str = ", ".join(str(v) for v in values)
                    parts.append(f"{key}: {value_str}")
            return "; ".join(parts) if parts else "No privacy information"
        
        # 其他情况返回默认值
        return "No privacy information"
    
    def _load_data(self):
        """加载所有 app 的数据"""
        if not os.path.exists(self.data_root):
            raise ValueError(f"数据根目录不存在: {self.data_root}")
        
        # 遍历所有 app 目录（排序，保证确定性）
        for app_name in sorted(os.listdir(self.data_root)):
            # 如果指定了app_filter，只加载该app
            if self.app_filter and app_name != self.app_filter:
                continue
            
            app_path = os.path.join(self.data_root, app_name)
            if not os.path.isdir(app_path):
                continue
            
            image_dir = os.path.join(app_path, "images")
            privacy_qa_path = os.path.join(app_path, "privacy_qa.json")
            normal_qa_path = os.path.join(app_path, "normal_qa.json")
            
            # 检查必要文件是否存在
            if not os.path.exists(image_dir):
                print(f"警告: {app_name} 缺少 images 目录，跳过")
                continue
            if not os.path.exists(privacy_qa_path):
                print(f"警告: {app_name} 缺少 privacy_qa.json，跳过")
                continue
            if not os.path.exists(normal_qa_path):
                print(f"警告: {app_name} 缺少 normal_qa.json，跳过")
                continue
            
            # 加载 QA 数据
            with open(privacy_qa_path, 'r', encoding='utf-8') as f:
                privacy_qa = json.load(f)
            with open(normal_qa_path, 'r', encoding='utf-8') as f:
                normal_qa = json.load(f)
            
            # 构建数据项（按图像键排序，保证确定性）
            # 注意：JSON中的key是.jpg，但实际文件是.png
            for img_key in sorted(privacy_qa.keys()):
                # 尝试不同的文件扩展名
                img_name_base = os.path.splitext(img_key)[0]
                
                # 尝试找到实际的图像文件
                img_path = None
                for ext in ['.png', '.jpg', '.jpeg']:
                    test_path = os.path.join(image_dir, img_name_base + ext)
                    if os.path.exists(test_path):
                        img_path = test_path
                        break
                
                if img_path is None:
                    print(f"警告: 图像不存在 {img_name_base}.*，跳过")
                    continue
                
                # 获取该图像的隐私和正常任务 QA 对
                privacy_qa_list = privacy_qa.get(img_key, [])
                normal_qa_list = normal_qa.get(img_key, [])
                
                # 转换格式
                # 隐私QA: 将answer转换为统一的文本格式（支持字符串和字典）
                converted_privacy_qa = []
                for qa in privacy_qa_list:
                    question = qa.get('question', '')
                    # 兼容 'answer' 和 'answers' 两种字段名
                    answer_data = qa.get('answers', qa.get('answer', ''))
                    answer_text = self._parse_privacy_answer(answer_data)
                    converted_privacy_qa.append({
                        'question': question,
                        'answer': answer_text
                    })
                
                # 正常QA: 兼容 answers(list) 与 answer(str) 两种格式，统一为单个字符串
                converted_normal_qa = []
                for qa in normal_qa_list:
                    question = qa.get('question', '')
                    answers_list = []
                    if isinstance(qa.get('answers', None), list):
                        answers_list = qa.get('answers', [])
                    else:
                        single = qa.get('answer', '')
                        if isinstance(single, list):
                            answers_list = single
                        elif isinstance(single, str) and len(single) > 0:
                            answers_list = [single]
                        else:
                            answers_list = []
                    # 取第一个答案，或者为空字符串
                    answer_text = answers_list[0] if len(answers_list) > 0 else ""
                    converted_normal_qa.append({
                        'question': question,
                        'answer': answer_text
                    })
                
                if len(converted_privacy_qa) == 0 or len(converted_normal_qa) == 0:
                    print(f"警告: {img_key} 缺少 QA 对，跳过")
                    continue
                
                self.data_items.append({
                    'app_name': app_name,
                    'image_path': img_path,
                    'privacy_qa': converted_privacy_qa,
                    'normal_qa': converted_normal_qa
                })
        
        print(f"成功加载 {len(self.data_items)} 个数据项")
        if self.app_filter:
            print(f"仅加载应用: {self.app_filter}")
    
    def __len__(self):
        return len(self.data_items)
    
    def __getitem__(self, idx):
        """
        返回一个数据项
        Returns:
            dict: {
                'image': Tensor (C, H, W),
                'privacy_qa': List[Dict],
                'normal_qa': List[Dict],
                'app_name': str,
                'image_path': str
            }
        """
        item = self.data_items[idx]
        
        # 加载图像
        image = Image.open(item['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'privacy_qa': item['privacy_qa'],
            'normal_qa': item['normal_qa'],
            'app_name': item['app_name'],
            'image_path': item['image_path']
        }


def collate_fn(batch):
    """
    自定义 collate 函数，用于 DataLoader
    由于每个样本的 QA 对数量不同，需要特殊处理
    """
    images = torch.stack([item['image'] for item in batch])
    privacy_qa_list = [item['privacy_qa'] for item in batch]
    normal_qa_list = [item['normal_qa'] for item in batch]
    app_names = [item['app_name'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    
    return {
        'images': images,
        'privacy_qa_list': privacy_qa_list,
        'normal_qa_list': normal_qa_list,
        'app_names': app_names,
        'image_paths': image_paths
    }
