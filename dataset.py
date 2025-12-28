import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class PrivacyProtectionDataset(Dataset):
    
    def __init__(self, data_root, image_size=224, transform=None, app_filter=None, split='all', split_ratio=0.8):
        self.data_root = data_root
        self.image_size = image_size
        self.app_filter = app_filter
        self.split = split
        self.split_ratio = float(split_ratio) if split_ratio is not None else 0.8
        
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

        if isinstance(answer_data, str):
            return answer_data
        

        if isinstance(answer_data, dict):
            parts = []
            for key, values in answer_data.items():
                if isinstance(values, list) and len(values) > 0:

                    value_str = ", ".join(str(v) for v in values)
                    parts.append(f"{key}: {value_str}")
            return "; ".join(parts) if parts else "No privacy information"
        

        return "No privacy information"
    
    def _load_data(self):

        if not os.path.exists(self.data_root):
            raise ValueError(f"data file not exit: {self.data_root}")
        

        for app_name in sorted(os.listdir(self.data_root)):

            if self.app_filter and app_name != self.app_filter:
                continue
            
            app_path = os.path.join(self.data_root, app_name)
            if not os.path.isdir(app_path):
                continue
            
            image_dir = os.path.join(app_path, "images")
            privacy_qa_path = os.path.join(app_path, "privacy_qa.json")
            normal_qa_path = os.path.join(app_path, "normal_qa.json")
            
            if not os.path.exists(image_dir):
                print(f"Warning: {app_name} have not images skip")
                continue
            if not os.path.exists(privacy_qa_path):
                print(f"Warning: {app_name} have not privacy_qa skip")
                continue
            if not os.path.exists(normal_qa_path):
                print(f"Warning: {app_name} have not normal_qa skip")
                continue
            
            with open(privacy_qa_path, 'r', encoding='utf-8') as f:
                privacy_qa = json.load(f)
            with open(normal_qa_path, 'r', encoding='utf-8') as f:
                normal_qa = json.load(f)
            

            app_items = []
            for img_key in sorted(privacy_qa.keys()):

                img_name_base = os.path.splitext(img_key)[0]

                img_path = None
                for ext in ['.png', '.jpg', '.jpeg']:
                    test_path = os.path.join(image_dir, img_name_base + ext)
                    if os.path.exists(test_path):
                        img_path = test_path
                        break
                
                if img_path is None:
                    print(f"warning: image no exit {img_name_base}.*,skip")
                    continue
                
                privacy_qa_list = privacy_qa.get(img_key, [])
                normal_qa_list = normal_qa.get(img_key, [])
                

                converted_privacy_qa = []
                for qa in privacy_qa_list:
                    question = qa.get('question', '')
                    answer_data = qa.get('answers', qa.get('answer', ''))
                    answer_text = self._parse_privacy_answer(answer_data)
                    converted_privacy_qa.append({
                        'question': question,
                        'answer': answer_text
                    })
                
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
                    answer_text = answers_list[0] if len(answers_list) > 0 else ""
                    converted_normal_qa.append({
                        'question': question,
                        'answer': answer_text
                    })
                
                if len(converted_privacy_qa) == 0 or len(converted_normal_qa) == 0:
                    print(f"warning: {img_key} have not QA pair, skip")
                    continue
                
                app_items.append({
                    'app_name': app_name,
                    'image_path': img_path,
                    'privacy_qa': converted_privacy_qa,
                    'normal_qa': converted_normal_qa
                })


            if len(app_items) > 0:
                num_train = int(len(app_items) * self.split_ratio)
                if self.split == 'train':
                    self.data_items.extend(app_items[:num_train])
                elif self.split == 'eval':
                    self.data_items.extend(app_items[num_train:])
                else:  # 'all'
                    self.data_items.extend(app_items)
        
        print(f"load {len(self.data_items)} data")
        if self.app_filter:
            print(f"just app: {self.app_filter}")
        if self.split in ('train', 'eval'):
            print(f"data split: split={self.split}, train_ratio={self.split_ratio}")
    
    def __len__(self):
        return len(self.data_items)
    
    def __getitem__(self, idx):
        item = self.data_items[idx]
        

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
