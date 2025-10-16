import os
import math
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import CLIPModel, CLIPProcessor


# 与 InternVL 输入构建保持一致的特殊标记
IMG_START_TOKEN  = "<img>"
IMG_END_TOKEN    = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_internvl_inputs(question, answer, tokenizer, model, max_len=1024, num_patches=1):
    img_ctx_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    if not hasattr(model, "img_context_token_id") or model.img_context_token_id is None:
        model.img_context_token_id = img_ctx_id

    num_img_tokens = int(getattr(model, "num_image_token", 256)) * int(num_patches)

    user_text = f"<image>\n{question}".strip()
    assistant_text = f" {answer}"

    image_block = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * num_img_tokens) + IMG_END_TOKEN
    user_with_visual = user_text.replace("<image>", image_block, 1)

    full_text = user_with_visual + assistant_text

    q_enc = tokenizer(
        user_with_visual, return_tensors='pt', padding=False, truncation=True, max_length=max_len
    )
    q_len = q_enc["input_ids"].shape[1]

    enc = tokenizer(
        full_text, return_tensors='pt', padding=False, truncation=True, max_length=max_len
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    labels = input_ids.clone()
    labels[:, :q_len] = -100

    img_start_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
    img_end_id   = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
    special_mask = (input_ids == img_ctx_id) | (input_ids == img_start_id) | (input_ids == img_end_id)
    labels[special_mask] = -100

    image_flags = torch.ones(1, num_img_tokens, dtype=torch.long)

    return input_ids, attention_mask, labels, image_flags


class SaliencyAttention:
    """
    基于梯度的显著图提取器：针对特定隐私 QA，计算 surrogate MLLM 的损失
    对输入像素的梯度范数作为注意力图，归一化到 [0,1]。
    """

    def __init__(self, model, tokenizer, device, save_dir=None, method: str = "pixel_grad"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.save_dir = save_dir
        # method: 'pixel_grad' | 'xattn_grad'
        self.method = method
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        self.resize = T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC)

        # Lazy CLIP members (for non-gradient attention)
        self._clip_model = None
        self._clip_processor = None
        self._clip_model_name = getattr(self, "clip_model_name", "openai/clip-vit-base-patch32")
        # Textness weighting strength for CLIP attention fusion
        self._textness_gamma = 1.2

    def _lazy_load_clip(self):
        if self._clip_model is None or self._clip_processor is None:
            self._clip_model = CLIPModel.from_pretrained(self._clip_model_name).to(self.device)
            self._clip_model.eval()
            self._clip_processor = CLIPProcessor.from_pretrained(self._clip_model_name)

    def _compute_single_map(self, image_bchw, qa_list):
        """
        image_bchw: Tensor [1,3,H,W], 值域 [0,1]
        qa_list: List[{question, answer}]
        return: Tensor [1,1,H,W] 于原图尺寸
        """
        if not qa_list:
            h, w = image_bchw.shape[-2:]
            return torch.zeros(1, 1, h, w, device=self.device, dtype=torch.float32)

        # 预处理到模型分辨率
        image_resized = self.resize(image_bchw)
        pixel_values = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(image_resized)
        pixel_values = pixel_values.to(device=self.device, dtype=next(self.model.parameters()).dtype)
        pixel_values.requires_grad_(True)

        total_loss = None
        for qa in qa_list:
            question, answer = qa['question'], qa['answer']
            input_ids, attention_mask, labels, image_flags = _build_internvl_inputs(
                question, answer, self.tokenizer, self.model, max_len=1024, num_patches=1
            )
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)
            image_flags = image_flags.to(self.device)

            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                image_flags=image_flags,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True
            )
            loss = outputs.loss
            total_loss = loss if total_loss is None else (total_loss + loss)

        # d loss / d pixel_values
        grads = torch.autograd.grad(total_loss, pixel_values, retain_graph=False, create_graph=False)[0]
        grads = grads.float()  # [1,3,448,448]
        # 通道范数 -> [1,1,448,448]
        saliency = grads.abs().mean(dim=1, keepdim=True)
        # 归一化
        saliency = saliency - saliency.amin(dim=(-2, -1), keepdim=True)
        denom = saliency.amax(dim=(-2, -1), keepdim=True) + 1e-8
        saliency = saliency / denom
        
        # 【改进1】使用更小的核或去掉高斯模糊，减少扩散
        # saliency = TF.gaussian_blur(saliency, kernel_size=[5, 5], sigma=1.0)  # 原始
        saliency = TF.gaussian_blur(saliency, kernel_size=[3, 3], sigma=0.5)  # 更小的核
        
        # 【改进2】锐化处理：提升高值区域，抑制低值区域
        if getattr(self, 'sharpening_strength', 0.0) > 0.0:
            # 使用分位数拉伸
            quantile_50 = torch.quantile(saliency.flatten(), 0.5)
            quantile_90 = torch.quantile(saliency.flatten(), 0.9)
            # 将中位数以下的值压低
            low_factor = 1.0 - getattr(self, 'sharpening_strength', 0.0) * 0.7  # 0.3-1.0
            saliency = torch.where(saliency < quantile_50, saliency * low_factor, saliency)
            # 提升高分位数区域
            high_factor = 1.0 + getattr(self, 'sharpening_strength', 0.0) * 0.5  # 1.0-1.5
            saliency = torch.where(saliency > quantile_90, saliency * high_factor, saliency).clamp(0.0, 1.0)
        
        # 回到原图尺寸
        saliency = F.interpolate(saliency, size=image_bchw.shape[-2:], mode='bilinear', align_corners=False)
        return saliency.clamp(0.0, 1.0).detach()

    def _maybe_collect_cross_attn(self, outputs, num_img_tokens: int):
        """
        从模型输出中尽力收集跨模态 cross-attention 张量列表。
        返回: List[Tensor]，每个形状约为 [B, Heads, T_text, T_img]
        """
        candidates = []
        # 常见字段名尝试（不同 VLM 可能字段名不同）
        for key in [
            'cross_attentions', 'attentions', 'vision_attentions', 'text_attentions',
            'encoder_attentions', 'decoder_cross_attentions', 'conditional_attentions'
        ]:
            val = getattr(outputs, key, None)
            if val is None:
                continue
            if isinstance(val, (list, tuple)):
                for t in val:
                    if not torch.is_tensor(t):
                        continue
                    # 形状通常为 [B, Heads, T_q, T_kv]
                    if t.dim() == 4 and t.shape[-1] == num_img_tokens:
                        candidates.append(t)
        return candidates

    def _compute_single_map_xattn(self, image_bchw, qa_list):
        """
        使用 cross-attention × gradient 生成注意力图。
        若无法取得 cross-attn 或梯度，回退到像素梯度法。
        """
        if not qa_list:
            h, w = image_bchw.shape[-2:]
            return torch.zeros(1, 1, h, w, device=self.device, dtype=torch.float32)

        # 预处理
        image_resized = self.resize(image_bchw)
        pixel_values = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(image_resized)
        pixel_values = pixel_values.to(device=self.device, dtype=next(self.model.parameters()).dtype)

        num_img_tokens = int(getattr(self.model, "num_image_token", 256))
        grid_size = int(math.sqrt(max(1, num_img_tokens)))

        qa_maps = []
        for qa in qa_list:
            question, answer = qa['question'], qa['answer']
            input_ids, attention_mask, labels, image_flags = _build_internvl_inputs(
                question, answer, self.tokenizer, self.model, max_len=1024, num_patches=1
            )
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)
            image_flags = image_flags.to(self.device)

            # 允许梯度用于 cross-attn grad
            pv = pixel_values.detach().requires_grad_(True)

            outputs = self.model(
                pixel_values=pv,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                image_flags=image_flags,
                output_hidden_states=False,
                output_attentions=True,
                return_dict=True
            )

            loss = outputs.loss if outputs is not None else None
            if loss is None or torch.isnan(loss):
                return self._compute_single_map(image_bchw, qa_list)

            xattn_list = self._maybe_collect_cross_attn(outputs, num_img_tokens)
            if not xattn_list:
                return self._compute_single_map(image_bchw, qa_list)

            # 计算答案段起点
            user_text = f"<image>\n{question}".strip()
            image_block = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * num_img_tokens) + IMG_END_TOKEN
            user_with_visual = user_text.replace("<image>", image_block, 1)
            q_enc = self.tokenizer(user_with_visual, return_tensors='pt', padding=False, truncation=True, max_length=1024)
            q_len = q_enc["input_ids"].shape[1]

            per_layer = []
            # 对每个层的 attn 求对 loss 的梯度（若可用）
            grads = torch.autograd.grad(loss, xattn_list, retain_graph=False, allow_unused=True)
            any_valid = False
            for attn, g in zip(xattn_list, grads):
                if not torch.is_tensor(attn):
                    continue
                attn = attn.float()
                if g is None:
                    # 没有梯度，使用原始 attn 作为弱代理
                    g = torch.ones_like(attn, dtype=torch.float32)
                else:
                    g = F.relu(g.float())
                    any_valid = True
                # 仅答案段文本位置
                if attn.shape[-2] > q_len:
                    attn_ans = attn[:, :, q_len:, :]
                    g_ans = g[:, :, q_len:, :]
                else:
                    attn_ans = attn
                    g_ans = g
                cam = (attn_ans * g_ans).mean(dim=2).mean(dim=1)  # [B, T_img]
                cam = cam.reshape(cam.shape[0], 1, grid_size, grid_size)
                per_layer.append(cam)

            if not per_layer:
                return self._compute_single_map(image_bchw, qa_list)

            cam_mean = torch.stack(per_layer, dim=0).mean(dim=0)
            # 归一化 + 上采样
            cam_mean = cam_mean - cam_mean.amin(dim=(-2, -1), keepdim=True)
            cam_mean = cam_mean / (cam_mean.amax(dim=(-2, -1), keepdim=True) + 1e-8)
            cam_up = F.interpolate(cam_mean, size=image_bchw.shape[-2:], mode='bilinear', align_corners=False)
            qa_maps.append(cam_up)

        if not qa_maps:
            return self._compute_single_map(image_bchw, qa_list)

        saliency = torch.stack(qa_maps, dim=0).mean(dim=0)
        saliency = TF.gaussian_blur(saliency, kernel_size=[5, 5], sigma=1.0)
        return saliency.clamp(0.0, 1.0).detach()

    def _compute_map_by_method(self, image_bchw, qa_list):
        method = getattr(self, 'method', 'pixel_grad')
        if 'xattn_grad' in method:
            return self._compute_single_map_xattn(image_bchw, qa_list)
        if 'clip_text_match' in method:
            return self._compute_single_map_clip(image_bchw, qa_list)
        return self._compute_single_map(image_bchw, qa_list)

    def _extract_target_texts(self, qa_list):
        texts = []
        for qa in qa_list:
            ans = (qa.get('answer', '') or '').strip()
            val = ans.split(':', 1)[1].strip() if ':' in ans else ans
            if val:
                variants = [val, val.lower(), val.replace(' ', ''), val.title()]
                for v in variants:
                    if v and v not in texts:
                        texts.append(v)
        return texts or [""]  # 不再添加通用关键词

    # def _extract_target_texts(self, qa_list):
    #     texts = []
    #     for qa in qa_list:
    #         ans = (qa.get('answer', '') or '').strip()
    #         # Try to extract the phrase after the first ':' if present
    #         if ':' in ans:
    #             after = ans.split(':', 1)[1].strip()
    #             if after:
    #                 texts.append(after)
    #         # Fallback to entire answer
    #         if not texts and ans:
    #             texts.append(ans)
    #         # Also consider question keywords (e.g., "name")
    #         q = (qa.get('question', '') or '').lower()
    #         for key in ["name", "phone", "email", "address", "id", "birthday", "age"]:
    #             if key in q:
    #                 texts.append(key)
    #     # Deduplicate while preserving order
    #     uniq = []
    #     for t in texts:
    #         if t and t not in uniq:
    #             uniq.append(t)
    #     return uniq or ["name"]

    @torch.no_grad()
    def _compute_single_map_clip(self, image_bchw, qa_list):
        """
        使用 CLIP 文本-图像相似度在图像 patch 上生成注意力图（无梯度）。
        步骤：
          1) 提取答案短语/关键词，编码为文本特征
          2) 提取图像所有 patch 的视觉特征（排除 CLS）并投影
          3) 计算每个 patch 与文本特征的余弦相似度，得到 [grid, grid]
          4) 归一化并上采样到原图尺寸
        """
        h, w = image_bchw.shape[-2:]
        if not qa_list:
            return torch.zeros(1, 1, h, w, device=self.device, dtype=torch.float32)

        self._lazy_load_clip()

        # Prepare image for CLIP
        image_bchw = image_bchw.to(self.device)
        image_bchw = image_bchw.clamp(0.0, 1.0)
        # CLIPProcessor expects PIL or tensor in [0,1], convert batch of 1
        inputs = self._clip_processor(images=TF.to_pil_image(image_bchw[0].cpu()), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        # Texts to match
        target_texts = self._extract_target_texts(qa_list)
        text_inputs = self._clip_processor(text=target_texts, return_tensors="pt", padding=True)
        input_ids = text_inputs["input_ids"].to(self.device)
        attention_mask = text_inputs["attention_mask"].to(self.device)

        # Encode text and image
        outputs = self._clip_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True
        )

        # Text embeddings (projected and normalized)
        text_embeds = outputs.text_embeds  # [T, D]
        text_embeds = F.normalize(text_embeds, dim=-1)

        # Vision patch embeddings: take vision last_hidden_state excluding CLS, project then normalize
        vision_hidden = outputs.vision_model_output.last_hidden_state  # [1, 1+P, H]
        patch_tokens = vision_hidden[:, 1:, :]  # remove CLS -> [1, P, H]
        patch_proj = self._clip_model.visual_projection(patch_tokens)  # [1, P, D]
        patch_proj = F.normalize(patch_proj, dim=-1)

        # Similarity per text, then aggregate by max over texts (focus on any matching phrase)
        # sim: [T, 1, P] => aggregate -> [1, P]
        sim_list = []
        for t in range(text_embeds.shape[0]):
            te = text_embeds[t:t+1].unsqueeze(1)  # [1,1,D]
            sim = torch.matmul(patch_proj, te.transpose(-1, -2)).squeeze(-1)  # [1, P]
            sim_list.append(sim)
        sim_all = torch.stack(sim_list, dim=0).amax(dim=0)  # [1, P]

        # Reshape to grid
        num_patches = sim_all.shape[-1]
        grid = int(math.sqrt(num_patches))
        if grid * grid != num_patches:
            # Fallback: mean to a nearest square
            grid = int(math.sqrt(num_patches))
            keep = grid * grid
            sim_grid = sim_all[:, :keep].reshape(1, 1, grid, grid)
        else:
            sim_grid = sim_all.reshape(1, 1, grid, grid)

        # Normalize to [0,1]
        sim_grid = sim_grid - sim_grid.amin(dim=(-2, -1), keepdim=True)
        sim_grid = sim_grid / (sim_grid.amax(dim=(-2, -1), keepdim=True) + 1e-8)

        # Light smoothing and upsample to original size
        sim_grid = TF.gaussian_blur(sim_grid, kernel_size=[3, 3], sigma=0.5)
        sim_up = F.interpolate(sim_grid, size=(h, w), mode='bilinear', align_corners=False)

        # Optional: multiply by a fast textness prior to focus on text-like regions
        textness = self._compute_textness_map(image_bchw)  # [1,1,H,W]
        attn = sim_up * (textness.clamp(0.0, 1.0) ** self._textness_gamma)

        # Normalize to [0,1]
        attn = attn - attn.amin(dim=(-2, -1), keepdim=True)
        attn = attn / (attn.amax(dim=(-2, -1), keepdim=True) + 1e-8)
        return attn.clamp(0.0, 1.0)

    @torch.no_grad()
    def _compute_textness_map(self, image_bchw):
        """
        Fast textness prior using Sobel gradient magnitude on luminance.
        Text regions typically have dense high-frequency edges.
        Returns [1,1,H,W] in [0,1].
        """
        x = image_bchw.to(self.device).clamp(0.0, 1.0)
        # RGB to luminance
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        # Sobel kernels
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=gray.dtype, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=gray.dtype, device=self.device).view(1, 1, 3, 3)
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
        # Normalize locally
        mag = mag - mag.amin(dim=(-2, -1), keepdim=True)
        mag = mag / (mag.amax(dim=(-2, -1), keepdim=True) + 1e-8)
        # Slight blur to reduce noise
        mag = TF.gaussian_blur(mag, kernel_size=[3, 3], sigma=0.8)
        return mag

    def get_attention_map(self, images_bchw, privacy_qa_list, normal_qa_list=None):
        """
        images_bchw: [B,3,H,W], 0-1
        privacy_qa_list: List[List[qa_dict]] 长度为 B
        normal_qa_list: List[List[qa_dict]] 或 None
        return: [B,1,H,W]
        """
        self.model.eval()
        b, _, h, w = images_bchw.shape
        maps = []
        for i in range(b):
            img = images_bchw[i:i+1]
            qa_list = privacy_qa_list[i] if i < len(privacy_qa_list) else []
            method = getattr(self, 'method', 'pixel_grad')
            is_contrast = ('contrast' in method) or method == 'contrastive'
            if is_contrast:
                # base method inferred from method name
                base_method = 'pixel_grad'
                if 'xattn_grad' in method:
                    base_method = 'xattn_grad'
                elif 'clip_text_match' in method:
                    base_method = 'clip_text_match'
                # temporarily override for dispatch
                prev_method = self.method
                self.method = base_method
                attn_priv = self._compute_map_by_method(img, qa_list)
                qa_norm = (normal_qa_list[i] if (normal_qa_list is not None and i < len(normal_qa_list)) else [])
                attn_norm = self._compute_map_by_method(img, qa_norm)
                self.method = prev_method
                # Contrastive diff with ReLU and renormalize
                attn = (attn_priv - attn_norm).clamp(min=0.0)
                # avoid all-zero
                if float(attn.max()) > 0:
                    attn = attn / (attn.amax(dim=(-2, -1), keepdim=True) + 1e-8)
            else:
                attn = self._compute_map_by_method(img, qa_list)
            maps.append(attn)
            # 可选保存注意力图
            if self.save_dir is not None:
                try:
                    import torchvision.utils as vutils
                    # 保存注意力热力图（灰度）
                    attn_path = os.path.join(self.save_dir, f"attn_{i:04d}.png")
                    vutils.save_image(attn, attn_path)
                    # 叠加到原图上做可视化
                    attn_rgb = attn.repeat(1, 3, 1, 1)
                    overlay = (0.7 * img + 0.3 * attn_rgb).clamp(0.0, 1.0)
                    overlay_path = os.path.join(self.save_dir, f"overlay_{i:04d}.png")
                    vutils.save_image(overlay, overlay_path)
                except Exception:
                    pass
        return torch.cat(maps, dim=0)


