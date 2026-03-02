"""
动态教师选择与融合
支持所有 5 个专家的协同工作
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from config import cfg


class TeacherOrchestrator:
    """
    教师编排器
    负责场景感知、动态选择、输出对齐与融合
    """

    def __init__(self, experts_dict: Dict[str, nn.Module]):
        """
        Args:
            experts_dict: 专家模型字典
                {
                    'CountVid': CountVid(),
                    'GraspMamba': GraspMamba(),
                    'CrowdMPM': CrowdMPM(),
                    'OMAN': OMAN()
                }
        """
        self.experts = experts_dict
        self.scene_to_expert_map = self._build_scene_expert_map()

        print(f"[TeacherOrchestrator] 已加载 {len(experts_dict)} 个专家")
        print(f"[TeacherOrchestrator] 场景映射: {self.scene_to_expert_map}")

    def _build_scene_expert_map(self) -> Dict[int, List[str]]:
        """
        构建场景→专家映射表

        映射策略（基于7类场景）:
        0: Extreme_Dense_Static    → CrowdMPM (物理模拟擅长极端高密度)
        1: High_Dense_Directional  → OMAN (轨迹跟踪)
        2: Medium_Dense_Chaotic    → CountVid (鲁棒检测)
        3: Low_Dense_Sparse        → CountVid (精确检测)
        4: Abrupt_Change           → GraspMamba (语义鲁棒 / zero-shot)
        5: Open_World_Objects      → CountVid (开放世界能力)
        6: Uncertain_Mixed         → 所有可用专家
        """
        mapping = {
            0: ['CrowdMPM'],
            1: ['OMAN'],
            2: ['CountVid'],
            3: ['CountVid'],
            4: ['GraspMamba'],
            5: ['CountVid'],
            6: ['CountVid', 'OMAN', 'CrowdMPM', 'GraspMamba']
        }

        # 过滤未启用的专家
        filtered_mapping = {}
        for scene_id, expert_names in mapping.items():
            available = [name for name in expert_names if name in self.experts]
            if not available:
                # 回退到任意可用专家
                available = list(self.experts.keys())[:1] if self.experts else []
            filtered_mapping[scene_id] = available

        return filtered_mapping

    def select_teachers(
        self,
        scene_probs: torch.Tensor,
        strategy: str = 'soft',
        top_k: int = 2,
        temperature: float = 1.0
    ) -> Tuple[List[List[str]], Optional[torch.Tensor]]:
        """
        根据场景概率选择教师（Top-K稀疏门控）

        Args:
            scene_probs: (B, 7) 场景概率分布
            strategy: 'soft' (概率加权) 或 'hard' (argmax)
            top_k: Top-K 稀疏
            temperature: Softmax 温度

        Returns:
            selected_teachers: List[List[str]] 每样本的教师列表
            teacher_weights: (B, K) 权重矩阵（soft模式）或 None（hard模式）
        """
        B = scene_probs.shape[0]
        device = scene_probs.device

        if strategy == 'hard':
            # 硬选择：选择概率最高的场景
            scene_indices = scene_probs.argmax(dim=1).cpu().numpy()
            selected_teachers = []
            for idx in scene_indices:
                teachers = self.scene_to_expert_map[int(idx)]
                # 只选第一个教师
                selected_teachers.append(teachers[:1])
            return selected_teachers, None

        else:  # soft with Top-K
            selected_teachers = []
            all_weights = []

            for b in range(B):
                # 温度调节的 Softmax
                probs = scene_probs[b] / temperature
                probs = torch.softmax(probs, dim=0).cpu().numpy()

                # 收集所有可能的教师及其权重
                teacher_weight_dict = {}
                for scene_id, prob in enumerate(probs):
                    if prob > 0.01:  # 概率过滤
                        for teacher in self.scene_to_expert_map[scene_id]:
                            teacher_weight_dict[teacher] = teacher_weight_dict.get(teacher, 0) + prob

                # Top-K 稀疏选择
                sorted_items = sorted(teacher_weight_dict.items(), key=lambda x: -x[1])
                top_items = sorted_items[:top_k]

                # 归一化权重
                total = sum(w for _, w in top_items)
                if total > 0:
                    top_items = [(t, w / total) for t, w in top_items]
                else:
                    # 如果所有权重为0，均匀分配
                    top_items = [(t, 1.0 / len(top_items)) for t, _ in top_items]

                selected_teachers.append([t for t, _ in top_items])
                all_weights.append([w for _, w in top_items])

            # 补齐到最大K（方便批处理）
            max_k = max(len(w) for w in all_weights) if all_weights else 1
            padded_weights = torch.zeros(B, max_k, device=device)
            for b, weights in enumerate(all_weights):
                padded_weights[b, :len(weights)] = torch.tensor(weights, device=device)

            return selected_teachers, padded_weights

    def get_teacher_outputs(
        self,
        teachers_list: List[List[str]],
        inputs: Dict
    ) -> List[Dict[str, Dict]]:
        """
        获取教师输出（逐样本推理）

        Args:
            teachers_list: List[List[str]] 每样本的教师列表
            inputs: Dict 输入数据 {'frames': (B,T,3,H,W), ...}

        Returns:
            List[Dict[str, Dict]] 每样本的教师输出字典
                [
                    {'CountVid': {...}},  # 样本0
                    {'OMAN': {...}},     # 样本1
                    ...
                ]
        """
        B = len(teachers_list)
        all_outputs = []

        with torch.no_grad():
            for b in range(B):
                sample_outputs = {}

                for teacher_name in teachers_list[b]:
                    if teacher_name not in self.experts:
                        continue

                    expert = self.experts[teacher_name]

                    # 提取单样本输入
                    sample_input = {}
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            sample_input[k] = v[b:b+1]
                        elif isinstance(v, list):
                            sample_input[k] = [v[b]] if b < len(v) else v
                        else:
                            sample_input[k] = v

                    # 特殊处理：GraspMamba 需要文本描述（zero-shot 语义先验）
                    if teacher_name == 'GraspMamba':
                        if 'text_description' not in sample_input:
                            # 使用场景标签生成描述
                            sample_input['text_description'] = None

                    # 在调用专家前，将输入移动到专家所在设备，避免 device mismatch
                    try:
                        # determine expert device
                        try:
                            expert_device = next(expert.parameters()).device
                        except StopIteration:
                            expert_device = torch.device(cfg.device)

                        # move tensor inputs to expert device
                        for kk, vv in list(sample_input.items()):
                            if isinstance(vv, torch.Tensor):
                                sample_input[kk] = vv.to(expert_device)
                            elif isinstance(vv, list):
                                # move any tensor elements inside lists
                                sample_input[kk] = [x.to(expert_device) if isinstance(x, torch.Tensor) else x for x in vv]

                        output = expert(sample_input)
                        sample_outputs[teacher_name] = output
                    except Exception as e:
                        print(f"[TeacherOrchestrator 警告] {teacher_name} 推理失败 (样本 {b}): {e}")
                        continue

                all_outputs.append(sample_outputs)

        return all_outputs

    def fuse_teacher_outputs(
        self,
        teacher_outputs_list: List[Dict[str, Dict]],
        weights: Optional[torch.Tensor] = None
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        融合教师输出（保留batch维度）

        注意事项：
        - GraspMamba 不输出 density_map，不参与融合（提供语义先验）
        - OMAN 输出 flux 和 density_map
        - 其他专家输出 density_map

        Args:
            teacher_outputs_list: List[Dict[str, Dict]]
            weights: (B, K) 权重矩阵

        Returns:
            Dict {
                'density_map': (B, T, H, W),
                'confidence': (B,)
            } 或 None（无有效教师）
        """
        B = len(teacher_outputs_list)
        if B == 0:
            return None

        fused_densities = []
        fused_confidences = []

        for b in range(B):
            sample_teachers = teacher_outputs_list[b]

            if len(sample_teachers) == 0:
                # 无教师，返回零
                T, H, W = cfg.num_frames, cfg.student_output_size[0], cfg.student_output_size[1]
                fused_densities.append(torch.zeros(1, T, H, W, device=cfg.device))
                fused_confidences.append(torch.tensor([0.1], device=cfg.device))
                continue

            # 仅融合有 density_map 的教师
            densities = []
            confidences = []
            teacher_names = []

            for name, out in sample_teachers.items():
                if 'density_map' in out:
                    densities.append(out['density_map'])  # (1, T, H, W)
                    confidences.append(out['confidence'])  # (1,)
                    teacher_names.append(name)
                elif name == 'GraspMamba':
                    # GraspMamba 不参与密度融合，跳过（提供语义先验）
                    continue

            if len(densities) == 0:
                # 无密度图教师
                T, H, W = cfg.num_frames, cfg.student_output_size[0], cfg.student_output_size[1]
                fused_densities.append(torch.zeros(1, T, H, W, device=cfg.device))
                fused_confidences.append(torch.tensor([0.1], device=cfg.device))
                continue

            # 加权融合
            if weights is not None and weights[b].sum() > 0:
                # 找到对应的权重
                w_list = []
                all_teacher_names = list(sample_teachers.keys())
                for name in teacher_names:
                    if name in all_teacher_names:
                        idx = all_teacher_names.index(name)
                        if idx < weights.shape[1]:
                            w_list.append(weights[b, idx].item())
                        else:
                            w_list.append(0.0)
                    else:
                        w_list.append(0.0)

                w = torch.tensor(w_list, device=densities[0].device)
                w = w / (w.sum() + 1e-8)

                # 加权和
                density_fused = sum(d * w[i].item() for i, d in enumerate(densities))
                conf_fused = sum(c * w[i].item() for i, c in enumerate(confidences))
            else:
                # 均匀加权
                density_fused = sum(densities) / len(densities)
                conf_fused = sum(confidences) / len(confidences)

            fused_densities.append(density_fused)
            fused_confidences.append(conf_fused)

        # 拼接为 batch
        return {
            'density_map': torch.cat(fused_densities, dim=0),  # (B, T, H, W)
            'confidence': torch.cat(fused_confidences, dim=0)  # (B,)
        }

    def get_semantic_priors(
        self,
        teacher_outputs_list: List[Dict[str, Dict]]
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        提取语义先验（来自 GraspMamba）

        Returns:
            Dict {
                'semantic_embedding': (B, D),
                'scene_probs': (B, 7)
            } 或 None
        """
        B = len(teacher_outputs_list)

        embeddings = []
        scene_probs_list = []

        for b in range(B):
            if 'GraspMamba' in teacher_outputs_list[b]:
                out = teacher_outputs_list[b]['GraspMamba']
                embeddings.append(out.get('semantic_embedding', torch.zeros(1, 768, device=cfg.device)))
                scene_probs_list.append(out.get('scene_probs', torch.zeros(1, 7, device=cfg.device)))
            else:
                embeddings.append(torch.zeros(1, 768, device=cfg.device))
                scene_probs_list.append(torch.zeros(1, 7, device=cfg.device))

        if len(embeddings) == 0:
            return None

        return {
            'semantic_embedding': torch.cat(embeddings, dim=0),
            'scene_probs': torch.cat(scene_probs_list, dim=0)
        }