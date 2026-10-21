"""可导出为 ONNX / TensorRT 的决策模块（torch 侧）。

核心设计决定：**固定 256 槽输出，K 的变化只发生在 Python 侧。**

两条候选设计（来源：Jev 架构逆向文，第三方、非官方）：

- **pointer 式**：读数位隐状态与每个候选自身的隐状态做点积 → 输出宽度 = K（动态）。
- **final-position head（固定 256-slot）**：读数位隐状态过 ``Linear(H, 256)`` → 宽度恒 256。

本模块用 **fixed-slots 作默认**，理由是纯工程性的但后果很大：

1. 输出形状恒定 ⇒ **TensorRT 引擎不需要为不同 K 重建**。K 从 2 到 255 只换 Python 侧 mask，
   引擎一次构建、长期复用。（引擎与 shape profile 绑定，动态宽度会逼出成堆 profile 甚至重 build。）
2. K 的上限由 ``schema.MAX_CHOICE_OPTIONS``（255）兜住，与 System One 验证层一致。
3. 代价：超出 K 的槽是**未受约束的输出**，必须显式 mask 再归一化——见 ``spec.mask_and_normalize``。

``noul`` 也走同一条路：把"是/否"两个描述当候选写进输入，K = 2。
于是整个服务只有一个算子图，没有分支。

本模块需要 torch；后处理与形状契约在 ``spec.py``（纯标准库）。
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .spec import N_SLOTS, ExportSpec, assert_export_contract

HeadKind = Literal["slots", "pointer"]


class DecisionModule(nn.Module):
    """``backbone`` + 决策头，导出成单个 ONNX 图。

    输入（全是常规算子，无自定义 op，TRT 可直接解析）::

        input_ids      : int64 (B, L)
        attention_mask : int64 (B, L)
        read_index     : int64 (B,)     ← 读数位在该序列中的下标

    输出::

        slot_logits    : float32 (B, n_slots)   ← 恒定形状
        read_hidden    : float32 (B, H)         ← 供诊断与 correctness 头使用
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        hidden_size: int,
        n_slots: int = N_SLOTS,
        head: HeadKind = "slots",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.n_slots = n_slots
        self.head_kind = head
        self.slot_head = nn.Linear(hidden_size, n_slots, bias=True)

    def _hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            hidden = out[0] if isinstance(out, (tuple, list)) else out
        return hidden

    @staticmethod
    def _read(hidden: torch.Tensor, read_index: torch.Tensor) -> torch.Tensor:
        """按 ``read_index`` 取读数位隐状态 → (B, H)。

        用三维索引 ``view(B,1,1).expand(B,1,H)`` 是 ``gather`` 要求的形状；
        写成二维会**静默**沿错误的轴取（不报错、形状也可能对），
        所以这里带一条形状断言。
        """
        b, _, h = hidden.shape
        if read_index.shape != (b,):
            raise ValueError(f"read_index 形状应为 ({b},)，得到 {tuple(read_index.shape)}")
        idx = read_index.view(b, 1, 1).expand(b, 1, h)
        gathered = hidden.gather(1, idx)
        assert gathered.shape == (b, 1, h), gathered.shape
        return gathered.squeeze(1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        read_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self._hidden(input_ids, attention_mask)
        read_hidden = self._read(hidden, read_index)
        if self.head_kind != "slots":
            raise NotImplementedError(
                "pointer 头需要把候选位集合作为额输入才能导出；默认请用 slots 头"
            )
        return self.slot_head(read_hidden), read_hidden


class TwoStageDecoder(nn.Module):
    """Qwen-Kev 路径的「前缀复用」两段式。

    这是两条路线在部署侧的真正取舍点：

    - **ModernBERT（双向）**：state 的表示**依赖**问题与候选 token（彼此互相 attend），
      所以**不能**缓存 state 再换问题复用，只能整条重算——靠把多个问题压进同一个 batch 摊薄。
    - **Qwen-Kev（因果）**：state 是**前缀**，问题只读它、它不读问题 ⇒ state 的 KV 可缓存，
      每个问题只跑后缀。

    所以「state 只编码一次」这件事，只有因果主干能做到。
    """

    def __init__(self, backbone: nn.Module, *, hidden_size: int, n_slots: int = N_SLOTS) -> None:
        super().__init__()
        self.backbone = backbone
        self.slot_head = nn.Linear(hidden_size, n_slots, bias=True)

    def prefill(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """编码 state 前缀，返回 (read_hidden, past_key_values)。

        这一段只跑一次；后续每个问题复用 ``past_key_values``。
        """
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        hidden = out.last_hidden_state
        return hidden[:, -1, :], out.past_key_values

    def tail(self, input_ids: torch.Tensor, past_key_values, read_index: torch.Tensor) -> torch.Tensor:
        """在缓存前缀之后跑问题后缀，并在 ``read_index`` 处读数。

        ⚠️ ``read_index`` 是**后缀内部**的下标，不是整条序列的下标。
        混用两者会读到前缀里某个位置——形状对、不报错、答案系统性偏错。
        """
        out = self.backbone(
            input_ids=input_ids, past_key_values=past_key_values, use_cache=True
        )
        hidden = out.last_hidden_state
        read = self._gather(hidden, read_index)
        return self.slot_head(read)

    @staticmethod
    def _gather(hidden: torch.Tensor, read_index: torch.Tensor) -> torch.Tensor:
        b, _, h = hidden.shape
        idx = read_index.view(b, 1, 1).expand(b, 1, h)
        return hidden.gather(1, idx).squeeze(1)


def load_for_export(ckpt_dir, *, n_slots: int = N_SLOTS, hidden_size: int = 768) -> DecisionModule:
    """从训练产物加载并组装。骨架：按你实际的保存格式改这一处即可。"""
    payload = torch.load(ckpt_dir / "decision_module.pt", map_location="cpu", weights_only=False)
    module = DecisionModule(payload["backbone"], hidden_size=hidden_size, n_slots=n_slots)
    module.load_state_dict(payload["state_dict"], strict=True)
    module.eval()
    return module


def contract_ok(spec: ExportSpec) -> bool:
    assert_export_contract(spec)
    return True
