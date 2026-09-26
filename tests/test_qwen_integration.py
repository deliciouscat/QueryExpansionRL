"""Optional HF integration with a randomly initialized miniature Qwen, no downloads."""

import pytest
import torch

from query_expansion.agents import Agents
from query_expansion.agents.tiny import ByteTokenizer
from query_expansion.rewards import sft_loss
from query_expansion.utils.backward import GradientEngine, gradient, make_optimizers


def test_multimodal_checkpoint_text_loading_hybrid_backward(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    config = Qwen3_5Config(
        text_config={
            "vocab_size": 258,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 24,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "max_position_embeddings": 512,
            "tie_word_embeddings": True,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000,
                "partial_rotary_factor": 1.0,
                "mrope_section": [1, 1, 2],
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_heads": 4,
            "out_hidden_size": 32,
            "num_position_embeddings": 16,
        },
        tie_word_embeddings=True,
    )
    original = Qwen3_5ForConditionalGeneration(config)
    original.save_pretrained(tmp_path)
    expected = original.model.language_model.layers[0].mlp.up_proj.weight.detach().clone()
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: ByteTokenizer()
    )
    agent = Agents(
        backend="qwen",
        name=str(tmp_path),
        revision="0" * 40,
        rank=2,
        alpha=4,
        gradient_checkpointing=True,
    )
    assert torch.equal(expected, agent.model.model.layers[0].mlp.up_proj.base.weight)
    assert agent.model.lm_head.weight is agent.model.model.embed_tokens.weight
    assert not agent.model.lm_head.weight.requires_grad
    engine = GradientEngine(agent.model, make_optimizers(agent.groups, lora_lr=0.01, upper_lr=0.01))
    ids = torch.tensor([[2, 3, 4, 5, 6, 1]])
    labels = ids.clone()
    labels[:, :3] = -100
    before = {n: p.detach().clone() for n, p in agent.model.named_parameters()}

    @gradient(engine)
    def forward():
        return sft_loss(agent({"input_ids": ids, "attention_mask": torch.ones_like(ids)}), labels)

    with engine.window(1):
        forward()
    assert engine.global_step == 1
    assert any(
        not torch.equal(before[n], p) for n, p in agent.model.named_parameters() if "lora_B" in n
    )
    assert any(
        not torch.equal(before[n], p)
        for n, p in agent.model.named_parameters()
        if n.startswith("model.layers.12.")
    )
    assert all(
        torch.equal(before[n], p) for n, p in agent.model.named_parameters() if not p.requires_grad
    )
    completion = agent.generate("car", "en")
    assert 1 <= len(completion.tokens) <= 64
