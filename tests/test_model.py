import torch

from spectra_v2 import SpectraConfig, SpectraEmbeddingModel


def tiny_config(mode: str = "dual") -> SpectraConfig:
    return SpectraConfig(
        vocab_size=101,
        hidden_size=32,
        num_heads=4,
        num_layers=2,
        max_positions=32,
        mixer_mode=mode,
        matryoshka_dims=(32, 16, 8),
    )


def test_output_contract_and_shapes():
    model = SpectraEmbeddingModel(tiny_config())
    ids = torch.randint(0, 101, (3, 12))
    mask = torch.ones_like(ids, dtype=torch.bool)
    output = model(ids, mask)
    assert output["embedding"].shape == (3, 32)
    assert output["matryoshka"][16].shape == (3, 16)
    assert output["diagnostics"]["branch_disagreement"].shape == (3,)


def test_padding_does_not_change_valid_embedding():
    torch.manual_seed(0)
    model = SpectraEmbeddingModel(tiny_config()).eval()
    short = torch.tensor([[4, 7, 9, 2]])
    short_mask = torch.ones_like(short, dtype=torch.bool)
    padded = torch.tensor([[4, 7, 9, 2, 33, 44, 55]])
    padded_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)

    with torch.no_grad():
        a = model(short, short_mask)["embedding"]
        b = model(padded, padded_mask)["embedding"]
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_semantic_ablation_has_no_spectral_module():
    model = SpectraEmbeddingModel(tiny_config("semantic"))
    assert all(block.spectral is None for block in model.blocks)
    assert all(block.semantic is not None for block in model.blocks)


def test_spectral_ablation_has_no_semantic_module():
    model = SpectraEmbeddingModel(tiny_config("spectral"))
    assert all(block.semantic is None for block in model.blocks)
    assert all(block.spectral is not None for block in model.blocks)


def test_dual_mode_backpropagates_through_both_paths():
    model = SpectraEmbeddingModel(tiny_config("dual"))
    ids = torch.randint(0, 101, (2, 10))
    mask = torch.ones_like(ids, dtype=torch.bool)
    loss = model(ids, mask)["embedding"].pow(2).mean()
    loss.backward()
    assert model.blocks[0].semantic.attn.in_proj_weight.grad is not None
    assert model.blocks[0].spectral.real_filter.grad is not None


def test_matryoshka_outputs_are_normalized():
    model = SpectraEmbeddingModel(tiny_config())
    ids = torch.randint(0, 101, (2, 8))
    mask = torch.ones_like(ids, dtype=torch.bool)
    output = model(ids, mask)["matryoshka"]
    for embedding in output.values():
        torch.testing.assert_close(
            embedding.norm(dim=-1),
            torch.ones(embedding.size(0)),
            atol=1e-5,
            rtol=1e-5,
        )


def test_invalid_configuration_is_rejected():
    try:
        SpectraConfig(vocab_size=100, hidden_size=30, num_heads=8)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid head geometry should fail")
