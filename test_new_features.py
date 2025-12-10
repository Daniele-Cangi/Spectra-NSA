#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for new features: Resume Training, Gradient Checkpointing, Multi-GPU
"""

import os
import sys
import torch
import tempfile
import shutil
from pathlib import Path

# Fix Windows console encoding issues
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from anomalous_embedding_ultimate import Config, FullEmbedder, Trainer, set_seed, apply_preset
from transformers import AutoTokenizer


def test_config_gradient_checkpointing():
    """Test that gradient checkpointing flag works in config"""
    print("\n[TEST 1] Testing gradient checkpointing configuration...")

    cfg = Config()
    assert cfg.gradient_checkpointing == False, "Default should be False"

    cfg.gradient_checkpointing = True
    assert cfg.gradient_checkpointing == True, "Should be able to enable"

    print("✓ Gradient checkpointing configuration works")
    return True


def test_model_with_gradient_checkpointing():
    """Test that model can be created with gradient checkpointing enabled"""
    print("\n[TEST 2] Testing model creation with gradient checkpointing...")

    cfg = Config()
    cfg.gradient_checkpointing = True
    cfg.hidden_size = 128  # Small for testing
    cfg.num_layers = 2
    cfg.num_heads = 4
    cfg.spectral_dim = 32

    model = FullEmbedder(cfg)
    assert model.backbone.gradient_checkpointing == True, "Model should have gradient checkpointing enabled"

    print("✓ Model with gradient checkpointing created successfully")
    return True


def test_checkpoint_save_load():
    """Test checkpoint save and load with new fields"""
    print("\n[TEST 3] Testing checkpoint save/load with resume training fields...")

    cfg = Config()
    cfg.hidden_size = 128
    cfg.num_layers = 2
    cfg.num_heads = 4
    cfg.spectral_dim = 32
    cfg.epochs = 1
    cfg.batch_train = 2
    cfg.save_dir = tempfile.mkdtemp()

    try:
        set_seed(42)
        model = FullEmbedder(cfg)
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True)

        # Create minimal trainer (we won't actually train)
        from torch.utils.data import DataLoader, TensorDataset
        dummy_data = TensorDataset(torch.zeros(10, 3), torch.zeros(10, 3), torch.zeros(10, 3))
        dummy_loader = DataLoader(dummy_data, batch_size=2)

        trainer = Trainer(cfg, model, dummy_loader, dummy_loader, tokenizer)

        # Set some state
        trainer.global_step = 100
        trainer.current_epoch = 1
        trainer.best_sts = 0.75
        trainer.patience_counter = 2
        trainer.checkpoint_counter = 5

        # Save checkpoint
        checkpoint_path = os.path.join(cfg.save_dir, "test_checkpoint.pt")
        trainer.save_ckpt("test_checkpoint")

        assert os.path.exists(checkpoint_path), "Checkpoint should be saved"

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Verify all fields exist
        assert "model" in ckpt, "Should have model state"
        assert "opt" in ckpt, "Should have optimizer state"
        assert "lr_sched" in ckpt, "Should have scheduler state"
        assert "step" in ckpt, "Should have step"
        assert "epoch" in ckpt, "Should have epoch"
        assert "best_sts" in ckpt, "Should have best_sts"
        assert "best_val_loss" in ckpt, "Should have best_val_loss"
        assert "checkpoint_counter" in ckpt, "Should have checkpoint_counter"
        assert "patience_counter" in ckpt, "Should have patience_counter"
        assert "rng_state" in ckpt, "Should have RNG states"

        # Verify values
        assert ckpt["step"] == 100, f"Step should be 100, got {ckpt['step']}"
        assert ckpt["epoch"] == 1, f"Epoch should be 1, got {ckpt['epoch']}"
        assert abs(ckpt["best_sts"] - 0.75) < 1e-6, f"Best STS should be 0.75, got {ckpt['best_sts']}"
        assert ckpt["patience_counter"] == 2, f"Patience counter should be 2, got {ckpt['patience_counter']}"
        assert ckpt["checkpoint_counter"] == 5, f"Checkpoint counter should be 5, got {ckpt['checkpoint_counter']}"

        # Test RNG state
        assert "python" in ckpt["rng_state"], "Should have Python RNG state"
        assert "numpy" in ckpt["rng_state"], "Should have NumPy RNG state"
        assert "torch" in ckpt["rng_state"], "Should have PyTorch RNG state"

        print("✓ Checkpoint save/load with all resume fields works correctly")
        return True

    finally:
        # Cleanup
        if os.path.exists(cfg.save_dir):
            shutil.rmtree(cfg.save_dir)


def test_resume_training():
    """Test that resume training loads state correctly"""
    print("\n[TEST 4] Testing resume training state restoration...")

    cfg = Config()
    cfg.hidden_size = 128
    cfg.num_layers = 2
    cfg.num_heads = 4
    cfg.spectral_dim = 32
    cfg.epochs = 1
    cfg.batch_train = 2
    cfg.save_dir = tempfile.mkdtemp()

    try:
        set_seed(42)
        model = FullEmbedder(cfg)
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True)

        from torch.utils.data import DataLoader, TensorDataset
        dummy_data = TensorDataset(torch.zeros(10, 3), torch.zeros(10, 3), torch.zeros(10, 3))
        dummy_loader = DataLoader(dummy_data, batch_size=2)

        # Create first trainer and save
        trainer1 = Trainer(cfg, model, dummy_loader, dummy_loader, tokenizer)
        trainer1.global_step = 200
        trainer1.current_epoch = 2
        trainer1.best_sts = 0.82
        trainer1.patience_counter = 3
        trainer1.checkpoint_counter = 7

        checkpoint_path = os.path.join(cfg.save_dir, "resume_test.pt")
        trainer1.save_ckpt("resume_test")

        # Create new trainer and load
        model2 = FullEmbedder(cfg)
        trainer2 = Trainer(cfg, model2, dummy_loader, dummy_loader, tokenizer)

        # Verify initial state is different
        assert trainer2.global_step == 0, "New trainer should start at step 0"
        assert trainer2.current_epoch == 0, "New trainer should start at epoch 0"

        # Load checkpoint
        trainer2.load_checkpoint(checkpoint_path)

        # Verify state was restored
        assert trainer2.global_step == 200, f"Step should be 200, got {trainer2.global_step}"
        assert trainer2.current_epoch == 2, f"Epoch should be 2, got {trainer2.current_epoch}"
        assert abs(trainer2.best_sts - 0.82) < 1e-6, f"Best STS should be 0.82, got {trainer2.best_sts}"
        assert trainer2.patience_counter == 3, f"Patience should be 3, got {trainer2.patience_counter}"
        assert trainer2.checkpoint_counter == 7, f"Checkpoint counter should be 7, got {trainer2.checkpoint_counter}"

        print("✓ Resume training state restoration works correctly")
        return True

    finally:
        if os.path.exists(cfg.save_dir):
            shutil.rmtree(cfg.save_dir)


def test_multi_gpu_info():
    """Test that multi-GPU info is correctly displayed"""
    print("\n[TEST 5] Testing multi-GPU information display...")

    cfg = Config()
    cfg.hidden_size = 128
    cfg.num_layers = 2
    cfg.num_heads = 4
    cfg.spectral_dim = 32
    cfg.save_dir = tempfile.mkdtemp()

    try:
        model = FullEmbedder(cfg)
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True)

        from torch.utils.data import DataLoader, TensorDataset
        dummy_data = TensorDataset(torch.zeros(10, 3), torch.zeros(10, 3), torch.zeros(10, 3))
        dummy_loader = DataLoader(dummy_data, batch_size=2)

        trainer = Trainer(cfg, model, dummy_loader, dummy_loader, tokenizer)

        # Check that trainer has accelerator with num_processes
        assert hasattr(trainer.acc, 'num_processes'), "Accelerator should have num_processes"
        assert hasattr(trainer.acc, 'device'), "Accelerator should have device"

        print(f"  Detected {trainer.acc.num_processes} process(es)")
        print(f"  Device: {trainer.acc.device}")
        print("✓ Multi-GPU information accessible")
        return True

    finally:
        if os.path.exists(cfg.save_dir):
            shutil.rmtree(cfg.save_dir)


def test_gradient_checkpointing_forward():
    """Test that gradient checkpointing works during forward pass"""
    print("\n[TEST 6] Testing gradient checkpointing during forward pass...")

    cfg = Config()
    cfg.gradient_checkpointing = True
    cfg.hidden_size = 128
    cfg.num_layers = 2
    cfg.num_heads = 4
    cfg.spectral_dim = 32

    model = FullEmbedder(cfg)
    model.train()  # Must be in training mode for checkpointing to activate

    # Create dummy input
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # Forward pass should work with gradient checkpointing
    try:
        output = model({"input_ids": input_ids, "attention_mask": attention_mask})
        assert "embedding_semantic" in output, "Should have semantic embeddings"
        assert "embedding_ranking" in output, "Should have ranking embeddings"
        print("✓ Gradient checkpointing forward pass works")
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("="*80)
    print("TESTING NEW FEATURES: Resume Training, Gradient Checkpointing, Multi-GPU")
    print("="*80)

    tests = [
        test_config_gradient_checkpointing,
        test_model_with_gradient_checkpointing,
        test_checkpoint_save_load,
        test_resume_training,
        test_multi_gpu_info,
        test_gradient_checkpointing_forward,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"✗ {test.__name__} failed")
        except Exception as e:
            failed += 1
            print(f"✗ {test.__name__} raised exception: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print(f"TEST RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("="*80)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
