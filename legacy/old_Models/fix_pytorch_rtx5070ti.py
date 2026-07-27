"""
PyTorch Reinstallation Script for RTX 5070Ti (Blackwell Architecture)

Your GPU has CUDA capability sm_120, which requires PyTorch nightly build.
This script will help you reinstall the correct version.
"""

print("="*80)
print("PYTORCH REINSTALLATION GUIDE FOR RTX 5070Ti")
print("="*80)

print("""
Your RTX 5070Ti uses the Blackwell architecture (CUDA sm_120) which is newer
than what the stable PyTorch release supports. You need PyTorch nightly build.

SOLUTION - Run these commands in order:
""")

print("="*80)
print("STEP 1: Uninstall current PyTorch")
print("="*80)
print("""
pip uninstall torch torchvision torchaudio -y
""")

print("="*80)
print("STEP 2: Install PyTorch Nightly (supports RTX 5070Ti)")
print("="*80)
print("""
# Option A: CUDA 12.4 (Recommended)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# Option B: CUDA 12.1 (if Option A fails)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
""")

print("="*80)
print("STEP 3: Verify Installation")
print("="*80)
print("""
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
""")

print("="*80)
print("ALTERNATIVE: Use CPU-only PyTorch (slower but works immediately)")
print("="*80)
print("""
If you want to test the code quickly without GPU:

1. In rop_cnn_training.py, change:
   DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   to:
   DEVICE = torch.device("cpu")  # Force CPU

2. This will work but be MUCH slower (10-20x slower)
   Only use this for testing, not actual training.
""")

print("="*80)
print("FULL REINSTALLATION COMMANDS (Copy-Paste)")
print("="*80)
print("""
# 1. Uninstall old PyTorch
pip uninstall torch torchvision torchaudio -y

# 2. Install PyTorch nightly with CUDA 12.4 support
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# 3. Verify it works
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# 4. If verification shows CUDA=True, you're ready!
python rop_cnn_training.py
""")

print("="*80)
print("WHAT CHANGED IN YOUR ERROR")
print("="*80)
print("""
❌ Current PyTorch supports: sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_90
❌ Your GPU needs: sm_120 (Blackwell architecture)

✅ PyTorch Nightly supports: All of the above + sm_120 (your GPU!)

This is a common issue with brand new GPUs. The nightly build has the fix.
""")

print("="*80)
print("EXPECTED FILE SIZE")
print("="*80)
print("""
PyTorch nightly with CUDA support is ~2.5GB download.
Make sure you have enough disk space and a stable internet connection.
""")

print("="*80)
print("TROUBLESHOOTING")
print("="*80)
print("""
If you get "Could not find a version that satisfies the requirement":
- Try CUDA 12.1 instead of 12.4
- Check your internet connection
- Make sure pip is updated: pip install --upgrade pip

If installation succeeds but CUDA still unavailable:
- Check NVIDIA drivers are updated (should be latest for RTX 5070Ti)
- Restart your computer after installation
- Run: nvidia-smi to verify driver is working
""")

print("="*80)
print("SUMMARY")
print("="*80)
print("""
Your code is perfect! Your GPU is perfect! 
You just need the newer PyTorch version that supports your GPU.

Run the commands above, then run your training script again.
Expected training time will be the same: ~3-4 hours for 50 epochs.
""")

print("="*80)
