# 🚀 QUICK START GUIDE - For Your Specific Dataset

## Your Dataset Structure
I can see you have:
```
C:\Users\vishw\Downloads\EPICS\HVDROPDB_RetCam_Neo_Classification\
├── RetCam_ROP/      (ROP images from RetCam system)
├── Neo_Normal/      (Normal images from Neo system)
├── Neo_ROP/         (ROP images from Neo system)
└── RetCam_Normal/   (Normal images from RetCam system)
```

## 📋 Complete Setup (3 Steps)

### STEP 1: Install Dependencies
```bash
pip install torch torchvision numpy pillow matplotlib seaborn scikit-learn tqdm
```

### STEP 2: Prepare Your Dataset

**Run the preparation script:**
```bash
python prepare_dataset.py
```

**What it does:**
- Combines RetCam_ROP + Neo_ROP → ROP class
- Combines RetCam_Normal + Neo_Normal → Normal class
- Splits into 70% train, 15% validation, 15% test
- Renames files to avoid conflicts (e.g., ROP_RetCam_32.png)

**Output structure:**
```
C:\Users\vishw\Downloads\EPICS\HVDROPDB_prepared\
├── train/
│   ├── ROP/
│   │   ├── ROP_RetCam_32.png
│   │   ├── ROP_Neo_45.png
│   │   └── ...
│   └── Normal/
│       ├── Normal_RetCam_12.png
│       ├── Normal_Neo_67.png
│       └── ...
├── val/
│   ├── ROP/
│   └── Normal/
└── test/
    ├── ROP/
    └── Normal/
```

### STEP 3: Train the Model
```bash
python rop_cnn_training.py
```

That's it! The script will automatically:
- Load the prepared dataset
- Train with mixed precision (FP16) to save memory
- Save checkpoints every 5 epochs
- Generate plots and confusion matrices
- Save the best model

---

## ⚙️ Configuration (Already Set for You!)

The training script is already configured for your hardware:

```python
# In rop_cnn_training.py - Config class
DATASET_PATH = r"C:\Users\vishw\Downloads\EPICS\HVDROPDB_prepared"
IMAGE_SIZE = (2048, 2048)          # Full resolution
MODEL_NAME = "efficientnet_b0"     # Best balance of speed/accuracy
BATCH_SIZE = 2                     # Safe for your 12GB VRAM
GRADIENT_ACCUMULATION_STEPS = 4    # Effective batch size = 8
MIXED_PRECISION = True             # CRITICAL - saves 40% memory!
NUM_EPOCHS = 50
```

**You don't need to change anything unless you want to!**

---

## 📊 Expected Output

### During Training:
```
================================================================================
Epoch 1/50
Learning Rate: 0.000100
================================================================================

Epoch 1/50 [TRAIN]: 100%|████████| 250/250 [05:23<00:00, loss=0.5234, acc=78.50%]
Epoch 1/50 [VAL]:   100%|████████|  50/50  [01:12<00:00, loss=0.4123, acc=82.00%]

📊 Epoch 1 Summary:
   Train Loss: 0.5234 | Train Acc: 78.50%
   Val Loss:   0.4123 | Val Acc:   82.00%
💾 Saved BEST model to checkpoints/best_model.pth
```

### Files Generated:
```
checkpoints/
├── best_model.pth                    ← Your best model
├── checkpoint_epoch_5.pth            ← Backup at epoch 5
├── checkpoint_epoch_10.pth           ← Backup at epoch 10
├── training_history.png              ← Loss/accuracy graphs
├── confusion_matrix_validation.png   ← Confusion matrix
└── training_history.json             ← Training metrics
```

---

## 🎮 GPU Memory Monitor

**Open a second terminal and run:**
```bash
nvidia-smi -l 1
```

This shows real-time GPU usage. You should see:
- **Memory Used:** ~5-9 GB (out of 12 GB)
- **GPU Utilization:** 80-100%
- **Temperature:** Should stay under 85°C

---

## 🔧 If You Get "Out of Memory" Error

**Option 1:** Reduce batch size
```python
BATCH_SIZE = 1  # Down from 2
GRADIENT_ACCUMULATION_STEPS = 8  # Up from 4 (keeps effective batch size same)
```

**Option 2:** Use a lighter model
```python
MODEL_NAME = "mobilenet_v2"  # Instead of efficientnet_b0
```

**Option 3:** Both options above

---

## 📈 Test Your Trained Model

After training completes, test on a single image:

```bash
python inference.py --checkpoint checkpoints/best_model.pth --image path/to/test_image.png
```

Test on a folder:
```bash
python inference.py --checkpoint checkpoints/best_model.pth --folder path/to/test_folder/
```

Evaluate accuracy on test set:
```bash
python inference.py --checkpoint checkpoints/best_model.pth --eval_folder HVDROPDB_prepared/test/ROP --true_label ROP
```

---

## 💡 Pro Tips

1. **First run:** Use `NUM_EPOCHS = 5` to test everything works, then increase to 50
2. **Monitor training:** Watch the validation accuracy - if it stops improving, you can stop early
3. **Data augmentation:** Already included (rotation, flipping, color jitter)
4. **Save often:** Checkpoints save every 5 epochs automatically
5. **Your 32GB RAM:** Perfect for data loading - no swapping issues!

---

## 📞 Common Issues

**"ModuleNotFoundError: No module named 'torch'"**
→ Run: `pip install torch torchvision`

**"FileNotFoundError: Dataset not found"**
→ Did you run `prepare_dataset.py` first?
→ Check the path in `DATASET_PATH`

**Training is slow**
→ Make sure `MIXED_PRECISION = True`
→ Close other GPU programs (Chrome, games)
→ Check `nvidia-smi` - GPU should be at 80-100% usage

**Low accuracy (<70%)**
→ Medical imaging can be challenging
→ Try training for more epochs (100+)
→ Check if dataset is balanced (equal ROP/Normal images)

---

## ✅ Checklist Before Training

- [ ] Installed PyTorch: `pip install torch torchvision`
- [ ] Ran `prepare_dataset.py` successfully
- [ ] Verified `HVDROPDB_prepared` folder exists
- [ ] Updated `DATASET_PATH` in `rop_cnn_training.py` (if needed)
- [ ] At least 20GB free disk space for checkpoints
- [ ] GPU memory is available (close other programs)

---

## 🎯 Your Exact Commands (Copy-Paste Ready!)

```bash
# 1. Install dependencies
pip install torch torchvision numpy pillow matplotlib seaborn scikit-learn tqdm

# 2. Prepare dataset
cd C:\Users\vishw\Downloads\EPICS
python prepare_dataset.py

# 3. Train model (after step 2 completes)
python rop_cnn_training.py

# 4. Monitor GPU (in separate terminal)
nvidia-smi -l 1

# 5. After training, test on image
python inference.py --checkpoint checkpoints/best_model.pth --image test.png
```

---

**Everything is ready! Your hardware can absolutely handle this.** 🚀

Expected total time: ~3-4 hours for 50 epochs
