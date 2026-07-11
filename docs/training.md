# Training Pipelines & Self-Improvement Loops

GenMusic VN provides automated training pipelines for both the Text Classifier and the Self-authored Music Diffusion Model.

---

## 1. Text Model Training & Self-Improvement

The classification model (Naive Bayes) can be trained locally or on Kaggle.

### Basic Local Training
Train the text model using local datasets:
```powershell
uv run python -m genmusic_vn.cli train-text-model --local --samples 800 --model-out data/trained_models/genmusic_text_model.json
```

### Self-Improvement Loop (`self-improve`)
The self-improvement cycle automates model upgrades through active learning:
1. **Initialize:** Loads the baseline text classification model.
2. **Simulate:** Generates a synthetic test set mimicking user queries.
3. **Predict:** Feeds the test queries into the classifier.
4. **Evaluate:** Grades outputs based on keyword recall, emotion consistency, and rhythm metrics.
5. **Identify Weaknesses:** Filters out cases where the classification confidence or matching scores are low.
6. **Augment & Retrain:** Augments the training set with corrected weak samples and retrains the model.

```powershell
uv run python -m genmusic_vn.cli self-improve --iterations 3 --samples 640 --eval-count 24 --out outputs/self_improve
```

---

## 2. Music Diffusion Model Training

The diffusion model is trained to convert text condition features into Mel-spectrogram matrices.

### Dataset Preparation
Generate a synthetic dataset of Mel-spectrogram tensor checkpoints (`.pt` files) on disk:
```powershell
uv run python -m genmusic_vn.cli make-random-dataset --out data/random_self_diffusion_training --count 16 --frames 128 --target-gb 1.0
```

### Dataset Validation
Verify the dataset schema, including file directories, metadata indices, and tensor dimensions:
```powershell
uv run python -m genmusic_vn.cli validate-dataset --dataset data/random_self_diffusion_training
```

### Model Training
Run the training loop using the `AdamW` optimizer:
```powershell
uv run python -m genmusic_vn.cli train-self --dataset data/random_self_diffusion_training --checkpoint outputs/self_music_checkpoint.pt --epochs 2 --batch-size 4
```
This loop loads target tensors, injects noise according to the diffusion schedule, computes the MSE loss backpropagated through time step embeddings, and saves a PyTorch checkpoint.
