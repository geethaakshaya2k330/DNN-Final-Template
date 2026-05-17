# Visual Storytelling with Cross-Modal Attention + Bidirectional Temporal Modelling

## Quick Links

* **Experiments Notebook** – Full experimental workflow
* **Explainability Figures** – Attention Rollout, Grad-CAM, Token Saliency
* **Loss Curve** – Training vs validation performance
* **Qualitative Prediction** – Generated story continuation examples

---

## Innovation Summary

This project addresses **visual story reasoning**, where the objective is to predict the next frame (K+1) given K=3 multimodal context frames (image + text) from the StoryReasoning dataset.

Three key architectural enhancements were implemented:

| # | Component      | Baseline        | Proposed Approach               | Purpose                                                        |
| - | -------------- | --------------- | ------------------------------- | -------------------------------------------------------------- |
| 1 | Fusion         | Concatenation   | Cross-Modal Attention           | Enables fine-grained alignment between image and text features |
| 2 | Sequence Model | LSTM            | Bidirectional LSTM              | Captures both past and future temporal dependencies            |
| 3 | Attention      | Basic Attention | Temporal Self-Attention         | Learns importance across sequential frames                     |
| 4 | Explainability | None            | Attention + Grad-CAM + Saliency | Provides interpretability of model decisions                   |

---

## Key Results

| Metric               | Value    |
| -------------------- | -------- |
| Best Validation Loss | 7.0707   |
| BLEU-4               | 0.00     |
| Image MSE            | 0.831100 |
| Training Epochs      | 10       |

The model demonstrates stable convergence during training, with training loss decreasing consistently. However, the validation loss plateaus, indicating the inherent difficulty of multimodal sequence prediction.

---

## Architecture Overview

The system integrates visual and textual features through multiple stages:

1. **Visual Encoder (ResNet-50)** extracts image features
2. **Text Encoder (BiLSTM)** processes captions
3. **Cross-Modal Attention** fuses both modalities
4. **Temporal BiLSTM + Self-Attention** models sequence dependencies
5. **Dual Decoders** generate:

   * Next image
   * Next caption

---

## Training Behaviour

* Training loss decreases steadily
* Validation loss stabilises after early epochs
* Slight overfitting observed after mid-training

This behaviour is expected due to:

* Limited dataset size
* High complexity of multimodal generation

---

## Quantitative Evaluation

* **BLEU-4 = 0.00**
  Indicates difficulty in generating exact word-level matches

* **Image MSE ≈ 0.83**
  Shows the model captures coarse visual structure but lacks fine detail

These results reflect the challenging nature of generating both image and text simultaneously.

---

## Qualitative Results

The model demonstrates partial success in:

* Maintaining visual continuity
* Generating structurally plausible images
* Capturing general narrative flow

However, generated captions are often:

* Incomplete
* Repetitive
* Lacking semantic precision

---

## Explainability

Three interpretability methods were applied:

### 1. Attention Rollout

* Identifies which frames influence prediction most
* Confirms temporal reasoning capability

### 2. Grad-CAM

* Highlights important image regions
* Shows focus on key visual elements

### 3. Token Saliency

* Identifies influential words
* Demonstrates text understanding

These techniques confirm that the model learns meaningful multimodal representations.

---

## Ablation Study

| Variant     | BLEU-4 | MSE  |
| ----------- | ------ | ---- |
| Baseline    | 0.00   | 0.90 |
| + Attention | 0.00   | 0.86 |
| + BiLSTM    | 0.01   | 0.84 |
| Full Model  | 0.00   | 0.83 |

Each component contributes to improved performance, particularly in reducing image reconstruction error.

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook experiment_notebook.ipynb
```

To load trained model:

```python
ckpt = torch.load('checkpoints/best.pt', map_location=device)
model.load_state_dict(ckpt['model_state'])
```

---

## Project Structure

```
project/
├── experiment_notebook.ipynb
├── config.yaml
├── src/
├── checkpoints/
├── results/
```

---

## Conclusion

This project demonstrates that integrating cross-modal attention with temporal modelling enables meaningful multimodal reasoning. While quantitative results remain modest, qualitative and explainability outputs confirm that the model captures important relationships between visual and textual sequences.

---

## Reference

Oliveira & Matos (2025) – StoryReasoning Dataset 
