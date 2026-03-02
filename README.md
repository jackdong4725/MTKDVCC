# MT-FKD Project

This repository implements the Meta-Teaching Framework for Adaptive Knowledge Distillation (MT-FKD) targeting video crowd/object counting.

## Project Structure
- `main.py` is the training entry point.
- Configurations are centralized in `config.py`.
- Datasets under `datasets/` including FDST, MALL and Venice (used in our evolutionary augmentation experiments).
- Expert models located in `models/experts`.

## Key Updates for the Paper
1. **Evolutionary Data Augmentation**: The augmentation module has been refactored as `EvolutionaryAugmentation` (alias `VideoMixAugmentation`). It operates on FDST, MALL and Venice datasets.
2. **Student Model**: `PointDGMamba` is adopted as the student network and is trained via distillation.
3. **Experts**: Only CountVid, GraspMamba (zero‑shot), CrowdMPM and OMAN are retained. The previous `RefAtomNet` (refAVA2) component has been fully removed from the codebase.
4. **CountVid Dependencies**: GroundingDINO model is required with weights at `weights/groundingdino_swint_ogc.pth` (symlinked from `countgd_box.pth`). Local GroundingDINO source code is accessible via `CountVid/groundingdino` or the symlink `GroundingDINO-main`. The BERT model for GroundingDINO should be placed under `bert-base-uncased` at the project root.
5. **Dataset Configuration**: Only the three datasets used in our experiments are enabled by default.
6. **Tests**: Simple scripts in `tests/` verify expert loading and weight availability; they have been updated accordingly.

## Running
Use `python main.py` to start training, ensuring you have required dependencies (`torch`, `cv2`, etc.) installed. The config validation will print warnings if essential files are missing.

## Notes
All references to the deprecated `refAVA2`/RefAtomNet have been eliminated. The codebase is now aligned with the paper's experimental design.