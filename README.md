# SSL of scene graph representation

## How to train
Run `train_uc.py` to train the visual module on photo-realistic Blocksworld dataset 
and `train_uc_sim.py` to train on simulation dataset. See 
`commands.txt` for usage examples.

## Pretrained model and dataset
Download from [here.](https://drive.google.com/drive/folders/1AoN8AmgMhqEvgHEJZ83iG25N-L4TJa1p?usp=sharing)

## Inference
Run `planner_rs.py` to plan for random pair of images from the simulation dataset.
 Intermediate
images are saved under `figures/` folder. The same thing applies
when run `planner.py` for the photo-realistic blocksworld. 