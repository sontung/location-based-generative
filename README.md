# Self-Supervised Learning of Scene-Graph Representations for Robotic Sequential Manipulation Planning
# [[video]](https://youtu.be/JZ4FepUo6TY) [[paper]](https://drive.google.com/file/d/1Pe_5WDbMg9UsaPR5GWXJKut7E1c5Vgw-/view) [[talk]](https://youtu.be/pHGppsmcnx4)

## How to clone
````
git clone https://github.com/sontung/location-based-generative.git
git submodule init
git submodule update
````

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

## Code files
- `data_loader.py` - pytorch compatible data generator
- `large_scale*` - large scale analysis in the paper
- `fine_scale.py` - fine scale analysis in the paper  
- `models.py` - arch of the visual module
- `train_uc.py` and `train_uc_sim.py` - train the visual module on the PBW and sim datasets
- `train_uc_trivial_obj.py` - train the visual module with trivial masks from COCO (see `data/trivial_masks` for such examples)
- `train_shape_predictor.py` - train a network to predict the shape from a mask
- `analyze_random_loc.py` - generalization to arbitrary novel stack locations
- `utils.py` - utility code
- `planner*` - planning with PBW and sim datasets  
- `plan_with_3d_rel.py` - planning with depth sensors
- `plan_with_heuristic.py` - planning with heuristic on *what is stable and what is unstable*

## Code citation
- Python implementation of the Connected Component Labelling algorithm [[1](https://github.com/jacklj/ccl)]
- Pytorch [[2](https://pytorch.org/)]
- Photorealistic Blocksworld generator (slightly modified to our need) [[our version]](https://github.com/sontung/photorealistic-blocksworld) [[original version]](https://github.com/IBM/photorealistic-blocksworld)
