# FlowerDockingSensorimotorLoopRL
This repo is an RL set up to train bat to dock on flower assuming flower is showing up on the echoic input. Motor command will be continuous action instead of discrete

## Prerequisite
Dataset was collected by ensonifying a flower *Cobaea Scandens*. The flower was a 3X scaled up 3D printed replicated of a real flower. Data collected was done on a realistic robotic bathead with single emission band at 42kHz. Due to the 3X scale up, we aimed to simulate the ~120 kHz of max return observed on real flower.
The cleaned up dataset used for this simulation work can be downloaded via this link:
[Download Dataset](https://mailuc-my.sharepoint.com/:u:/g/personal/nguye2t7_mail_uc_edu/ET_rE384-uJCvDcUE0RKjbIB-u56eXxl2skpdBiNgnUYzw?e=K9kNab)

After downloading the file, please extract and place the `flower3x_snippet` folder into the `Dataset` folder for the code to work.

Use the convenient script `install_dataset.sh` to set up the dataset if necessary.

Download the model using the link:
[Download Model](https://mailuc-my.sharepoint.com/:u:/g/personal/nguye2t7_mail_uc_edu/Efglh8nBqSxHut-I05v9i14BmP93h0MoRDwuzxf-iTqCKw?e=nZEkuw)
Save the model into the `saved_models` folder as below. Make sure file name is accurate.
```plaintext
FlowerDockingSensorimotorLoopRL
├── saved_models
|   └── flower_pose_estimator_v1.pt
├── Dataset
|   ├── flower3x_snippet
|   ├── emission_snippet.npz
|   └── noise.npz
...
```

## Demo
1. Visualize the waveforms bats experienced with a single band freq. 1 bat 1 flower in the scene.
- run `python apps/VisualizerFlowerEcho.py`
- run `python apps/VisualizeFlowerEchoBeautiful.py` for demo with graphical illustration of bat and flower.
2. Demo of Flower Home-in Loop app:
- run `python apps/FlowerHomeInLoopDemo.py`
- run `python apps/FlowerHomeInLoopDemoBeautiful.py` for demo with graphical illustration of bat and flower.
