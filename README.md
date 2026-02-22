### Codebase for: _"Reducing information dependency does not cause training data privacy. Adversarially non-robust features do."_ (ICLR 2026)

Our code is:
- Streamlined: Focus on what's important and skip boilerplate;
- Comprehensive;
- Integrated with Weights and Biases (W&B), Hydra, etc.
- Scalable and maintainable.

### Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [5-Minute Quickstart](#quickstart)
- [More Details](#details)

### Features <a name="features"></a>
The code base supports a wide range of model inversion defenses, attacks, target classifiers, and datasets for quick evaluation of SOTA methods:

#### Defenses
| Name | Citation | Implementation | Command (defense=) | 
|----------|----------|---------|---------|
| MID        | [Wang et al. 2020](https://arxiv.org/abs/2009.05241) | [Github](https://github.com/Jiachen-T-Wang/mi-defense) | mid
| BiDO       | [Peng et al. 2022](https://arxiv.org/abs/2206.05483)   | [Github](https://github.com/AlanPeng0897/Defend_MI) | bido
| TL-DMI     | [Ho et al. 2024](https://arxiv.org/abs/2405.05588) | [Github](https://github.com/hosytuyen/TL-DMI) | tldmi
| RoLSS      | [Koh et al. 2024](https://link.springer.com/chapter/10.1007/978-3-031-73004-7_9) | [Github](https://github.com/Pillowkoh/RoLSS/) | rolss
| Neg-LS     | [Struppek et al. 2024](https://arxiv.org/abs/2310.06549)   | [Github](https://github.com/LukasStruppek/Plug-and-Play-Attacks) | label_smoothing
| Trap-MID     | [Liu et al. 2024](https://arxiv.org/html/2411.08460v1)   | [Github](https://github.com/ntuaislab/Trap-MID) | trap_mid
| **AT-AT (Ours)**   | This paper  | This repo | **atat** 

#### Attacks
| Name | Resolution | Paper | Implementation | Command (attack=) | 
|----------|----------|---------|---------| ---------|
| PPA          | High-Res | [Struppek et al. 2022](https://proceedings.mlr.press/v162/struppek22a.html) | [Github](https://github.com/LukasStruppek/Plug-and-Play-Attacks) | plug_and_play
| IF-GMI   | High-Res | [Qiu et al. 2024](https://arxiv.org/abs/2407.13863) | [Github](https://github.com/final-solution/IF-GMI) | if_gmi
| PPDG    | High-Res | [Peng et al. 2024](https://openreview.net/forum?id=pyqPUf36D2) | [Github](https://github.com/tmlr-group/PPDG-MI) | ppdg

We thank the authors of the above papers for making their code publicly available.

#### Classifiers

We support a range of pretrained classifiers from torchvision, including the ResNet, DenseNet, ResNeXt, and ResNeSt families, as well as Inceptionv3. We also provide a simple MLP and CNN implementation for low-resolution, small datasets.

A custom classifier can also be tested by implementing `CustomClassifier` in `custom_classifier.py` and running `model=custom` in the command line. Parameters for this can be added easily in `configuration/classifier/custom.yaml`. 

#### Datasets
The datasets are implemented with automatic downloading and processing for ease of use. 

| Name | Resolution | Downloading |
|----------|---------|---------|
| Facescrub       |  High-Res  | Automatic  |
| Stanford Dogs   |  High-Res  | Automatic  |
| CelebA **         |  High-Res  | Automatic |

** HD-CelebA (via [HD-CelebA-Cropper](https://github.com/LynnHo/HD-CelebA-Cropper)) is auto-downloaded to `data/celeba/hdcrop`. For standard CelebA, a [torchvision bug](https://github.com/pytorch/vision/issues/8204#issuecomment-1935737815) requires manual download: get `img_align_celeba.zip` from [Google Drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg), place in `data/celeba/`, and unzip. 

#### Attack Metrics
| Name                      | Description |
|--------------------------|-------------|
| AttackAcc@1                      | Accuracy of an Inception evaluation model in correctly classifying faces produced by the attack |
| AttackAcc@5                     | Same as above, with top-5 accuracy |
| Average $\delta_{face}$        | Average L2 distance to nearest training data image in the target class, averaged over all target classes, in the FaceNet evaluation model feature space |
| Average $\delta_{eval}$        | Same as above, but instead in Inception evaluation model feature space (used for Stanford dogs dataset where vggface embeddings are meaningless)|

### Prerequisites <a name="prerequisites"></a>
- Python 3.11
- CUDA 12.1
- GPU with ≥12GB VRAM (recommended)

### 5-Minute Quickstart <a name="quickstart"></a>

Get started running a wide range of defenses and attacks in minutes, following the steps below.

#### Steps
1. Clone the repository.
2. Setup and activate your Python environment. The repository is developed for use with Python 3.11. For example, using conda:
```
conda create -n ENVIRONMENT_NAME python=3.11
```
3. Install dependencies.
```
pip install -r requirements.txt
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```
4. Run bash script for setting up other repositories and downloading pretrained GAN's.
```
bash setup_files.bash
```
5. Weights and Biases Logging.
   - Setup a [Weights and Biases](https://wandb.ai/site/) account
   - Create a project
   - Create a [wandb API key](https://docs.wandb.ai/quickstart/), make a file called `secret.txt` in the main scope of the repository, and paste the API key there.
   - Set the entity and project names in the configuration (`configuration/classifier/training/default.yaml` and `configuration/model_inversion/training/default.yaml` respectively)
   - Run scripts with `training.wandb.track=True` in the command line, or set it as default in the configuration.
   - Enjoy smart and scalable cloud logging for training classifiers and model inversion. 

#### Training Classifiers
Train a model with any of our models and datasets easily, such as with the following command:
```
python train_classifier.py dataset=FaceScrub model=pretrained model.architecture="ResNet152" model.hyper.epochs=50 training.wandb.track=True
```

#### Defending Classifiers
To defend classifiers, simply add the defense name in the command for training classifiers, for example:

```
python train_classifier.py defense=tldmi defense.freeze_layers=6 dataset=FaceScrub model=pretrained model.architecture="ResNet18" training.wandb.track=True
```
All the available configurable parameters and default values for training and defending classifiers can be found in the `.yaml` files located in `configuration/classifier/`.

#### Attacking Classifiers
Attacking classifiers can be configured similarly:
```
python run_model_inversion.py attack=plug_and_play target_wandb_id="TARGET_RUN_ID" attack.evaluation_model.wandb_id="EVAL_RUN_ID" training.wandb.track=True
```
Where `"TARGET_RUN_ID"` can be found in your wandb project. All the available configurable parameters and default values for attacking trained classifiers can be found in the `.yaml` files located in `configuration/model_inversion/`.

### More Details <a name="details"></a>
The repository utilizes [Hydra](https://hydra.cc/docs/intro/) for dynamic hierachical configuration. The default configuration values can be changed in the configuration folder or via the Hydra CLI syntax. The following are part of our Hydra configuration for training classifiers:
* models (e.g. `model=cnn` or `model=pretrained model.architecture=resnet18 model.hyper.lr=0.0001`)
* target model training (e.g. `training.wandb.track=True training.device=cuda`)
* defenses (e.g. `defense=bido defense.a1=0.1 defense.a2=0.05` or `defense=tldmi defense.freeze_layers=6`)

#### Defense Application via Overwriting
Defenses are implemented as functions that take in an "undefended" AbstractClassifier and output a new "defended" model that inherits from the undefended model. The implementation of this function then amounts to simply overwriting the specific methods of the model that are affected by the defense, isolating the essential features of the defense. The defense is then added to our hierarchical configuration structure using Hydra, so that training a model with defense X simply amounts to adding `defense=X` to the command line.

For more detailed instructions on adding your own custom defense, see [Adding your own defenses](defenses/) in the `defenses` folder.

### Citation
If you find this code helpful in your research, please consider citing.
```bibtex
@inproceedings{torp2026iclr,
  title={Reducing information dependency does not cause training data privacy. Adversarially non-robust features do.},
  author={Rasmus Torp, Shailen Smith, Adam Breuer},
  booktitle={ICLR},
  year={2026}
}
```
