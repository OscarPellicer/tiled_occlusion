# extra-attributions

This repo currently includes two attribution methods:

- Tiled Occlusion is a simple attribution method built upon standard Occlusion and implemented using Captum's interface.
- FusionGrad is an implementation of [FusionGrad](https://github.com/understandable-machine-intelligence-lab/NoiseGrad), that more closely follows Captum's interface.

## Installation

You can install directly using pip:

```{bash}
pip install git+https://github.com/OscarPellicer/extra-attributions.git
```

Or for development, clone the repository and install in editable mode:

```{bash}
git clone https://github.com/OscarPellicer/extra-attributions.git
cd extra-attributions
pip install -e .
```

## Usage

You can use the attribution methods as any other Captum attribution method. E.g.:

```{python}
from extra_attributions import TiledOcclusion, FusionGrad
from captum.attr import IntegratedGradients

# TiledOcclusion
tiled_occlusion= TiledOcclusion(model)
attributions_tocc= tiled_occlusion.attribute(input, target=target, k=[1,2,2], window= [3,18,18])

# FusionGrad
integrated_gradients = IntegratedGradients(model)
fusiongrad= FusionGrad(integrated_gradients, model=model)
attributions_ig_fg= fusiongrad.attribute(input, target=target,
                            std=0.05, mean=1., n=5, additive_noise=False, #Weight noise (mult)
                            sg_std=1.5, m=5, sg_additive_noise=True, #Input noise (add)
                                         )
```

## TiledOcclusion

For a full woring example, refer to the `Tutorial.ipynb`

Some notes about `TiledOcclusion`:

- If we set `k = [1,1,1]`, it is the same as using standard `Occlusion`
- By using higher values of `k`, the resolution of the attribution gets increased by that factor `k`
- `TiledOcclusion` supports from 1D to 4D tensors (without counting the batch dimension)
- It has been designed to share the interface with Captum, as such it is possible to use Captum's `NoiseTunnel(TiledOcclusion(model))` on top
- The computational costs are exactly the same as for the standard `Occlusion` for a given output resolution

Here we can see some examples of attributions. Notice that when `k = [1,1,1]` `TiledOcclusion` == `Occlusion`; also, notice that in some images class `cock` is being predicted, while on others it is class `hen`, as indicated in the title:

Using Captum's Occlusion (i.e. using TiledOcclusion with `k = [1,1,1]`):
![Using Occlusion](https://github.com/OscarPellicer/tiled_occlusion/blob/main/media/occlusion_1.png)
![Using Occlusion](https://github.com/OscarPellicer/tiled_occlusion/blob/main/media/occlusion_2.png)
![Using Occlusion](https://github.com/OscarPellicer/tiled_occlusion/blob/main/media/occlusion_3b.png)
![Using Occlusion](https://github.com/OscarPellicer/tiled_occlusion/blob/main/media/occlusion_4b.png)

And using TiledOcclusion:
![Using TiledOcclusion](https://github.com/OscarPellicer/tiled_occlusion/blob/main/media/tiled_occlusion_1.png)
![Using TiledOcclusion](https://github.com/OscarPellicer/tiled_occlusion/blob/main/media/tiled_occlusion_2.png)
![Using TiledOcclusion](https://github.com/OscarPellicer/tiled_occlusion/blob/main/media/tiled_occlusion_3.png)
![Using TiledOcclusion](https://github.com/OscarPellicer/tiled_occlusion/blob/main/media/tiled_occlusion_4b.png)

As can be seen, `TiledOcclusion` generally produces smoother results than `Occlusion` (specially at higher resolutions) which also are much better aligned with our intution for how attributions should behave, i.e.: they better identify the parts of the image that seem to help the model differentiate the `hen` from the `cock`.

### How does it work

The idea of the method is to combine the power of bigger occlusion patches while obtaining a high resolution smoother occlusion map, by adding occlusion results from several slightlhy shifted versions of the same input image. If you are using this repository and need more info on the method, open an issue and I will try to improve this description.

## FusionGrad

See the `Attribution_tests.ipynb` for an example of how to use `FusionGrad`, and the `README.md` in the [NoiseGrad repo](https://github.com/understandable-machine-intelligence-lab/NoiseGrad) for more details on the method.

## Citing

If you found this useful, simply cite this Github!
