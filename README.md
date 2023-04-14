# TiledOcclusion (+ FusionGrad)

Tiled Occlusion is a simple attribution method based on standard Occlusion and implemented using Captum's interface.

Additionally, this repository also includes a better implementation of [FusionGrad](https://github.com/understandable-machine-intelligence-lab/NoiseGrad), that more closely follows Captum's interface.

## Installation 
To use, make sure to have the following dependencies installed:

```{bash}
pip install torch torchvision captum "matplotlib<3.7"
```

Then simply clone the repo into your project's path:

```{bash}
git clone https://github.com/OscarPellicer/tiled_occlusion.git
```

## Examples

And then you can use it as any other Captum attribution method. E.g.:
```{python}
from ExtraAttrib import TiledOcclusion
tiled_occlusion= TiledOcclusion(model)
attributions_tocc= tiled_occlusion.attribute(image, target=target, k=[1,2,2], window= [3,18,18])
```

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

## How does it work

The idea of the method is to combine the power of bigger occlusion patches while obtaining a high resolution smoother occlusion map, by adding occlusion results from several slightlhy shifted versions of the same input image. If you are using this repository and need more info on the method, open an issue and I will try to improve this description.

## Citing

If you found this useful, simply cite this Github!
