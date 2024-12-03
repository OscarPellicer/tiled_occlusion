import torch
from captum.log import log_usage
from captum.attr import Occlusion
import numpy as np

from itertools import product
from functools import partial
import tqdm

class TiledOcclusion(Occlusion):
    ''' 
        Custom occlusion-based attribution method. Accepts 1D to 4D inputs (+ batch dimension)
        See: https://github.com/OscarPellicer/tiled_occlusion
    '''
    @log_usage()
    def attribute(self, input, target=0, k=(1,2,2), window=(3,10,10), baselines=0, show_progress=True, **occ_kwargs):
        '''
        Parameters
        ----------
        input: Batch of inputs, which is subject to explanations
        target: Batch of labels, which is subject to explanations
        k: upscaling factor for all dimensions except for batch dimension (set to 1 for a dimension to disable)
        window: occlusion window size. `window` values must be divisible by `target` values
        baselines: occlusion baselines, e.g. 0
        show_progress: shows a progress bar
        **occ_kwargs: additional parameters to pass to Captum's Occlusion class
        
        Returns
        -------
        Attributions
        '''
        #Convert to arrays for simplicity and perform basic checks
        k, window= np.array(k), np.array(window)
        assert not isinstance(input, tuple), 'At the moment, having tuples of tensors as input is not supported'
        assert len(window) <= 4, 'Window can at maximum be a 4-value tuple, e.g. (c, t, x, y) or (c, x, y, z)'
        assert len(input.shape) == len(window) + 1, f'{len(input.shape)=} != {(1 + len(window))=} (i.e. batch size + len(window))'
        assert window.shape == k.shape, f'{window.shape=} != {k.hape=}'
        assert not np.any(window % k),\
            f'All elements in {window=} must be divisible by the elements in {k=}'

        #Create expanded tensors
        occ_shapes= [input.shape[0]] + [ii + (wi//ki if ki > 1 else 0) for ii, wi, ki in zip(input.shape[1:], window, k)]
        padded_indexer= [slice(None)] + [slice(wi//ki if ki > 1 else 0, None) for wi, ki in zip(window, k)]

        attributions_occ= torch.zeros(*occ_shapes).to(input.device)
        padded_input= torch.clone(attributions_occ)
        #Equivalent to: padded_input[*padded_indexer]= input, but we are not allowed to use * when indexing
        padded_input.__setitem__(padded_indexer, input)

        #Iterate over all dimensions. We will use product() which creates nested loops
        k_ext= [1] * (4 - len(k)) + list(k) #Extend k
        iterator= tqdm.tqdm(product(range(k_ext[0]), range(k_ext[1]), range(k_ext[2]), range(k_ext[3])), 
                            desc="TiledOcclusion", disable=not show_progress, total=np.prod(k))
        for indices in iterator:
            shifted_indexer= [slice(None)] + [slice(wi//ki*index, ii + wi//ki*index) for ii, wi, ki, index in 
                                                    list(zip(input.shape[1:], window, k, indices[4-len(k):]))]
            shifted_input= padded_input.__getitem__(shifted_indexer)
            attr = super().attribute(shifted_input, strides=tuple([int(w) for w in window]), target=target,
                                     sliding_window_shapes=tuple([int(w) for w in window]), baselines=baselines, **occ_kwargs)
            #Equivalent to: attributions_occ[*shifted_indexer]+= attr
            attributions_occ.__setitem__(shifted_indexer, attributions_occ.__getitem__(shifted_indexer) + attr)
        return attributions_occ.__getitem__(padded_indexer)
    
class FusionGrad:
    def __init__(self, attribution_method, model=None):
        ''' Implementation based on:
            https://github.com/understandable-machine-intelligence-lab/NoiseGrad 
        '''
        self.attribution_method= attribution_method
        if model is None:
            self.model= self.attribution_method.forward_func
            assert isinstance(self.model, torch.nn.Module),\
                'For FusionGrad to work at the moment, either the model has to be passed, '\
                'or the callable function must be a torch Module'
        else:
            self.model= model

    @log_usage()     
    def attribute(self, input, *args, target=0,
                  mean: float=0., std: float=0.02, sg_mean: float=0., sg_std: float=0.1, 
                  n:int=10, m: int=10, additive_noise: bool=True, sg_additive_noise: bool=False,
                  show_progress: bool=True, **kwargs):
        '''
        Parameters
        ----------
        input: Batch of inputs, which is subject to explanations
        target: Batch of labels, which is subject to explanations
        mean: Mean of normal distribution, from which noise added to weights is sampled
        std: Standard deviation normal distribution, from which noise added to weights is sampled
        sg_mean: Mean of normal distribution, from which noise added to inputs is sampled
        sg_std: Standard deviation normal distribution, from which noise added to inputs is sampled
        n: Number of times noise for weights is sampled
        m: Number of times noise for inputs is sampled
        additive_noise: Noise type, either use additive (if True) or multiplicative (if False). Original implementation uses True
        sg_additive_noise: Noise type for input signal, either use additive (if True) or multiplicative (if False).
        show_progress: Indicates whether progress bar should be displayed, default=True

        Returns
        -------
        Attributions
        '''
        assert not isinstance(input, tuple), 'At the moment, having tuples of tensors as input is not supported'
        with torch.no_grad():
            explanation = torch.zeros(n, m, *input.shape)
            original_weights = self.model.state_dict().copy()
        
            if std > 0.: 
                self._distribution = torch.distributions.normal.Normal(loc=mean, scale=std)
            it = tqdm.auto.tqdm(range(n*m), desc="FusionGrad", disable=not show_progress)
            with it as pbar:
                for i in range(n):
                    self.model.load_state_dict(original_weights)
                    if std > 0.: 
                        for layer in self.model.parameters():
                            if additive_noise:
                                layer.add_(self._distribution.sample(layer.size()).to(layer.device))
                            else:
                                layer.mul_(self._distribution.sample(layer.size()).to(layer.device))
                    for j in range(m):
                        noise = torch.randn_like(input) * sg_std + sg_mean
                        if sg_additive_noise: 
                            inputs_noisy = input + noise
                        else: 
                            inputs_noisy = input * noise
                        explanation[i][j] = self.attribution_method.attribute(inputs_noisy, 
                                                                              *args, target=target, **kwargs)
                        pbar.update()

            self.model.load_state_dict(original_weights)
            return explanation.mean(axis=(0,1))# Move the entire content of ExtraAttrib.py here 