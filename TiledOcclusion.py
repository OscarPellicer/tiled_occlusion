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
        #Convert to arrays for simplicity and perform basic checks
        k, window= np.array(k), np.array(window)
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
        iter_wrapper= partial(tqdm.tqdm, total=np.prod(k)) if show_progress else lambda x: x
        for indices in iter_wrapper(product(range(k_ext[0]), range(k_ext[1]), range(k_ext[2]), range(k_ext[3]))):
            shifted_indexer= [slice(None)] + [slice(wi//ki*index, ii + wi//ki*index) for ii, wi, ki, index in 
                                                    list(zip(input.shape[1:], window, k, indices[4-len(k):]))]
            shifted_input= padded_input.__getitem__(shifted_indexer)
            attr = super().attribute(shifted_input, strides=tuple([int(w) for w in window]), target=target,
                                     sliding_window_shapes=tuple([int(w) for w in window]), baselines=baselines, **occ_kwargs)
            #Equivalent to: attributions_occ[*shifted_indexer]+= attr
            attributions_occ.__setitem__(shifted_indexer, attributions_occ.__getitem__(shifted_indexer) + attr)
        return attributions_occ.__getitem__(padded_indexer)