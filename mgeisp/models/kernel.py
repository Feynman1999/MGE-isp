import megengine as mge
from .registry import BACKBONES
import megengine.module as M
import megengine.functional as F


@BACKBONES.register_module
class pixelwisekernel(M.Module):
    def __init__(self, kernel_size):
        super(pixelwisekernel, self).__init__()
        self.kernel_size = kernel_size
        

    def forward(self, input_pts, input_views):
        
        # sigma
        h = input_pts
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h)

        sigma, geo_feat = h[..., 0:1], h[..., 1:]
        
        # color
        h = F.concat([input_views, geo_feat], axis=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h)
            
        color = h
        outputs = F.concat([color, sigma], axis = -1)

        return outputs
