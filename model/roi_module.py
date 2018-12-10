import torch
from torch.autograd import Function
import cupy
# factory function for creating tuple subclasses with named fields
from collections import namedtuple
from string import Template
from model.utils.roi_cupy import kernel_backward, kernel_forward

Stream = namedtuple('Stream', ['ptr'])

@cupy.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    cupy.cuda.runtime.free(0)
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)


class RoIPooling2D(torch.nn.Module):
    def __init__(self, height, width, spatial_scale):
        super(RoIPooling2D, self).__init__()
        self.RoI = RoI(height, width, spatial_scale)

    def forward(self, x, rois):
        return self.RoI(x, rois)


class RoI(Function):
    def __init__(self, h, w, spatial_scale):
        self.forward_fn = load_kernel('roi_forward', kernel_forward)
        self.backward_fn = load_kernel('roi_backward', kernel_backward)
        self.h, self.w, self.spatial_scale = h, w, spatial_scale

    def forward(self, x, rois):
        x = x.contiguous()
        rois = rois.contiguous()

        self.in_size = B, C, H, W = x.size()
        N = rois.size(0)
        self.N = N
        output = torch.zeros(N, C, self.h, self.w).cuda()
        self.argmax_data = torch.zeros(N, C, self.h, self.w).int().cuda()
        self.rois = rois

        args = [x.data_ptr(), rois.data_ptr(), output.data_ptr(),
                self.argmax_data.data_ptr(), self.spatial_scale, C, H, W,
                self.h, self.w, output.numel()]
        stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
        self.forward_fn(args=args, block=(1024, 1, 1), grid=((output.numel()+1024-1)//1024, 1, 1), stream=stream)
        
        return output

    def backward(self, grad_output):
        grad_output = grad_output.contiguous()
        B, C, H, W = self.in_size
        grad_input = torch.zeros(self.in_size).cuda()
        stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
        args = [grad_output.data_ptr(), self.argmax_data.data_ptr(),
                self.rois.data_ptr(), grad_input.data_ptr(), self.N, self.spatial_scale, 
                C, H, W, self.h, self.w, grad_input.numel()]
        self.backward_fn(args=args, block=(1024, 1, 1), grid=((grad_input.numel()+1024-1)//1024, 1, 1), stream=stream)
        
        return grad_input, None