function [] = decnet_init( proto, model, use_gpu )
    base = 'Data/DecoupledNet/Data';
    if nargin < 1 || isempty(proto)
        proto = fullfile(base, 'DecoupledNet_Full_anno_inference_deploy.prototxt');
    end
    if nargin < 2 || isempty(model)
        model = fullfile(base, 'DecoupledNet_Full_anno_inference.caffemodel');
    end
    if nargin < 3
        use_gpu = 1;
    end
    
    addpath('caffe/matlab/caffe');
    
    if caffe('is_initialized')
        caffe('reset');
    end
    caffe('init', proto, model)
    if use_gpu
        caffe('set_mode_gpu');
    else
        caffe('set_mode_cpu');
    end
    caffe('set_phase_test')
end