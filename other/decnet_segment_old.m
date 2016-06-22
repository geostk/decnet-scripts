function [cropped_score_map, hlabel, seg, clabels] = decnet_segment_old( I, thresh, ref_size )

    if nargin < 2 || isempty(thresh)
        thresh = 0.5;
    end
    if nargin < 3
        ref_size = 320;
    end
    
    % pad image to be square
    im_sz = max(size(I,1), size(I,2));
    caffe_im = padarray(I, [im_sz - size(I,1), im_sz - size(I,2)], 'post');
    caffe_im = decnet_preprocess(caffe_im, ref_size);
    label = single(zeros([1, 1, 20]));
    cnn_output = caffe('forward', {caffe_im; label});
    
    cls_score = cnn_output{1};
    % cls_score_max = cnn_output{2};
    seg_score = cnn_output{3};
            
    softmax_score = exp(seg_score - repmat(max(seg_score, [], 3), [1,1,size(seg_score,3)]));
    softmax_score = softmax_score ./ repmat(sum(softmax_score, 3), [1,1, size(softmax_score,3)]);
    
    score_map(:,:,1) = softmax_score(:,:,1);   
    num_c = sum(cls_score > thresh)+1;
    clabels = [0; find(cls_score > thresh)]';
    % force person class to be in
    if isempty(find(clabels == 15, 1))
        clabels = [clabels 15];
        num_c = num_c + 1;
    end
    if nargout == 1
        clabels = [0, 15]; % we only care about humans (and the background)
        num_c = 2;
    end
    for j = 1:20
        idx = find(clabels == j, 1);
        if ~isempty(idx)
            label = zeros([1,1,20]);
            label(j) = cls_score(j);
            label = single(label);
            tic;
            cnn_output = caffe('forward', {caffe_im;label});
            fprintf('[%d:%f]',j,toc);
            seg_score = cnn_output{3};
            
            softmax_score = exp(seg_score - repmat(max(seg_score, [], 3), [1,1,size(seg_score,3)]));
            softmax_score = softmax_score ./ repmat(sum(softmax_score, 3), [1,1, size(softmax_score,3)]);

            score_map(:,:,idx) = softmax_score(:,:,2);
        end    
    end
    
    score_map = score_map + repmat((single(sum(score_map,3)<=0)), [1, 1, num_c]);
        
    norm_score_map = score_map;
    norm_score_map = single(norm_score_map ./ repmat(sum(score_map, 3), [1,1,size(score_map, 3)]));
    
    resize_score_map = imresize(norm_score_map, [im_sz, im_sz]);    
    
    resize_score_map = permute(resize_score_map, [2,1,3]);
    cropped_score_map = single(resize_score_map(1:size(I,1), 1:size(I,2), :));
        
    [~, seg] = max(cropped_score_map, [], 3);
    seg = uint8(seg-1);
    hlabel = find(clabels == 15, 1);
    
    if nargout == 1
        cropped_score_map = cropped_score_map(:, :, hlabel);
    end
end