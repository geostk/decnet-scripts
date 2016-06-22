function [cropped_score_map, hlabel, seg, clabels] = decnet_segment( I, marginal, thresh, ref_size )

    % no argument => output marginal modes
    if nargin == 0
        cropped_score_map = {'activation', 'background', 'base', 'components', 'filter', 'norm', 'prob'};
        return
    end
    if nargin < 2 || isempty(marginal)
        marginal = 'activation';
    end
    if nargin < 3 || isempty(thresh)
        thresh = 0.5;
    end
    if nargin < 4
        ref_size = 320;
    end
    
    [h, w, ~] = size(I);
    
    % pad image to be square
    im_sz = max(size(I,1), size(I,2));
    caffe_im = padarray(I, [im_sz - h, im_sz - w], 'post');
    caffe_im = decnet_preprocess(caffe_im, ref_size);
    
    % pre-computation depending on the marginalization method
    label = single(zeros([1, 1, 20]));
    switch marginal
        case 'prob'
            % we want all the classes that have high enough activation
            % => need a dry run first
            cnn_output = caffe('forward', {caffe_im; label});
            cls_score = cnn_output{1};
            
            % we compute background using the high-enough classes
            label = single(cls_score .* (cls_score > thresh));
            
        case 'base'
            % activation without any class indicator
            
        otherwise
            % we assume interest in human class
            label(15) = 1;
    end
    
    % compute background and classification
    cnn_output = caffe('forward', {caffe_im; label});
    
    cls_score = cnn_output{1};
    % cls_score_max = cnn_output{2};
    seg_score = cnn_output{3};
    
    % marginalization
    softmax_score = exp(seg_score - repmat(max(seg_score, [], 3), [1,1,size(seg_score,3)]));
    softmax_score = softmax_score ./ repmat(sum(softmax_score, 3), [1,1, size(softmax_score,3)]);
    
    % single activation case
    if ~strcmp(marginal, 'prob')
        
        background = softmax_score(:, :, 1);
        human = softmax_score(:, :, 2);
        hlabel = 1;
        seg = human;
        clabels = double(~strcmp(marginal, 'base')) * 15;

        % normalize score (from DecoupledNet's implementation)
        switch marginal
            case 'norm'
                score = softmax_score;
                mask = single(max(score, [], 3) > 0);
                score = human ./ sum(score, 3) .* mask;

            case 'filter'
                score = human .* (human > background);

            case 'components'
                score = cat(3, human, background, human .* (human > background));

            case 'background'
                score = background;

            otherwise
                score = human;
        end

        % resize and post-process format
        resize_score_map = imresize(score, [im_sz, im_sz]);    
        resize_score_map = permute(resize_score_map, [2,1,3]);
        cropped_score_map = single(resize_score_map(1:h, 1:w, :));
        
    else

        score_map(:,:,1) = softmax_score(:,:,2); % background score

        % marginalization using threshold for activations
        % i.e. filter classes that are strong enough (and bg)
        num_c = sum(cls_score > thresh)+1;
        clabels = [0; find(cls_score > thresh)]';
        % force person class to be in
        if isempty(find(clabels == 15, 1))
            clabels = [clabels 15];
            num_c = num_c + 1;
        end
        % compute separate activations for the classes of interest
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
        fprintf('\n');
        
        resize_score_map = imresize(score_map, [im_sz, im_sz]);    

        resize_score_map = permute(resize_score_map, [2,1,3]);
        cropped_score_map = single(resize_score_map(1:h, 1:w, :));

        % find each pixel class
        [~, seg] = max(cropped_score_map, [], 3);
        seg = uint8(seg-1);
        hlabel = find(clabels == 15, 1);

        if nargout == 1
            cropped_score_map = exp(cropped_score_map - repmat(max(cropped_score_map, [], 3), [1,1,size(cropped_score_map,3)]));
            cropped_score_map = cropped_score_map ./ repmat(sum(cropped_score_map, 3), [1,1, size(cropped_score_map,3)]);
            cropped_score_map = cropped_score_map(:, :, hlabel); % probability distribution given all classes + background
        end
    end
end

function preprocessed_img = decnet_preprocess(img, img_sz)

    meanImg = [104.00698793, 116.66876762, 122.67891434]; % order = bgr
    meanImg = repmat(meanImg, [img_sz^2,1]);
    meanImg = reshape(meanImg, [img_sz, img_sz, 3]); 

    resized = imresize(double(img), [img_sz img_sz], 'bilinear'); % resize cropped image
    resized = resized(:,:,[3 2 1]) - meanImg; % convert color channer rgb->bgr and subtract mean 
    preprocessed_img = single(permute(resized, [2 1 3])); 

end