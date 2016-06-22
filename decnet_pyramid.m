function [score, mscore, pyr_scores, pyr_masks] = decnet_pyramid( I, levels, overlap, ref_size )
    if nargin < 2 || levels <= 0
        levels = inf;
    end
    if nargin < 3
        overlap = 0;
    end
    assert(overlap >= 0 && overlap < 1, 'Invalid overlap');
    if nargin < 4
        ref_size = 320;
    end
    % delta
    delta = ref_size * (1 - overlap);
    
    % initial full-scale score
    score = decnet_segment(I, [], [], ref_size);
    score = max(score, 0);
    score0 = score;
    if nargout > 1
        mscore = log(score);
    end
    
    [H, W, ~] = size(I); % original size
    level = 1;
    while maxDim(I) > ref_size && level < levels
        
        % workspace
        [h, w, ~] = size(I);
        subscore = zeros([h, w], 'like', score);
        submask  = zeros([h, w], 'like', score);
        fprintf('Processing level %d (%d x %d)\n', level, h, w);
        
        % compute score on each subblock
        num_blocks = numel(1:delta:w-1) * numel(1:delta:h-1);
        block_id = 1;
        for il = 1:delta:w-1
            ir = min(il + ref_size - 1, w);
            for jl = 1:delta:h-1
                jr = min(jl + ref_size - 1, h);
                
                fprintf('- block %d of %d\n', block_id, num_blocks);
                
                block = decnet_segment(I(jl:jr, il:ir, :), [], [], ref_size);
                %imwrite(I(jl:jr, il:ir, :), sprintf('block-l%d-i%d-j%d.png', level, il, jl));
                %imwrite(block, sprintf('block-l%d-i%d-j%d-gray.png', level, il, jl));
                %imwrite(labelize(block), jet(256), sprintf('block-l%d-i%d-j%d-cmap.png', level, il, jl));
                subscore(jl:jr, il:ir) = subscore(jl:jr, il:ir) + block;
                submask(jl:jr,  il:ir) = submask(jl:jr,  il:ir) + 1;
                
                block_id = block_id + 1;
            end
        end
        % should be positive
        subscore = max(subscore, 0);
        
        % optionally store level data
        if nargout > 2
            pyr_scores{level} = subscore;
            pyr_masks{level}  = submask;
        end
        
        % aggregate level data
        score = score + imresize(subscore ./ submask, [H, W]);
        if nargout > 1
            mscore = mscore + log(imresize(subscore ./ submask, [H, W]));
        end
        
        % go to level above
        I = imresize(I, 0.5);
        level = level + 1;
    end
    % initial full scale
    if nargout > 2
        pyr_scores{level} = score0;
        pyr_masks{level}  = ones(H, W, 'like', score);
    end
    % multiplicative score
    if nargout > 1
        mscore = exp(mscore);
    end
    % normalize by number of levels
    score = score * (1/level);
end

function m = maxDim(I)
    m = max(size(I, 1), size(I, 2));
end
