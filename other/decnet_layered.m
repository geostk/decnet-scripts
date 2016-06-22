function [score, layers] = decnet_layered( I, boxes, scaleup, ref_size )
    if nargin < 3
        scaleup = 1.3;
    end
    if nargin < 4
        ref_size = 320;
    end
    
    [H, W, ~] = size(I); % original size
    score = zeros([H, W]); % initial full-scale score
    L = numel(boxes);
    layers = cell(L, 1);
    for l = 1:L
        
        % workspace
        lscore = zeros([H, W], 'like', score);
        B = size(boxes{l}, 1);
        fprintf('Processing level %d (%d boxes)\n', l, B);
        
        for b = 1:B
            box = boxes{l}(b, :);
            [ry, rx] = get_block(H, W, box, scaleup);
            block = decnet_segment(I(ry, rx, :), [], ref_size);
            lscore(ry, rx) = max(lscore(ry, rx), block);
        end
        
        score = max(score, lscore);
        layers{l} = lscore;
        fprintf('\n');
    end
end

function [ry, rx] = get_block(H, W, bbox, scaleup)
    w  = bbox(3) - bbox(1);
    h  = bbox(4) - bbox(2);
    cx = (bbox(1) + bbox(3)) * 0.5;
    cy = (bbox(2) + bbox(4)) * 0.5;
    s  = max(w, h) * scaleup; % box side
    d  = ceil(s * 0.5); % delta = half side
    % new box
    ry = max(1, floor(cy - d)):min(H, ceil(cy + d));
    rx = max(1, floor(cx - d)):min(W, ceil(cx + d));
end