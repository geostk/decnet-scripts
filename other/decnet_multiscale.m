function [seg, score, min_score] = decnet_multiscale( I, box_scale, min_score, thresh, ref_size )
    if nargin < 2 || isempty(box_scale)
        box_scale = 1.2;
    end
    if nargin < 3
        min_score = 0.1;
    end
    if nargin < 4
        thresh = 0.5;
    end
    if nargin < 5
        ref_size = 320;
    end
    
    score = decnet_segment(I, thresh, ref_size);
    if ~isempty(min_score)
        blocks = bwlabel(score >= min_score);
        num_blocks = max(blocks(:));
    else
        % adaptive min score
        max_blocks = 22; % 22 players, 2 coaches and 2 referees, most likely not all at the same time
        num_blocks = 0;
        B = [];
        for m = 2.^(-[1:0.125:10])
            cur_blcks = bwlabel(score >= m);
            cur_count = max(cur_blcks(:));
            B = [B, cur_count];
            if cur_count > num_blocks && cur_count <= max_blocks
                blocks = cur_blcks;
                num_blocks = cur_count;
                min_score = m;
            end
        end
        fprintf('Using min_score=%f\n', min_score);
    end
    
    % do segmentation for each block at its natural scale
    seg = zeros(size(score), 'like', score);
    for i = 1:num_blocks
        fprintf('Processing block %d of %d\n', i, num_blocks);
        [x0, y0, x1, y1] = bbox(blocks == i, box_scale, ref_size);
        seg_block = decnet_segment(I(y0:y1, x0:x1, :), thresh, ref_size);
        normalized = normalize(seg_block);
        seg(y0:y1, x0:x1) = max(seg(y0:y1, x0:x1), normalized);
    end
end

function [x0, y0, x1, y1] = bbox( D, scale, min_size )
    row = sum(D, 1);
    x = find(row, 1, 'first');
    w = find(row, 1, 'last') - x + 1;
    col = sum(D, 2);
    y = find(col, 1, 'first');
    h = find(col, 1, 'last') - y + 1;
    
    % rescale from box center
    cx = x + w / 2; cw = ceil(max(min_size, w * scale) * 0.5);
    x0 = max(1, floor(cx - cw));
    x1 = min(ceil(cx + cw), size(D, 2));
    cy = y + h / 2; ch = ceil(max(min_size, h * scale) * 0.5);
    y0 = max(1, floor(cy - ch));
    y1 = min(ceil(cy + ch), size(D, 1));
    
    assert(x0 > 0 && x1 <= size(D, 2) ...
        && y0 > 0 && y1 <= size(D, 1), 'Out of bounds');
end

function n = normalize(block)
    M = max(block(:));
    if ~M
        M = 1;
    end
    n = block * (1/M);
end