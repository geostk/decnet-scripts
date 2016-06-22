function [TPR, FPR, P, N] = pixel_roc( I, M, thresholds )
    [H, W, ch] = size(M);
    if ch > 1, M = rgb2gray(M); end
    [h, w, ch] = size(I);
    if ch > 1, I = rgb2gray(I); end
    if H * W ~= h * w
        I = imresize(I, [H, W]); % resize to same size
    end
%     D = numel(thresholds);
%     truePos = zeros(1, D);
%     falsePos = zeros(1, D);
%     for d = 1:D
%         truePos(d) = sum( double(I(:) >= thresholds(d)) .* M(:) );
%         falsePos(d) = sum( double(I(:) >= thresholds(d)) .* (1 - M(:)) );
%     end
    indices = [1:H*W]';
    T = bsxfun(@(i, t)((I(i) >= t) .* M(i)), indices, thresholds);
    F = bsxfun(@(i, t)((I(i) >= t) .* (1-M(i))), indices, thresholds);
    % counts for ratios
    P = max(1, sum(M(:) > 0));
    N = max(1, sum(M(:) == 0));
    assert(P + N == H * W, 'Losing data');
    truePos = sum(T);
    falsePos = sum(F);
    % rate computation
    TPR = truePos / P;
    FPR = falsePos / N;
end