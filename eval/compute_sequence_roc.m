function [TPR, FPR, D] = compute_sequence_roc( base_dir, prefix, thresh )
    if nargin < 1, base_dir = 'Highlights/1/Complex/1'; end
    if nargin < 2,  prefix = 'DecoupledNet'; end
    if nargin < 3,  thresh = linspace(0.0, 1.0, 100); end
    
    res_dir = fullfile(base_dir, prefix);
    ann_dir = fullfile(base_dir, 'Annotated');
    need_data = 1; % nargout > 2;
    
    files = rdir(sprintf('%s/*.png', res_dir));
    TPR = 0; FPR = 0;
    N = numel(files);
    for i = 1:N
        % load segmentation and reference ground truth
        file = files(i).name;
        [~, base, ext] = fileparts(file);
        fprintf('#%3d/%3d: %s%s\n', i, N, base, ext);
        anno = fullfile(ann_dir, [base ext]);
        I = im2double(imread(file));
        M = im2double(imread(anno));
        % compute true and false positive rates
        [T, F, Pos, Neg] = pixel_roc(I, M, thresh);
        % accumulate
        TPR = TPR + T;
        FPR = FPR + F;
        % store data if needed
        if need_data
            [H, W, ~] = size(M);
            D(i).T = T;
            D(i).F = F;
            D(i).P = Pos;
            D(i).N = Neg;
            D(i).H = H;
            D(i).W = W;
        end
    end
    TPR = TPR / N;
    FPR = FPR / N;
    % write data
    if need_data
        save(fullfile(base_dir, [prefix '.roc.mat']), 'D');
    end
    % generate ROC plot
    addpath('export_fig');
    h = plot(FPR, TPR, '-ob');
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title('ROC for thresh=[0;1]');
    set(h, 'Visible', 'off');
    export_fig(fullfile(base_dir, [prefix '.roc.png']), '-m2');
end