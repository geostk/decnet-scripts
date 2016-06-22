function [] = test_all_layered()

addpath('caffe/matlab/caffe');

img_files = rdir('Highlights/*/*/*/Original/*.png'); %rdir('Data/Baseline/*/*.png');

N = numel(img_files);
permute = 1;

% order for faster sampling (with permutation => see various results soon)
if permute
    p = randperm(N);
else
    p = 1:N;
end

for i = 1:N
    f = p(i);
    Fname = img_files(f).name;
    [Fdir, Bname, ~] = fileparts(Fname);
    [Pdir, section, ~] = fileparts(Fdir);
    if strcmp(section, 'Original')
        Rdir = fullfile(Pdir, 'FasterRCNN');
    else
        error('Not in an Original directory!\n')
    end
    
    if ~exist(Rdir, 'dir')
        mkdir(Rdir);
    end
    
    Mfile = sprintf('%s/%s.mat', Rdir, Bname);
    Lfile = @(level) sprintf('%s/%s-l%d.png', Rdir, Bname, level);
    Rfile = sprintf('%s/%s.png', Rdir, Bname);
    if ~exist(Mfile, 'file')
        fprintf('No boxes for %s\n', Mfile);
        continue
    end
    if exist(Rfile, 'file')
        fprintf('Skipping %s\n', Rfile);
        continue
    else
        fprintf('Processing %s (%d of %d)\n', Rfile, i, N);
    end
        
    I = imread(Fname);
    boxes = load_mat(Mfile);
    [S, L] = decnet_layered(I, boxes);
    for l = 1:numel(L)
        imwrite(L{l}, Lfile(l));
    end
    imwrite(S, Rfile);
    subplot(121); imshow(I); subplot(122); imagesc(S); axis image;
end

fprintf('\nDone.\n\n');