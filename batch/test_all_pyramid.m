function [] = test_all_pyramid()

addpath('caffe/matlab/caffe');

img_files = rdir('Highlights/*/*/*/Original/*.png'); %rdir('Data/Baseline/*/*.png');

N = numel(img_files);
p = randperm(N); % sampling frames

for i = 1:N
    f = p(i);
    Fname = img_files(f).name;
    [Fdir, Bname, ~] = fileparts(Fname);
    [Pdir, section, ~] = fileparts(Fdir);
    if strcmp(section, 'Original')
        Rdir = fullfile(Pdir, 'DecoupledNet-pyramid');
        Rdir2 = fullfile(Pdir, 'DecoupledNet-mpyramid');
        Rdir3 = fullfile(Pdir, 'DecoupledNet-pyramid-scales');
    else
        error('Not in an Original directory!\n')
    end
    
    if ~exist(Rdir, 'dir')
        mkdir(Rdir);
    end
    if ~exist(Rdir2, 'dir')
        mkdir(Rdir2);
    end
    if ~exist(Rdir3, 'dir')
        mkdir(Rdir3);
    end
    
    Rfile = sprintf('%s/%s.png', Rdir, Bname);
    Rfile2 = sprintf('%s/%s.png', Rdir2, Bname);
    RfileS = @(i)sprintf('%s/%d/%s.png', Rdir3, i, Bname);
    if exist(Rfile, 'file') && exist(Rfile2, 'file') && exist(RfileS(1), 'file')
        fprintf('Skipping %s\n', Rfile);
        continue;
    else
        fprintf('Processing %s (%d of %d)\n', Rfile, i, N);
    end
        
    I = imread(Fname);
    [S, E, P, M] = decnet_pyramid(I, 0, 0.5);
    imwrite(S, Rfile);
    imwrite(E, Rfile2);
    for l = 1:numel(P)-1
        Si = P{end-l} ./ M{end-l};
        if ~exist(fullfile(Rdir3, num2str(l)), 'dir')
            mkdir(fullfile(Rdir3, num2str(l)));
        end
        imwrite(Si, RfileS(l));
    end
end

fprintf('\nDone.\n\n');