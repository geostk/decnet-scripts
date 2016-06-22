function [] = test_all_simple()

addpath('caffe/matlab/caffe');

img_files = rdir('Highlights/*/*/*/Original/*.png'); %rdir('Data/Baseline/*/*.png');
%img_files = rdir('/media/akaspar/BigBertha/akaspar/Dropbox/QCRI/Player Detection/Hockey/RGB/*.png');

cmap = jet(256);

N = numel(img_files);
p = randperm(N); % sampling frames

mode = 4;
suffix = { '', '-pyramid', '-retrained', '-retrained-voc-fullanno' };

for i = 1:N
    f = p(i);
    Fname = img_files(f).name;
    [Fdir, Bname, ~] = fileparts(Fname);
    [Pdir, section, ~] = fileparts(Fdir);
    if strcmp(section, 'Original')
        result_dir = fullfile(Pdir, ['DecoupledNet' suffix{mode}]);
        if ~exist(result_dir, 'dir')
            mkdir(result_dir);
        end
    else
        error('Not in an Original directory!\n')
    end
    
    Rdir = result_dir; % fullfile(result_dir, section);
    %Rdir = fullfile(Fdir, ['DecoupledNet' suffix{mode} ]);
    if ~exist(Rdir, 'dir')
        mkdir(Rdir);
    end
    
    Rfile = @(suffix)sprintf('%s/%s%s.png', Rdir, Bname, suffix);
    if exist(Rfile(''), 'file')
        fprintf('Skipping %s\n', Rfile(''));
        continue;
    else
        fprintf('Processing %s (%d of %d)\n', Rfile(''), i, N);
    end
        
    I = imread(Fname);
    switch mode
        case {1, 3, 4}
            S = decnet_segment(I);
        case 2
            S = decnet_pyramid(I, 0, 0.5);
        otherwise
            error('Invalid mode');
    end
    imwrite(S, Rfile(''));
    imwrite(labelize(S, 256), cmap, Rfile('-cmap'));
end

fprintf('\nDone.\n\n');