function [] = process_all()

addpath('caffe/matlab/caffe');

img_files = rdir('Data/2D-3D/WorkSpace/Mohamed/sequence_1/Original/*.png'); %rdir('Data/Baseline/*/*.png');

result_dir = 'Data/2D-3D/WorkSpace/Mohamed/sequence_1/Segmentation/'; % 'Data/Res';
cmap = jet(256);

N = numel(img_files);
p = randperm(N); % sampling frames
% img_files(p(1)).name = 'Data/2D-3D/WorkSpace/Mohamed/sequence_1/Original/552.png';

use_pyramid = 1;
if use_pyramid
    result_dir = fullfile(result_dir, 'pyramid-0.5');
end

for i = 1:N
    f = p(i);
    Fname = img_files(f).name;
    [Fdir, Bname, ~] = fileparts(Fname);
    [~, section, ~] = fileparts(Fdir);
    
    Rdir = result_dir; % fullfile(result_dir, section);
    
    trg_init = sprintf('%s/%s_init.jpg', Rdir, Bname);
    if exist(trg_init, 'file')
        fprintf('Skipping %s/%s\n', section, Bname);
        continue;
    else
        mkdir(Rdir);
        fprintf('Processing %s/%s (%d of %d)\n', section, Bname, i, N);
    end
        
    I = imread(Fname);
    
    if use_pyramid
        
        overlap = 0.5;
        %for overlap = [0.3 0.5 0.7]
        t = tic;
        [S, pyr, masks] = decnet_pyramid(I, 0, overlap);
        toc(t);
        copyfile(Fname, trg_init);
        imwrite(S, sprintf('%s/%s-gray.png', Rdir, Bname));
        imwrite(labelize(S), cmap, sprintf('%s/%s-cmap.png', Rdir, Bname));
        % end
    else
        
        fprintf('Using automatic min score.\n');
    
        [S, S0] = decnet_multiscale(I, 1.5, []); % automatic min score

        copyfile(Fname, trg_init);
        imwrite(S0, sprintf('%s/%s_base-gray.png', Rdir, Bname));
        imwrite(S, sprintf('%s/%s_multi-gray.png', Rdir, Bname));
        imwrite(labelize(S0), cmap, sprintf('%s/%s_base-cmap.png', Rdir, Bname));
        imwrite(labelize(S), cmap, sprintf('%s/%s_multi-cmap.png', Rdir, Bname));

        % fixed min score
        for m = [0.1 0.05 0.01]
            fprintf('Using min score %f\n', m);
            [S1, ~] = decnet_multiscale(I, 1.5, m);
            Sdir = fullfile(Rdir, num2str(m));
            mkdir(Sdir);
            imwrite(S1, sprintf('%s/%s_multi-gray.png', Sdir, Bname));
            imwrite(labelize(S1), cmap, sprintf('%s/%s_multi-cmap.png', Sdir, Bname));
        end
        
    end
end

fprintf('\nDone.\n\n');