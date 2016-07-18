function [] = gen_player_masks( base_dir )

    addpath('..');
    addpath('util');

    if nargin < 1 || isempty(base_dir)
        base_dir = '/media/Mandelbrot/akaspar/Tmp/Sequences/';
    end
    
    images = rdir(fullfile(base_dir, '**/image*.png'));
    N = numel(images);
    
    for i = 1:N
        fprintf('[%d of %d] %s\n', i, N, path_suffix(images(i).name, 3));
        [image_dir, name, ext] = fileparts(images(i).name);
        
        % create output dir if needed
        output_dir = fullfile(image_dir, 'players');
        if ~exist(output_dir, 'dir')
            mkdir(output_dir);
        end
        
        % check if result already done
        if exist(fullfile(output_dir, [name ext]), 'file')
            fprintf('Skipping, already done.\n');
            continue
        end
        
        % compute player mask
        I = imread(images(i).name);
        S = decnet_segment(I);
        
        % save player mask
        imwrite(S, fullfile(output_dir, [name ext]));
    end

end

function suffix = path_suffix(file, count)
    suffix = '';
    while count > 0
        [parent, name, ext] = fileparts(file);
        if isempty(suffix)
            suffix = [name ext];
        else
            suffix = fullfile([name ext], suffix);
        end
        file = parent;
        count = count - 1;
    end
end