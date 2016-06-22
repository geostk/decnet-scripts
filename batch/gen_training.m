function gen_training(output_dir)

    if nargin < 1
        output_dir = '/media/akaspar/BigBertha/akaspar/Data/DecoupledNet/Training';
    end
    % using rerouting with origin files
    reroute = 0;
    input_dir = '/media/akaspar/Ogrodzieniec/akaspar/Projects/PlayerSegmentation/aws/images';
    
    label_basedir = 'Data/Player Detection/MechanicalTurks';
    ref_size = 320;
    overlap = 0.5;
    
    masks = rdir(fullfile(label_basedir, '**/*.png'));
    
    filenum = 1;
    
    for i = 1:numel(masks)
        M = imread(masks(i).name);
        iname = strrep(masks(i).name, '.png', '.jpg');
        if exist(iname, 'file')
            % use jpg version
            I = imread(iname);
        elseif reroute
            % look for .origin in parents, then follow that path
            iname = find_origin( label_basedir, masks(i).name );
            I = imread(iname);
        else
            % read directly from a directory mirroring the mask directory
            I = imread(fullfile(input_dir, get_prefix(masks(i).name, '/')));
        end
        
        % name prefix
        prefix = get_prefix(masks(i).name);
        
        fprintf('%04d of %04d | Processing %s -> %s\n', i, numel(masks), masks(i).name, iname);
        
        % pyramidal decomposition
        [h, w, ch] = size(M);
        assert(h == size(I, 1) && w == size(I, 2), 'Mask and image do not match: %s', iname);
        tile_size = max(h, w);
        % have we already generated the tiles of this file?
        if exist(fullfile(output_dir, [prefix '-' num2str(tile_size) '-1-1-label.png']), 'file')
            fprintf('Skipping, already done\n');
            continue
        end
        % is it black and white?
        if ch == 3
            M = rgb2gray(M);
        end
        
        fprintf('Creating files:');
        
        while tile_size > ref_size
            delta = ceil(tile_size * (1 - overlap));
            for y1 = 1:delta:h
                y2 = min(y1 + tile_size - 1, h);
                for x1 = 1:delta:w
                    x2 = min(x1 + tile_size - 1, w);
                    
                    fprintf('%d, ', filenum);
                    filenum = filenum + 1;
                    
                    input = create_tile(I(y1:y2, x1:x2, :), ref_size);
                    label = create_tile(M(y1:y2, x1:x2, :), ref_size);
                    
                    
                    imwrite(input, fullfile(output_dir, [prefix '-' num2str(tile_size) '-' num2str(y1) '-' num2str(x1) '-input.png']));
                    imwrite(label, fullfile(output_dir, [prefix '-' num2str(tile_size) '-' num2str(y1) '-' num2str(x1) '-label.png']));
                end
            end
            tile_size = ceil(tile_size / 2);
        end
        fprintf('\n');
    end
    
end

function img = create_tile( img, ref_size )
    [h, w, ~] = size(img);
    im_sz = max(h, w);
    img = padarray(img, [im_sz - h, im_sz - w], 'post');
    img = imresize(img, [ref_size, ref_size]);
end

function name = basename( path )
    [~, f, e] = fileparts(path);
    name = [f e];
end

function prefix = get_prefix( path, sep )
    if nargin < 2, sep = '_'; end
    [parent, name, ext] = fileparts(path);
    prefix = [name ext];
    dirname = '';
    while ~strcmp(basename(parent), 'MechanicalTurks')
        [parent, name, ext] = fileparts(parent);
        prefix = [name ext sep prefix];
        dirname = [name ext];
    end
end

function path = read_origin( file )
    f = fopen(file);
    path = fgetl(f);
    fclose(f);
end

function iname = find_origin( origin_base, path )
    [parent, fname, ext] = fileparts(path);
    subpath = [fname ext];
    while numel(parent) > 1
        ori_file = fullfile(parent, '.origin');
        if exist(ori_file, 'file')
            
            % load origin content
            newpath = read_origin( ori_file );
            iname = fullfile(origin_base, newpath, subpath);
            if exist(iname, 'file')
                return
            else
               error('Broken origin: %s -> %s -> %s', path, newpath, iname); 
            end
            
        else
            % go to parent and store the directory
            [parent, dir1, dir2] = fileparts(parent);
            subpath = fullfile([dir1 dir2], subpath);
        end
    end
    error('Did not find an origin file for %s', path)
end