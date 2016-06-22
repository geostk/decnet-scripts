function preprocessed_img = decnet_preprocess(img, img_sz)

meanImg = [104.00698793, 116.66876762, 122.67891434]; % order = bgr
meanImg = repmat(meanImg, [img_sz^2,1]);
meanImg = reshape(meanImg, [img_sz, img_sz, 3]); 

resized = imresize(double(img), [img_sz img_sz], 'bilinear'); % resize cropped image
resized = resized(:,:,[3 2 1]) - meanImg; % convert color channer rgb->bgr and subtract mean 
preprocessed_img = single(permute(resized, [2 1 3])); 

end