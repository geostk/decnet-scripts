function final_img = decnet_postprocess(img, ori_sz)
resized = imresize(double(img), ori_sz(1:2), 'bicubic'); % resize cropped image
final_img = permute(resized, [2 1 3]); 
end