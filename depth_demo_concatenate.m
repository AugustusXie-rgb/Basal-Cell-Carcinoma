% depth_demo_concatenate

img_dir='D:\Basal Cell Carcinoma\depth_map\159\';
imglist=dir([img_dir '*.bmp']);
im_cat=[];
for i=10:length(imglist)
    im=imread([img_dir imglist(i).name]);
    im_cat=cat(2,im_cat,im);
end
for i=1:9
    im=imread([img_dir imglist(i).name]);
    im_cat=cat(2,im_cat,im);
end
imshow(im_cat)
    