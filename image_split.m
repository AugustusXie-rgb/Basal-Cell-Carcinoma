% image_split

imgPath='D:\Basal Cell Carcinoma\Grad_CAM\Original\4cut\';
output_path='D:\Basal Cell Carcinoma\Grad_CAM\Original\16cut\';

imgDir_bmp=dir([imgPath '*.bmp']);
imgDir_png=dir([imgPath '*.png']);
imgDir=[imgDir_bmp;imgDir_png];

for i=1:length(imgDir)
    im=imread([imgPath imgDir(i).name]);
    imsize=size(im);
    cen=round(imsize/2);
    im_1=im(1:cen(1),1:cen(2),:);
    im_2=im(1:cen(1),cen(2)+1:end,:);
    im_3=im(cen(1)+1:end,1:cen(2),:);
    im_4=im(cen(1)+1:end,cen(2)+1:end,:);
    imwrite(im_1,[output_path imgDir(i).name '_1.bmp']);
    imwrite(im_2,[output_path imgDir(i).name '_2.bmp']);
    imwrite(im_3,[output_path imgDir(i).name '_3.bmp']);
    imwrite(im_4,[output_path imgDir(i).name '_4.bmp']);
    i
end