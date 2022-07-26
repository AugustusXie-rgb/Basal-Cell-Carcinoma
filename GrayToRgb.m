% GrayToRgb

imgPath='D:\Basal_Cell_Carcinoma\Normal_skin_val\';
output_path='D:\Basal_Cell_Carcinoma\Normal_skin_val_rgb\';
imgDir=dir([imgPath '*.tif']);

for i=1:length(imgDir)
    im=imread([imgPath imgDir(i).name]);
    im_size=size(im);
    im_out=zeros(im_size(1),im_size(2),3);
    for j=1:3
        im_out(:,:,j)=im;
    end
    outputname=erase(imgDir(i).name,'.tif');
    imwrite(im_out/65535,[output_path outputname '_rgb.tif']);
    i
end