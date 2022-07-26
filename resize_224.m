% resize_224

imgPath='D:\Basal_Cell_Carcinoma\Normal skin sort\bmp\NS_1_val\';
imgDir=dir([imgPath '*.bmp']);

for i=1:length(imgDir)
    im=imread([imgPath imgDir(i).name]);
    im_out=zeros(224,224,3);
    for k=1:3
        im_out(:,:,k)=imresize(im,224/1000);
    end
    
    outputpath='D:\Basal_Cell_Carcinoma\keras\NS_1_val\';
    imwrite(im_out/255,[outputpath,imgDir(i).name]);
    i
end
    
    