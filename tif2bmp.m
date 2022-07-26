% tif2bmp

imgPath='F:\Basal_Cell_Carcinoma\BCC_imagewise\All_sequence\all\';
output_path='F:\Basal_Cell_Carcinoma\BCC_imagewise\All_sequence\all\';
imgDir=dir([imgPath '*.tif']);

for i=1:length(imgDir)
    im=imread([imgPath imgDir(i).name]);
    outputname=erase(imgDir(i).name,' ');
    im=double(im(1:1000,1:1000));
    outputname=erase(imgDir(i).name,'.tif');
    if max(max(im))>255
        imwrite(im/65535,[output_path outputname '.bmp']);
    else
        imwrite(im/255,[output_path outputname '.bmp']);
    end
    delete([output_path imgDir(i).name]);
    i
end