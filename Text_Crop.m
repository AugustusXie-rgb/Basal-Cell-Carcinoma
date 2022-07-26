% Text_Crop

imgPath='D:\Basal_Cell_Carcinoma\BCC sort\';
imgDir=dir([imgPath '*.tif']);

for i=1:length(imgDir)
    temp=imread([imgPath imgDir(i).name]);
    img_size=size(temp);
    if img_size(1)>1000
        temp=temp(1:1000,:);
        imwrite(temp,[imgPath imgDir(i).name]);
    end
    i
end