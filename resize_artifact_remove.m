% resize_artifact_remove

for i=1:131
    mkdir(int2str(i));
    imgPath=['F:\Basal_Cell_Carcinoma\group_no_aug\BCC_group\',int2str(i),'\'];
    output_path=['F:\Basal_Cell_Carcinoma\group_no_aug\BCC_group_crop\',int2str(i),'\'];
    imgDir=dir([imgPath '*.bmp']);
    
    for j=1:length(imgDir)
        im=imread([imgPath imgDir(j).name]);
        im=im(51:950,101:1000);
        imwrite(im,[output_path imgDir(j).name]);
    end
    i
end