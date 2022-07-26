% tif2bmp_batch

maindir='F:\Basal_Cell_Carcinoma\comp_data\seq\B_seq_1cut';
subdir=dir(maindir);

for i=3:length(subdir)
    imgPath=[maindir,'\',subdir(i).name,'\'];
    output_path=imgPath;
    imgDir=dir([imgPath '*.tif']);
    for j=1:length(imgDir)
        im=imread([imgPath imgDir(j).name]);
        outputname=erase(imgDir(j).name,' ');
        im=double(im(1:1000,1:1000));
        outputname=erase(imgDir(j).name,'.tif');
        if max(max(im))>255
            imwrite(im/65535,[output_path outputname '.bmp']);
        else
            imwrite(im/255,[output_path outputname '.bmp']);
        end
    end
    subdir(i).name
end