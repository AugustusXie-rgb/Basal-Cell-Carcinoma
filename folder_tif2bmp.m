% folder_tif2bmp

folder='F:\Basal_Cell_Carcinoma\comp_data\seq\N_seq_1cut';
folderlist=dir(folder);
folderlist=folderlist(3:end);

for f=1:length(folderlist)
    imgPath=strcat(folder,'\',folderlist(f).name,'\');
    imgDir=dir([imgPath '*.png']);
    for i=1:length(imgDir)
        im=imread([imgPath imgDir(i).name]);
        outputname=erase(imgDir(i).name,' ');
        im=double(im(1:1000,1:1000));
        outputname=erase(imgDir(i).name,'.png');
        if max(max(im))>255
            imwrite(im/65535,[imgPath outputname '.bmp']);
        else
            imwrite(im/255,[imgPath outputname '.bmp']);
        end
        delete([imgPath imgDir(i).name]);
%         i
    end
    f
end