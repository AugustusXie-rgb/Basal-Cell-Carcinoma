% folder_image_split

folder='F:\Basal_Cell_Carcinoma\comp_data\seq\N_seq_4cut';
output_folder='F:\Basal_Cell_Carcinoma\comp_data\seq\N_seq_16cut';
folderlist=dir(folder);
folderlist=folderlist(3:end);

for f=1:length(folderlist)
    imgPath=strcat(folder,'\',folderlist(f).name,'\');
    mkdir(output_folder,folderlist(f).name);
    output_path=strcat(output_folder,'\',folderlist(f).name,'\');
    imgDir=dir([imgPath '*.bmp']);
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
    end
    f
end