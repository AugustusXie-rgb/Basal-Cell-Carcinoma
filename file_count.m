% file_count

folder='F:\Basal_Cell_Carcinoma\comp_data\seq\N_seq_1cut';
folderlist=dir(folder);
folderlist=folderlist(3:end);
l=[];
for f=1:length(folderlist)
    imgPath=strcat(folder,'\',folderlist(f).name,'\');
    imgDir=dir([imgPath '*.bmp']);
    l=[l,length(imgDir)];
end