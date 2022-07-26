% contrast_cal

imgPath='F:\Basal_Cell_Carcinoma\comp_data\4class\N_NB\train\';
imgDir=dir([imgPath '*.bmp']);

contrast=zeros(1,length(imgDir));

filters=zeros(3,3,9);
filters(2,2,:)=1;
for i=1:3
    for j=1:3
        if i~=2 || j~=2
            filters(i,j,3*(i-1)+j)=-1;
        end
    end
end
filters=cat(3,filters(:,:,1:4),filters(:,:,6:9));

for i=1:length(imgDir)
    im=imread([imgPath imgDir(i).name]);
    im=double(im(:,:,1));
    imsize=size(im);
    im_fil=zeros(imsize(1),imsize(2),8);
    for j=1:8
        im_fil(:,:,j)=imfilter(im,filters(:,:,j));
    end
    im_var=zeros(imsize(1),imsize(2));
    for j=1:8
        im_var=im_var+im_fil(:,:,j).^2;
    end
    im_var=im_var/8;
    contrast(i)=mean(mean(im_var));
    i
end

max_contrast=max(contrast);
log_con=log(contrast);
mean_con=mean(log_con);
std_con=std(log_con);

% for i=1:length(imgDir)
%     im=imread([imgPath imgDir(i).name]);
%     imwrite(im,['D:\Basal_Cell_Carcinoma\BCC_contrast\' int2str(10*contrast(i)/max_contrast) '_' imgDir(i).name]);
% end
    
