% quality_filter

img_folder='F:\Basal_Cell_Carcinoma\BCC_imagewise\All_sequence\all\';
imgDir=dir([img_folder '*.bmp']);
mean_all=zeros(1,length(imgDir));
std_all=zeros(1,length(imgDir));
psnr_all=zeros(1,length(imgDir));
% ref=zeros(1000);

for i=1:length(imgDir)
    im=imread([img_folder imgDir(i).name]);
    im=double(im);
    im_line=im(:);
    mean_all(i)=mean(im_line);
    std_all(i)=std(im_line);
%     if psnr(im,ref)<0
%         psnr_all(i)=psnr(im,ref);
%     else
%         psnr_all(i)=-10^(-4);
%     end
end

[mean_sort,pos]=sort(mean_all(1,:));

% comb=mean_all.*std_all.*(-psnr_all);
% [comb_sort,pos]=sort(comb(1,:));
% 
% sort_name_list={};
% for i=1:length(imgDir)
%     sort_name_list=[sort_name_list;imgDir(pos(i)).name];
% end

% log_comb_sort=log(comb_sort);
mean_log_comb=10.6141;
std_log_comb=0.8647;
threshold_1=mean_log_comb-3*std_log_comb;
threshold_2=mean_log_comb-2*std_log_comb;
threshold_3=mean_log_comb-std_log_comb;
