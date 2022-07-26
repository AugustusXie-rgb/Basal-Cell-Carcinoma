% Image_Augmentation

imgPath='F:\ICL_COVID\im_aug_v2\P\';
output_path='F:\ICL_COVID\im_aug_v2\P\';
% mean_log_comb=10.6141;
% std_log_comb=0.8647;

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

imgDir_bmp=dir([imgPath '*.bmp']);
imgDir_png=dir([imgPath '*.png']);
imgDir_jpg=dir([imgPath '*.jpg']);
imgDir=[imgDir_bmp;imgDir_png;imgDir_jpg];
im=imread([imgPath imgDir(1).name]);
imsize=size(im);
ref=zeros(imsize(1),imsize(2));
count=0;
% while count<20000-length(imgDir)
%     ID=randsrc(1,1,[1:length(imgDir)]);
%     im=imread([imgPath imgDir(ID).name]);
%     im=double(im(:,:,1));
%     im_line=im(:);
%     mean_all=mean(im_line);
%     std_all=std(im_line);
%     if psnr(im,ref)<0
%         psnr_all=psnr(im,ref);
%     else
%         psnr_all=-10^(-4);
%     end
%     comb=mean_all*std_all*(-psnr_all);
%     im_fil=zeros(imsize(1),imsize(2),8);
%     for j=1:8
%         im_fil(:,:,j)=imfilter(im,filters(:,:,j));
%     end
%     im_var=zeros(imsize(1),imsize(2));
%     for j=1:8
%         im_var=im_var+im_fil(:,:,j).^2;
%     end
% %     im_var=im_var/8;
% %     log_con=log(mean(mean(im_var)));
%     
% %     if log_con>mean_con-std_con && log_con<mean_con+std_con
%     if log(comb)>mean_log_comb-std_log_comb
%         %brightness 90%-110%
%         brightness=0.9+0.2*rand;
%         im=im*brightness;     
%     
%         %contrast 90%-110%
%         contrast=0.9+0.2*rand;
%         im=(im-mean(mean(im)))*contrast+mean(mean(im));
%     
%         %flip
%         lr=randsrc(1,1,[0,1]);
%         if lr
%             im=fliplr(im);
%         end
%         ud=randsrc(1,1,[0,1]);
%         if ud
%             im=flipud(im);
%         end
%     
%         %rotate90
%         r=randsrc(1,1,[0,1,2,3]);
%         im=rot90(im,r);
%         
%         %gaussian blur
%         gau=randsrc(1,1,[0,1]);
%         if gau
%             gau_fil=fspecial('gaussian',[10,10],5);
%             im=filter2(gau_fil,im);
%         end
%     
% %         crop/rotate_crop
%         d=randsrc(1,1,[0,1,2]);
%         if d==1
%             sec=randsrc(1,2,[1:3]);
%             im=im(200*(sec(1)-1)+1:200*(sec(1)-1)+600,200*(sec(2)-1)+1:200*(sec(2)-1)+600);
%             [xi,yi]=meshgrid(1:600/1001:600,1:600/1001:600);
%             im=interp2(im,xi,yi,'spline');
%         elseif d==2
%             deg=90*rand;
%             im=imrotate(im,-deg,'bicubic','loose');
%             rad=deg*pi/180;
%             x=round(1000*cos(rad)*sin(rad)/(sin(rad)+cos(rad)))+1;
%             im=im(x:end-x-1,x:end-x-1);
%             temp_size=size(im);
%             [xi,yi]=meshgrid(1:temp_size(1)/1001:temp_size(1),1:temp_size(2)/1001:temp_size(2));
%             im=interp2(im,xi,yi,'spline');
%         end
%     
%         outputname=erase(imgDir(ID).name,'.bmp');
%         outputname=erase(imgDir(ID).name,'.png');
%         im=cat(3,im,im,im);
%         imwrite(im/255,[output_path outputname '_aug_' int2str(count) '.bmp']);
%         count=count+1
%     end
% end
    
while count<10000-length(imgDir)
    ID=randsrc(1,1,[1:length(imgDir)]);
    im=imread([imgPath imgDir(ID).name]);
%     im=double(im(:,:,1));
    im=double(im);
    im_line=im(:);
    mean_all=mean(im_line);
    std_all=std(im_line);
%     if psnr(im,ref)<0
%         psnr_all=psnr(im,ref);
%     else
%         psnr_all=-10^(-4);
%     end
%     comb=mean_all*std_all*(-psnr_all);
%     im_fil=zeros(imsize(1),imsize(2),8);
%     for j=1:8
%         im_fil(:,:,j)=imfilter(im,filters(:,:,j));
%     end
%     im_var=zeros(imsize(1),imsize(2));
%     for j=1:8
%         im_var=im_var+im_fil(:,:,j).^2;
%     end
%     if log(comb)>mean_log_comb-std_log_comb
        %brightness 90%-110%
        brightness=0.9+0.2*rand;
        im=im*brightness;     

        %contrast 90%-110%
        contrast=0.9+0.2*rand;
        im=(im-mean(mean(im)))*contrast+mean(mean(im));

%         %flip
%         lr=randsrc(1,1,[0,1]);
%         if lr
%             im=fliplr(im);
%         end
%         ud=randsrc(1,1,[0,1]);
%         if ud
%             im=flipud(im);
%         end

%         %rotate90
%         r=randsrc(1,1,[0,1,2,3]);
%         im=rot90(im,r);

%         %crop/rotate_crop
%         d=randsrc(1,1,[0,1,2]);
%         if d==1
%             sec=randsrc(1,2,[1:3]);
%             im=im(200*(sec(1)-1)+1:200*(sec(1)-1)+600,200*(sec(2)-1)+1:200*(sec(2)-1)+600);
%             [xi,yi]=meshgrid(1:600/1001:600,1:600/1001:600);
%             im=interp2(im,xi,yi,'spline');
%         elseif d==2
%             deg=90*rand;
%             im=imrotate(im,-deg,'bicubic','loose');
%             rad=deg*pi/180;
%             x=round(1000*cos(rad)*sin(rad)/(sin(rad)+cos(rad)))+1;
%             im=im(x:end-x-1,x:end-x-1);
%             temp_size=size(im);
%             [xi,yi]=meshgrid(1:temp_size(1)/1001:temp_size(1),1:temp_size(2)/1001:temp_size(2));
%             im=interp2(im,xi,yi,'spline');
%         end
        outputname=erase(imgDir(ID).name,'.bmp');
        outputname=erase(imgDir(ID).name,'.png');  
        outputname=erase(imgDir(ID).name,'.jpg'); 
%         im=cat(3,im,im,im);
        count=count+1
        imwrite(im/255,[output_path outputname '_aug_' int2str(count) '.bmp']);
%     end
end  
    
