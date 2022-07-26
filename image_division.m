% image_division

imgPath='D:\Basal_Cell_Carcinoma\NS_bmp\NS_train\';
imgDir=dir([imgPath '*.bmp']);

for i=1:length(imgDir)
    im=imread([imgPath imgDir(i).name]);
    outputname=erase(imgDir(i).name,'.bmp');
    
    % 4 fold
    outputpath='D:\Basal_Cell_Carcinoma\NS_bmp\NS_4_train\';
    for m=1:2
        for n=1:2
            im_part=im((m-1)*500+1:m*500,(n-1)*500+1:n*500);
            imwrite(im_part,[outputpath outputname '_4_' int2str(2*(m-1)+n) '.bmp']);
        end
    end
%     
%     % 9 fold
%     outputpath='D:\Basal_Cell_Carcinoma\Normal skin sort\bmp\Normal_skin_9_val\';
%     for m=1:3
%         for n=1:3
%             im_part=im((m-1)*333+1:m*333,(n-1)*333+1:n*333);
%             imwrite(im_part,[outputpath outputname '_9_' int2str(3*(m-1)+n) '.bmp']);
%         end
%     end
    
    % 16 fold
    outputpath='D:\Basal_Cell_Carcinoma\NS_bmp\NS_16_train\';
    for m=1:4
        for n=1:4
            im_part=im((m-1)*250+1:m*250,(n-1)*250+1:n*250);
            imwrite(im_part,[outputpath outputname '_16_' int2str(4*(m-1)+n) '.bmp']);
        end
    end    
    
    i
end