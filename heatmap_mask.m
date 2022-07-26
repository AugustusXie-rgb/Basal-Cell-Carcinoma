% heatmap_mask

% im: Grad_CAM output;  heatmap: 4-16SVM output;  ori: original RCM image
im=round(im);
im=max(0,im);
im=min(255,im);
heatmap=round(255*imresize(heatmap,250));
heatmap=max(0,heatmap);
heatmap=min(255,heatmap);

colormap=hsv;
colormap=colormap(1:172,:);
colormap=imresize(colormap,[256,3]);
colormap=flipud(colormap);

ori=double(ori);
Grad_mask=zeros(1000,1000,3);
heat_mask=zeros(1000,1000,3);
for i=1:1000
    for j=1:1000
        Grad_mask(i,j,:)=ori(i,j)*colormap(im(i,j)+1,:);
        heat_mask(i,j,:)=ori(i,j)*colormap(heatmap(i,j)+1,:);
    end
end
Grad_mask=Grad_mask/255;
heat_mask=heat_mask/255;
figure(1)

imshow(Grad_mask);
figure(2)
imshow(heat_mask);
figure(3)   
imshow(ori/255);