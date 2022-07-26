% heatmap_inclusion

% im: Grad_CAM output;  heatmap: 4-16SVM output
im=im/255;
heatmap=imresize(heatmap,250);

% im_ind=im>=mean(im(:));
% heat_ind=heatmap>=mean(heatmap(:));
im_ind=im>=0.5;
heat_ind=heatmap>=0.5;
overlap=im_ind.*heat_ind;
sum(overlap(:))/sum(im_ind(:))
