% SVM_heatmap

heat_4cut=zeros(2);
heat_16cut=zeros(4);
heat_4cut(1,1)=cut(1);
heat_4cut(1,2)=cut(2);
heat_4cut(2,1)=cut(3);
heat_4cut(2,2)=cut(4);
heat_16cut(1,1)=cut1(1);
heat_16cut(1,2)=cut1(2);
heat_16cut(2,1)=cut1(3);
heat_16cut(2,2)=cut1(4);
heat_16cut(1,3)=cut1(5);
heat_16cut(1,4)=cut1(6);
heat_16cut(2,3)=cut1(7);
heat_16cut(2,4)=cut1(8);
heat_16cut(3,1)=cut1(9);
heat_16cut(3,2)=cut1(10);
heat_16cut(4,1)=cut1(11);
heat_16cut(4,2)=cut1(12);
heat_16cut(3,3)=cut1(13);
heat_16cut(3,4)=cut1(14);
heat_16cut(4,3)=cut1(15);
heat_16cut(4,4)=cut1(16);
heat_4cut=(heat_4cut-min(min(heat_4cut)))./(max(max(heat_4cut))-min(min(heat_4cut)));
heat_16cut=(heat_16cut-min(min(heat_16cut)))./(max(max(heat_16cut))-min(min(heat_16cut)));

heatmap=0.25*imresize(heat_4cut,2)+0.75*heat_16cut;
