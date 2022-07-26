% depth_demo_output

% get depth and pre_14
color=jet;
% if max(pre_14)>0.95
%     pre_14=0.95*pre_14;
% elseif max(pre_14)>0.9
%     pre_14=0.9*pre_14;
% end

d=depth*10;
gap=round((d(end)-d(1))/(length(d)-1));
d=round(d);
map=zeros(d(end)+gap,10,3);
weight=0;

for i=1:length(d)
    weight=weight+pre_14(i)*depth(i);
    for j=d(i)+1:d(i)+gap
        c=1+round(255*pre_14(i));
        if c>256
            c=256;
        end
        map(j,:,1)=color(c,1);
        map(j,:,2)=color(c,2);
        map(j,:,3)=color(c,3);
    end
end
round(weight/sum(pre_14))
imshow(map);

