% B_score

% input pre_14
l=length(pre_14);
score=0;
count=0;
base=1/l;
total_score=0;

ind_8=pre_14>=0.8;
state_8=double(ind_8);
ind_6=pre_14>=0.6;
state_6=double(ind_6);
ind_4=pre_14>=0.4;
state_4=double(ind_4);
ind_2=pre_14>=0.2;
state_2=double(ind_2);
for i=2:l
    if state_8(i)>0
        state_8(i)=state_8(i-1)+1;
    end
    if state_6(i)>0
        state_6(i)=state_6(i-1)+1;
    end
    if state_4(i)>0
        state_4(i)=state_4(i-1)+1;
    end
    if state_2(i)>0
        state_2(i)=state_2(i-1)+1;
    end
end
for i=l:-1:2
    if state_8(i)>0 && state_8(i-1)>0
        state_8(i-1)=state_8(i);
    end
    if state_6(i)>0 && state_6(i-1)>0
        state_6(i-1)=state_6(i);
    end
    if state_4(i)>0 && state_4(i-1)>0
        state_4(i-1)=state_4(i);
    end
    if state_2(i)>0 && state_4(i-1)>0
        state_2(i-1)=state_2(i);
    end
end
part_8=max(0,pre_14-0.8);
part_6=max(0,pre_14-0.6);
part_6=min(0.2,part_6);
part_4=max(0.4,pre_14);

total_score=0.01;
for i=1:l
    total_score=total_score+part_8(i)*2^(2*state_8(i))+part_6(i)*2^state_6(i)+part_4(i);
end
log(total_score*base)