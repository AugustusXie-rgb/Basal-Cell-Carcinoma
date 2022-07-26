% sort_input

BCC=zeros(21,786);
NS=zeros(21,783);

for i=1:786
    BCC(1:16,i)=BCC16(16*i-15:16*i);
    BCC(17:20,i)=BCC4(4*i-3:4*i);
    BCC(21,i)=BCC1(i);
end
for i=1:783
    NS(1:16,i)=NS16(16*i-15:16*i);
    NS(17:20,i)=NS4(4*i-3:4*i);
    NS(21,i)=NS1(i);
end