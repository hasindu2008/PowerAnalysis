clear
clc

%minimum number of power traces
min=50;
%maximum number of power traces
max=1000;
%step for number of power traces
step=50;

data=cell(1);
y=cell(1);
x=min:step:max;
repeat=(max-min)/step+1;

cc=hsv(256);

n=1;
for i=x
   filename=sprintf('source/all_%d.txt',i);
   size=[8 256];
   file=fopen(filename);
   readdata=fscanf(file,'%f',size);
   data{n}=readdata';
   n=n+1;
end

for keybyte=1:8
    fig=figure;
    hold;
    figname=sprintf('Keybyte : %d',keybyte-1);
    title(figname);
    for guess=1:256
        for num=1:repeat
        y{guess}(num)=data{num}(guess,keybyte);
        end
    plot(x,y{guess},'color',cc(guess,:));  
    end
    
    xlabel('Number of traces');
    ylabel('Correlation');
    filename=sprintf('graphs/%d',keybyte);
    saveas(fig,filename,'fig')
    saveas(fig,filename,'jpg')
   
end