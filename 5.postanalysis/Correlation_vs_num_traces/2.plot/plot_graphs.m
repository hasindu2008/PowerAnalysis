% The A to Z of Building a Testbed for Power Analysis Attacks
% Matlab script for plotting how correlation varies with number of power traces 
% First generate the data using the CPA algorithm under 1.generate_data
% Copy all_*.txt generated from there to the folder named "source"
%Then run this matlab script to get graphs for each keybyte in the folder "graphs"

% Authors : Hasindu Gamaarachchi, Harsha Ganegoda and Roshan Ragel, 
% Department of Computer Engineering, 
% Faculty of Engineering, University of Peradeniya, 22 Dec 2015
 
% For more information read 
% Hasindu Gamaarachchi, Harsha Ganegoda and Roshan Ragel, 
% "The A to Z of Building a Testbed for Power Analysis Attacks", 
% 10th IEEE International Conference on Industrial and Information Systems 2015 (ICIIS)]
 
% Any bugs, issues or suggestions please email to hasindu2008@live.com

clear
clc


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CHANGE THESE PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% minimum number of power traces. Should match what you gave when generating data
min_traces=10

% maximum number of power traces to test. Should match what you gave when generating data
max_traces=200

% step value for number of traces. Should match what you gave when generating data
step_traces=10

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END OF CHANGEABLE THESE PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


data=cell(1);
y=cell(1);

%x axis values
x=min_traces:step_traces:max_traces;

%number of repetitions (size of x axis values)
repeat=(max_traces-min_traces)/step_traces+1;

%color value for each key in a graph
cc=hsv(256);

n=1;
%read the data from files in source
for i=x
   filename=sprintf('source/all_%d.txt',i);
   size=[16 256];
   file=fopen(filename);
   readdata=fscanf(file,'%f',size);
   data{n}=readdata';
   n=n+1;
end

%generate graphs
for keybyte=1:16
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
    
    xlabel('Number of power traces');
    ylabel('Correlation coefficient');
    filename=sprintf('graphs/keybyte_%d',keybyte);
    saveas(fig,filename,'fig')
    saveas(fig,filename,'jpg')
   
end