% The A to Z of Building a Testbed for Power Analysis Attacks
% Matlab script for plotting how the correlation coefficient varies with time

% First generate the data using the CPA algorithm under 1.generate_data
% It will generate 4096 files under names subbyte<x>_keyguess<y> where <x> and <y> are integers
% Copy those subbyte<x>_keyguess<y> files to the folder named "source"
% Then run this matlab script to get 4096 graphs for each combination of keybyte and keyguess in the folder "graphs"

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CONFIGURATION PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%The number of sampling points in a single power trace
%This value can be found by inspecting the stat.txt generated after collecting a set of power traces*/
sample_points=100000;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  END OF CONFIGURATION PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% There are 16 keybytes in an AES 128 bit key
keybytes=1:16;
% For each keybyte there are 256 possible keyguesses
keyguess=1:256;

%Draw a plot for each combination of keybyte and keyguess
for i=keybytes
    for j=keyguess
        fig=figure;
        filename=sprintf('source/subbyte%d_keyguess%d',i-1,j-1);
        file=fopen(filename);
        corr=fscanf(file,'%f');
        fclose(file);
        plot(1:sample_points,corr);
        filename=sprintf('graphs/subbyte%d_keyguess%.2x',i-1,j-1);
        saveas(fig,filename,'fig');
        saveas(fig,filename,'jpg');
        close(fig);
        fprintf('keybyte : %d \tkeyguess : %d\n',i-1,j-1);
    end
end