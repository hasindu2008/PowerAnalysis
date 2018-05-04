clear
clc

%define the wavelength of you traces here
wavelength=100000;
%since there are 18 keybytes
keybytes=1:8;
%since there are 256 possibilities for a byte
keyguess=1:256;

for i=keybytes
    for j=keyguess
        fig=figure;
        filename=sprintf('raw/subbyte%d_keyguess%d',i-1,j-1);
        file=fopen(filename);
        corr=fscanf(file,'%f');
        fclose(file);
        plot(1:wavelength,corr);
        filename=sprintf('graph/subbyte%d_keyguess%.2x',i-1,j-1);
        saveas(fig,filename,'fig');
        saveas(fig,filename,'jpg');
        close(fig);
        fprintf('keybyte : %d \tkeyguess : %d\n',i-1,j-1);
    end
end