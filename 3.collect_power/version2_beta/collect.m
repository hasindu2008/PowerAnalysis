% The A to Z of Building a Testbed for Power Analysis Attacks
% Matlab script for collecting power traces using an oscilloscope

% Authors : Hasindu Gamaarachchi, Harsha Ganegoda and Roshan Ragel, 
% Department of Computer Engineering, 
% Faculty of Engineering, University of Peradeniya, 22 Dec 2015
 
% For more information read 
% Hasindu Gamaarachchi, Harsha Ganegoda and Roshan Ragel, 
% "The A to Z of Building a Testbed for Power Analysis Attacks", 
% 10th IEEE International Conference on Industrial and Information Systems 2015 (ICIIS)]
 
% Any bugs, issues or suggestions please email to hasindu2008@live.com


%start timer
tic 

%check whether continuation or fresh start
if exist('lastposition.txt')==0
    clear;
    clc;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Parameter setup %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Serial port of the cryptographic device.  
serialport='COM3';

%no of samples required to be captured. Make sure that plain text file has number of samples equal or greater than this number
last=200;

%plain text sample file
inputfile='plain.txt';

%output file for encrypted text that is coming from the cryptographic device
outputfile='cipher_device.txt';

%output format for power traces. 
% put 0 for ASCII format power traces. Readable but file size would be high
% put 1 to save power traces in binary. Note readable directly, but file size is less
waveformat=1;

%output file for power traces
wavefile='wave.dat';

%file to save statistics about the power capture. Information such as the the number of samples in a power trace, time taken
%would be saved here
stat='stat.txt';

%if need to verify cipher text coming from the cryptographic device  by comparing with "verifyfile" set this to 1. Else 0.
verifyen=0;
%if verify set to 1 the file name for test vectors
verifyfile='ciphertest.txt';

%parameter to do just encryption or do power collection. 
%Set to 0 only if you are testing the cryptosystem functionality without collecting power
%For power collection this should be 1
pwrcol=1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Oscilloscope setup%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This part may require modification to match your oscilloscope

if (pwrcol==1)
    
    % Create a VISA-USB object. Change the identifier by the one corresponding to your oscilloscope.
	% You can find this identifier in Instrument Control Toolbox
    interfaceObj = instrfind('Type', 'visa-usb', 'RsrcName', 'USB0::0x0699::0x039E::C010474::0::INSTR', 'Tag', '');

    % Create the VISA-USB object if it does not exist
    % otherwise use the object that was found.
    if isempty(interfaceObj)
		%change the identifier by the one corresponding to your oscilloscope.
		% You can find this identifier in Instrument Control Toolbox
        interfaceObj = visa('TEK', 'USB0::0x0699::0x039E::C010474::0::INSTR');
    else
        fclose(interfaceObj);
        interfaceObj = interfaceObj(1);
    end

    % Create a device object. 
	% This requires to point to the name of the MATLAB instrument driver wrapper file corresponding to your oscilloscope.
	% Check if the driver wrapper for you device is in <program files>\MATLAB\R2013a\toolbox\instrument\instrument\drivers
	% Else check whether they are available on Internet to download
	% Otherwise you will have to create them from IVI drivers
    deviceObj = icdevice('MSO4032.mdd', interfaceObj);

    %set buffer size
    interfaceObj = get(deviceObj, 'interface');
    % Allow for a 10000000 point waveform with a little extra room.
    set(interfaceObj, 'InputBufferSize', 20001000);

    % Connect device object to hardware.
    connect(deviceObj);

    % Execute device object function(s).
    groupObj = get(deviceObj, 'Waveform');
    groupObj = groupObj(1);
   
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Cryptosystem interface setup%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%check if we are collecting a new set of power traces from beginning or continue a previously interrupted session
if exist('lastposition.txt')==0
    file=fopen('lastposition.txt','w');
    fprintf(file,'%d',0);
    fclose(file);
    temp=fopen(outputfile,'w'); 
    fclose(temp);
    temp=fopen(wavefile,'w'); 
    fclose(temp);
end

%get the number of power traces collected in the previous session if any
position=fopen('lastposition.txt','r');
start=fscanf(position,'%d')+1;
fprintf('starting from %d\r\n',start);
fclose(position);

%open files
input=fopen(inputfile);
if(verifyen==1)
	verify=fopen(verifyfile);
end
output=fopen(outputfile,'a');
wave=fopen(wavefile,'a');

%set serial port. Change the baud rate if required to match the settings on your cryptographic device
s=serial(serialport);
s.InputBufferSize = 32;
s.Terminator='';
set(s,'BaudRate',9600);
fopen(s);
pause(3);

%just read and discard the plain text samples that which power traces are already captured for
for i=1:start-1
    pin=fgets(input);
    if(verifyen==1)
        pin=fgets(verify);
    end
end

%%carryout power collection
for i=start:last

	%read the plain text sample as well as the cipher text sample if verification is enabled 
    pin=strtrim(fgets(input));
	if(verifyen==1)
		cverify=strtrim(fgets(verify));
	end
    
	%pause(0.1);
    
	%send the plain text sample to the device and check whether the cryptosystem prints back correctly
	%useful for checking whether proper communication happens
	fprintf(s,'%s',pin);
    [pgot,b1]=fscanf(s,'%s');
    if(strcmp(pgot,pin)==0)
        fprintf('Plain text has changed. Expected %s. But in the pic %s\r\n',pin,pgot);
    end
    
	%pause(1);
    
	%aquire the power trace
    if (pwrcol==1)
        [pwr,time] = invoke(groupObj, 'readwaveform', 'channel1');
        fprintf('trace obtained. ');
		
		%for the first power trace acquire the the wave form of the rigger as well
		%then plot it and save them as well
		if (i==1)
			[trigger,time2]=invoke(groupObj, 'readwaveform', 'channel2');
			fig=figure;			
			plot(time,pwr);
			hold
			plot(time2,trigger,'r')
			filename=sprintf('firstwave');
			saveas(fig,filename,'fig');
			saveas(fig,filename,'jpg');
			close(fig);		
			trig=fopen('trigger.txt','w');
			fprintf(trig,'time : \r\n');
		    fprintf(trig,'%f ',time2);  
			fprintf(trig,'\r\ntrigger waveform : \r\n');			
			fprintf(trig,'%f ',trigger); 
			fclose(trig);	
		end
    end
    
	%inform the cryptographic device to stop the current encryption get ready for the next encryption
    fprintf(s,'%s','z');
    [cout,c1]=fscanf(s,'%s');
    %we should receive the cipher text from the cryptographic device at this point
	%if we did not receive properly send 'y to try a reset on the device
	if (b1<32 || c1<32)
        fprintf(s,'%s','y');
    end
    
	%if the plaintext it sent back at the beginning was wrong then something is wrong and going to a loop until success
    while(strcmp(pgot,pin)==0 || (verifyen==1 && (strcmp(cout,cverify)==0)))
            flushinput(s);
            flushoutput(s);
            fprintf(s,'%s',pin);
            [pgot,b1]=fscanf(s,'%s');
            if(strcmp(pgot,pin)==0)
                fprintf('Plain text has changed. Expected %s. But in the pic %s\r\n',pin,pgot);
            end  
            if(verifyen==1 && strcmp(cout,cverify)==0)
                fprintf('Cipher text has changed. Expected %s. But in the pic %s\r\n',cverify,cout);
            end         			
            %pause(1);
            if (pwrcol==1)
                [pwr,time] = invoke(groupObj, 'readwaveform', 'channel1');
            end
            fprintf(s,'%s','z');
            [cout,c1]=fscanf(s,'%s');
            if (b1<32 || c1<32)
                fprintf(s,'%s','y');
            end
    end
    
	%print the cipher text that received from the cryptographic device
    fprintf(output,'%s\r\n',cout);
    
    %print the power trace to file
    if (pwrcol==1)
        if (waveformat==0)
            fprintf(wave,'%f ',pwr);
            fprintf(wave,'\r\n');
        elseif (waveformat==1)
            fwrite(wave,pwr,'float');
        end
    end        
    
    position=fopen('lastposition.txt','w');
    fprintf(position,'%d',i);
    fclose(position);
    fprintf('Sample %d done\n',i);
    
end

%time value
timeval=toc

%get the successfully acquired number of traces
statout=fopen(stat,'w');

%print the stats about the power collection
fprintf(statout,'Sampling points in wave: %d\r\n',length(pwr));
fprintf(statout,'Number of traces: %d\r\n',i);
fprintf(statout,'Recorded time: %s \r\n',datestr(now, 'mm/dd/yy HH:MM:SS'));
fprintf(statout,'Time elapsed: %f \r\n',timeval);
fclose(statout);

%close everything
fclose(s);    
fclose(input);
if(verifyen==1)
    fclose(verify);
end
fclose(output);
fclose(wave);


