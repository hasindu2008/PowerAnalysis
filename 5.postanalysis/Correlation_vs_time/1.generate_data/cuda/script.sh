#!/bin/bash
	
#	The A to Z of Building a Testbed for Power Analysis Attacks
#	Linux Script file to generate data for plotting the variation of correlation coefficient with number of power traces

#	This is used to generate data for plotting the variation of correlation coefficient with time
#	Saves the all correlation coefficients for all key possibilities along time
#	This will generate 4096 files under names subbyte<x>_keyguess<y> where <x> and <y> are integers
	
#	This program needs open 4096 files at the same time.
#	For that you may need log as root using "sudo su"
#	then increase the maximum simultaneously opened file limit by "ulimit -n 5000"


#    Authors : Hasindu Gamaarachchi, Harsha Ganegoda and Roshan Ragel, 
#    Department of Computer Engineering, 
#    Faculty of Engineering, University of Peradeniya, 22 Dec 2015
 
#    For more information read 
#    Hasindu Gamaarachchi, Harsha Ganegoda and Roshan Ragel, 
#    "The A to Z of Building a Testbed for Power Analysis Attacks", 
#    10th IEEE International Conference on Industrial and Information Systems 2015 (ICIIS)]
 
#    Any bugs, issues or suggestions please email to hasindu2008@live.com

#########################################################################################################

# Run as ./script. Change the following arguments appropriately

	
# The number of power traces to be used for calculation. If your input has more than num_of_traces, the first num_of_traces will be used for calulation	
num_of_traces=200

#File containing power traces
power_trace_file=wave.dat

#File containing plain text samples
plain_text_file=plain.txt


#input file format for the power trace file
#put ascii for ASCII format power traces. 
#put binary to save power traces in binary. 
power_trace_file_format=binary

#The number of sampling points in a single power trace
#This value can be found by inspecting the stat.txt generated after collecting a set of power traces*/
sample_points=100000

#Global memory on a GPU is limited (RAM as well) and hence if the power traces are large sized, all the things won't fit at once to memory
#If you experience a memory allocation failure when running, reduce WAVELENGTH value
#This will force the program to read the power traces part by part 
#But when using this implementation make sure that ALLWAVELENGTH is divisible by WAVELENGTH 
sample_point_partition_size=50000

clear

#remove temporary files from a previous session
test -e cpa.out && rm cpa.out
test -e cpatemp.cu && rm cpatemp.cu

#print used parameters
echo "Parameters being used are : "
echo "num_of_traces : $num_of_traces"
echo "power_trace_file : $power_trace_file"
echo "plain_text_file : $plain_text_file"
echo "power_trace_file_format : $power_trace_file_format"
echo "sample_points : $sample_points"
echo "sample_point_partition_size : $sample_point_partition_size"


#Generate a temporary script based on the provided parameters
cp cpa.cu cpatemp.cu
sed -i "s/#define SAMPLES .*/#define SAMPLES $num_of_traces/" cpatemp.cu
[ $power_trace_file_format = binary ] && sed -i "s/#define WAVFORMAT .*/#define WAVFORMAT 1/" cpatemp.cu
[ $power_trace_file_format = ascii ] && sed -i "s/#define WAVFORMAT .*/#define WAVFORMAT 0/" cpatemp.cu
sed -i "s/#define ALLWAVELEN .*/#define ALLWAVELEN $sample_points/" cpatemp.cu
sed -i "s/#define WAVELENGTH .*/#define WAVELENGTH $sample_point_partition_size/" cpatemp.cu
	
#compile using cuda. Please change sm_20 to match the appropriate compute capability of your graphics card
nvcc cpatemp.cu helpers.cu -arch=sm_20 -o cpa.out 

#run the CPA algorithm
./cpa.out $power_trace_file $plain_text_file > result.txt


echo ""


