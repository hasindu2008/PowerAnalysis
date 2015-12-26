#!/bin/bash
	
#	The A to Z of Building a Testbed for Power Analysis Attacks
#	Linux Script file to generate data for plotting the variation of correlation coefficient with number of power traces

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
# First data is obtained for min_traces number of power traces
# Then for min_traces+step, min_traces+2*step ... etc until max_traces 

# minimum number of power traces to test
min_traces=10

#maximum number of power traces to test
max_traces=200

#step value for number of traces
step_traces=10

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
sample_point_partition_size=100000


#remove temporary files from a previous session
test -e cpa.out && rm cpa.out
test -e cpatemp.cu && rm cpatemp.cu

#print used parameters
echo ""
echo "Parameters being used are as follows. If you did not change them open the script and edit them and run: "
echo "min_traces : $min_traces"
echo "max_traces : $max_traces"
echo "step_traces : $step_traces"
echo "power_trace_file : $power_trace_file"
echo "plain_text_file : $plain_text_file"
echo "power_trace_file_format : $power_trace_file_format"
echo "sample_points : $sample_points"
echo "sample_point_partition_size : $sample_point_partition_size"
echo ""
echo ""

num_of_traces=$min_traces
while [ $num_of_traces -le $max_traces ]
do
	echo "Doing for $num_of_traces of power traces" 
	
	#Generate a temporary script based on the provided parameters
	cp cpa.cu cpatemp.cu
	
	#change parameters in source
	sed -i "s/#define SAMPLES .*/#define SAMPLES $num_of_traces/" cpatemp.cu
	[ $power_trace_file_format = binary ] && sed -i "s/#define WAVFORMAT .*/#define WAVFORMAT 1/" cpatemp.cu
	[ $power_trace_file_format = ascii ] && sed -i "s/#define WAVFORMAT .*/#define WAVFORMAT 0/" cpatemp.cu
	sed -i "s/#define ALLWAVELEN .*/#define ALLWAVELEN $sample_points/" cpatemp.cu
	sed -i "s/#define WAVELENGTH .*/#define WAVELENGTH $sample_point_partition_size/" cpatemp.cu
	
	#compile using cuda. Please change sm_20 to match the appropriate compute capability of your graphics card
	nvcc cpatemp.cu helpers.cu -arch=sm_20 -o cpa.out 
	
	#run the CPA algorithm
	./cpa.out $power_trace_file $plain_text_file > "$num_of_traces".txt
	
	#rename results
	mv all_corr.txt all_"$num_of_traces".txt

	echo ""
	
	#increment the number of traces
	num_of_traces=$(echo $num_of_traces+$step_traces | bc)
	
done

