#!/bin/bash
	
#	The A to Z of Building a Testbed for Power Analysis Attacks
#	Linux Script file to run the correlation power analysis algorithm 

#    Authors : Hasindu Gamaarachchi, Harsha Ganegoda and Roshan Ragel, 
#    Department of Computer Engineering, 
#    Faculty of Engineering, University of Peradeniya, 22 Dec 2015
 
#    For more information read 
#    Hasindu Gamaarachchi, Harsha Ganegoda and Roshan Ragel, 
#    "The A to Z of Building a Testbed for Power Analysis Attacks", 
#    10th IEEE International Conference on Industrial and Information Systems 2015 (ICIIS)]
 
#    Any bugs, issues or suggestions please email to hasindu2008@live.com


#########################################################################################################

#There are 3 methods to run

# 1) ./script.sh
#		This will run the program with default arguments
#  2) ./script <num_of_traces> <result_file>
#		This will run the program with <num_of_traces> number of power traces with other arguments set to default
#  3) ./script.sh <num_of_traces> <power_trace_file> <plain_text_file> <result_file> <power_trace_file_format> <sample_points> <num_threads>
#		This will run with arguments as provided by you. FOr description of those arguments see below.
		

	
# The number of power traces to be used for calculation. If your input has more than num_of_traces, the first num_of_traces will be used for calulation	
num_of_traces=200

#File containing power traces
power_trace_file=wave.dat

#File containing plain text samples
plain_text_file=plain.txt

#file to output the result which is the most five correlated keys
result_file=results.txt

#input file format for the power trace file
#put ascii for ASCII format power traces. 
#put binary to save power traces in binary. 
power_trace_file_format=binary

#The number of sampling points in a single power trace
#This value can be found by inspecting the stat.txt generated after collecting a set of power traces*/
sample_points=100000

#The number of threads to launch*/
num_threads=32

clear

#if all arguments are provided
if	[ $# -eq 7 ]
then
	num_of_traces=$1
	power_trace_file=$2
	plain_text_file=$3
	result_file=$4
	power_trace_file_format=$5
	sample_points=$6
	num_threads=$7

#if only two arg (number of power traces and the result file) are given	
elif [ $# -eq 2 ]
then
	num_of_traces=$1
	result_file=$2
	echo ""
	echo "Only two arguments were entered."
	echo "Hence assigning them for number of samples and result file and assuming default values for other arguments"
	echo "If you do not like default args use as :  eg. ./script.sh <num_of_traces> <power_trace_file> <plain_text_file> <result_file> <power_trace_file_format> <sample_points> <num_threads>" 
	echo "Check the script file comments for description of these arguments"
	echo ""
	echo ""

#if arguments are given wrong	
else
	echo ""
	echo "Arguments were not properly entered"
	echo "Hence using default arguments"
	echo "If you do not like default args use as :  eg. ./script.sh <num_of_traces> <power_trace_file> <plain_text_file> <result_file> <power_trace_file_format> <sample_points> <num_threads>" 
	echo "Check the script file comments for description of these arguments"
	echo ""
	echo ""

fi


#remove temporary files from a previous session
test -e cpa.out && rm cpa.out
test -e cpatemp.c && rm cpatemp.c

#print used parameters
echo "Parameters being used are : "
echo "num_of_traces : $num_of_traces"
echo "power_trace_file : $power_trace_file"
echo "plain_text_file : $plain_text_file"
echo "result_file : $result_file"
echo "power_trace_file_format : $power_trace_file_format"
echo "sample_points : $sample_points"
echo "num_threads : $num_threads"


#Generate a temporary script based on the provided parameters
cp cpa.c cpatemp.c
sed -i "s/#define SAMPLES .*/#define SAMPLES $num_of_traces/" cpatemp.c
[ $power_trace_file_format = binary ] && sed -i "s/#define WAVFORMAT .*/#define WAVFORMAT 1/" cpatemp.c
[ $power_trace_file_format = ascii ] && sed -i "s/#define WAVFORMAT .*/#define WAVFORMAT 0/" cpatemp.c
sed -i "s/#define WAVELENGTH .*/#define WAVELENGTH $sample_points/" cpatemp.c
sed -i "s/#define THREADS .*/#define THREADS $num_threads/" cpatemp.c
	
#compile using c. Please change sm_20 to match the appropriate compute capability of your graphics card
gcc -Wall cpatemp.c helpers.c -o cpa.out -lpthread -lm

#run the CPA algorithm
./cpa.out $power_trace_file $plain_text_file > $result_file 


echo ""


