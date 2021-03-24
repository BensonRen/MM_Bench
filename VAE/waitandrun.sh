#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="/home/sr365/MM_Bench/:$PYTHONPATH"

# Waiting orders
#PIDA=714535
#PIDB=714632
#while [ -e /proc/$PIDA ] || [ -e /proc/$PIDB]
#do
#    echo "Process: $PID is still running" 
#        sleep 10m
#done

TIME=`date`
PWD=`pwd`
# The command to execute
COMMAND=evaluate.py
#COMMAND=predict.py
#COMMAND=train.py
#COMMAND=create_folder_modulized.py
#COMMAND= delete_after_BP_FF.py
#COMMAND plotsAnalysis.py
SPACE='        '
SECONDS=0
nohup python $COMMAND 1>output.out 2>error.err & 
echo $! > pidfile.txt

# Make sure it waits 10s for super fast programs
sleep 10s

PID=`cat pidfile.txt`
while [ -e /proc/$PID ]
do
    echo "Process: $PID is still running" 
        sleep 3m
done
#If the running time is less than 200 seconds (check every 180s), it must have been an error, abort
duration=$SECONDS
limit=2
if (( $duration < $limit )) 
then
    echo The program ends very shortly after its launch, probably it failed
    exit
fi

H=$(( $duration/3600 ))
M=$((( ($duration%3600 )) / 60 ))
S=$(( $duration%60 ))
#echo $H
#echo $M
#echo $S

CURRENTTIME=`date`
{
	echo To: rensimiao.ben@gmail.com
	echo From: Cerus Machine
	echo Subject: Your Job has finished!
	echo -e "Dear mighty Machine Learning researcher Ben, \n \n"
	echo -e  "    Your job has been finished and again, you saved so many fairies!!!\n \n"
	echo -e  "Details of your job:\n
        Job:  $COMMAND \n   
	PID:   `cat pidfile.txt` \n 
	TIME SPENT: $H hours $M minutes and $S seconds \n
        StartTime:   $TIME \n 
        ENDTIME: $CURRENTTIME \n
	PWD:  $PWD\n"
        cat parameters.py
} | ssmtp rensimiao.ben@gmail.com

echo "Process $PID has finished"

#Copying the parameters to the models folder as a record
#Lastfile=`ls -t models/ | head -1`
#mv parameters.txt models/$Lastfile/.
#cp parameters.py models/$Lastfile/.
#cp running.log models/$Lastfile/.
#cp running.err models/$Lastfile/.
