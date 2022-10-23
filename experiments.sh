#!/bin/bash

unzip good_instances.zip
mkdir solutions
python3 experiments.py

# for directory in Instances/Instances_*/
# do
#   for FILE in "$directory"*txt
#   do
# 	echo "solving $FILE"
#     prefix=$directory
#     suffix=".txt"
# 	string=$FILE
# 	no_instance=${string#"$prefix"}
# 	no_txt=${no_instance%"$suffix"}
#     python3 MLP.py "$FILE" $"solutions/${no_txt}-solution.txt" > notes
# 	rm notes
#   done
# done

