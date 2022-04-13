#! /bin/sh
#source ~/tensorflow/bin/activate
NET_FILES=/homework/lv/Time analysis and Optimization Based on CNN accelerator/net/alexnet/alex.json
OUT_FILE=/homework/lv/Time analysis and Optimization Based on CNN accelerator/result/AlexNet/netData.txt


for NET_FILE in $NET_FILES 
  do

./paleo.sh profile  $NET_FILE \
    --direction=forward \
    --executor=tensorflow \
    >> $OUT_FILE

  echo '\n' >> $OUT_FILE
done
    exit
