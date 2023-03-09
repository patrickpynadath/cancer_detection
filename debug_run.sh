device=$1

if [[ $device == "cpu" ]]; then
  device_string="--device cpu --accelerator cpu"
elif [[ $device == "cuda" ]]; then
  device_string="--device cuda --accelerator gpu --devices 1"
fi


run_resnet(){
  echo "cur command"
  echo "python main.py train_resnet_clf --fast_dev_run True --criterion $1 --sample_strat $2 $device_string"
  python main.py train_resnet_clf --fast_dev_run True --criterion $1 --sample_strat $2 $device_string
}

run_transfer_learn_clf(){
  echo "cur command"
  echo "main.py train_transfer_learn_clf --fast_dev_run True --criterion $1 --sample_strat $2 --learning_mode $3 $device_string"
  python main.py train_transfer_learn_clf --fast_dev_run True --criterion $1 --sample_strat $2 --learning_mode $3 $device_string
}

run_train_ae(){
  echo "cur command"
  echo "python main.py train_transfer_learn_ae --fast_dev_run True --learning_mode $1 --res_type $2 $device_string"
  python main.py train_transfer_learn_ae --fast_dev_run True --learning_mode $1 --res_type $2 $device_string
}

# what do I need to test?
# train classifier resnet
# removed orig resnet arch -- issues with decoder
for LEARNMODE in "jigsaw" "normal"; do
  for RESTYPE in "custom"; do
    run_train_ae $LEARNMODE $RESTYPE
  done
  for CRITERION in "CE" "MSFE"; do
    for SAMPLESTRAT in "ros" "rus" "dynamic_ros" "dynamic_kmeans_ros"; do
    run_transfer_learn_clf $CRITERION $SAMPLESTRAT $LEARNMODE
    done
  done
done
for CRITERION in "CE" "MSFE"; do
  for SAMPLESTRAT in "ros" "rus"; do
    run_resnet $CRITERION $SAMPLESTRAT
  done
done

