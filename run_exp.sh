#!/usr/bin/env bash
echo "Arguments start: $@"
optimizer='fast_run'
run_folder=$(date +"run_%Y-%m-%d_%H:%M:%S")
export THEANO_FLAGS="device=gpu,optimizer=$optimizer,optimizer_including=cudnn,lib.cnmem=1.0,dnn.enabled=True"
echo "THEANO_FLAGS set to $THEANO_FLAGS"
echo "PWD: $(pwd)"

qsub -q itonb -d $(pwd) -N wavenet -u 310237018 <<< "#!/usr/bin/env bash
export THEANO_FLAGS=\"$THEANO_FLAGS\"
/home/310237018/.conda/envs/wavenet/bin/wpython -u wavenet.py with batch_run $@ 2>&1 | tee \"logs/$run_folder.log\"
"""

