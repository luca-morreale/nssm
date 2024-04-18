#! /bin/bash

if [ $# -lt 1 ]
then
    echo "Please specify a folder!"
    exit 1
fi

BASE_FOLDER=$1

LAST="${BASE_FOLDER:0-1}"

if [[ $LAST != '/' ]]
then
    BASE_FOLDER="${BASE_FOLDER}/"
fi


for FOLDER in `ls -d ${BASE_FOLDER}*`
do

    echo "Generate config files"
    python -m scripts.make_configs --folder ${FOLDER} ; echo

    echo "Normalize input meshes"
    python -m scripts.normalize_meshes --folder ${FOLDER} ; echo

    echo "Running Alignment"
    python -m scripts.align_meshes --folder ${FOLDER} --num_rotations 12 --save_imgs ; echo
    echo "Running Matches"
    python -m scripts.compute_fuzzy_matches --folder ${FOLDER} --num_rotations 20 --save_imgs --mode bidirectional ; echo

    echo "Extracting cones"
    python -m scripts.extract_cones --verbose --folder ${FOLDER} --topk 3 --recompute  ; echo
    echo "Cutting meshes"
    python -m scripts.process_genus0_pair --verbose --folder ${FOLDER} ; echo

    cd parametrization_cpp/
    echo "Computing parametrization with cpp"
    ./build/slim_bnd ${FOLDER}/samples/source.obj
    ./build/slim_bnd ${FOLDER}/samples/target.obj

    cd ../
    echo "Updating parametrization"
    python -m scripts.update_parametrization --verbose --pth ${FOLDER}/samples/source.pth --new ${FOLDER}/samples/source_slim.obj ; echo
    python -m scripts.update_parametrization --verbose --pth ${FOLDER}/samples/target.pth --new ${FOLDER}/samples/target_slim.obj ; echo

    echo "Overfit source"
    python -m mains.training ${FOLDER}/configs/source.json ; echo
    echo "Overfit target"
    python -m mains.training ${FOLDER}/configs/target.json ; echo
    echo "Optimizing Map"
    python -m mains.training ${FOLDER}/configs/map.json ; echo

done

