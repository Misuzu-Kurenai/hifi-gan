#! /bin/bash

wav_PT_22k_dir="no7singing/wav_PT_22k"
inferred_dataset_dir="mlp-singer/mel_inferred_dataset"
outputdir="koko2022_from_gen"

for filename in ${wav_PT_22k_dir}/*.wav; do
    base=$(basename $filename .wav)
    melfile=$inferred_dataset_dir/$base.npy
    python create_koko_dataset_from_gen.py $filename $melfile $outputdir
done




#python create_koko_dataset_from_gen.py no7singing/wav_PT_22k/01.wav mlp-singer/mel_inferred_dataset/01.npy koko2022_from_gen
