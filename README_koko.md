# HiFI-GAN for no7singing dataset
This is a fork of HiFI-GAN for no7singing dataset (KOKO2022).
Scripts and commands are in preparation and needs to be fixed to work well.

## Setup docker
Use docker container in the mlp-singer

## Dataset preprocess
1. Create dataset using mlp-singer
2. Do preprocess with the following command:
```bash
bash do_create_koko2022_from_gen.sh
```

## Train HiFi-GAN
Train with the following command:
```bash
bash do_train_koko.sh
```
