EmiGen-RL is a chemistry-aware generation framework coupled with reinforcement learning (RL) targeting multiple optical properties.

**Environment requirements:**

Python 3.7.3

Pytorch 1.10.0

RDKit 2020.03.2

We reccomend that the environment can be installed through anaconda.

For the usage of SubOptGragph as the property predictor, please refer to https://github.com/Jo3690/SOG to get all packages prepared.

**The pretraining of EmiGen-RL:**

First, preprocess the vocabulary files.
python get_vocab.py  < data.txt > vocab.txt

Then, get the graph data by using the following command
python preprocess.py --train data.txt --vocab vocab.txt --save_dir ./data/

Last, for the pretraining process:
python train_generator.py --train ./data/ --vocab vocab.txt --save_dir ./ckpt/

**The finetuning of the model:**

python transfer_train.py --train blue_mols.txt --vocab vocab.txt --save_dir ./finetune_ckpt/ --generative_model ./ckpt/model.ckpt

**The RL process:**

python RL_two.py --vocab vocab.txt --save_dir ./RL_dir/ --generative_model ./finetune_ckpt/finetune.ckpt

We provide a checkpoint file under the path of ./ckpt for the weight of pretrained model. You can also train your own model through the data we uploaded.

