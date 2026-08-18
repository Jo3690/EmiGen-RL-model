EmiGen-RL is a chemistry-informed generation framework coupled with reinforcement learning (RL) targeting multiple properties for complex funcational molecules.

**Description:**

- EmiGen-RL encompasses three levels  within a variational autoencoder (VAE) and progressively constructs molecules by coordinating global structural organization with fine-grained variability.
- A dataset comprising 133,670 organic luminescent molecules (OLMs) is established as a test case for EmiGen-RL.
- Against five generative models with different representation strategies, EmiGen-RL achieves the best overall performance not only on validity and plausibility, but also on uniqueness and novelty.
- Reinforcement learning is integrated for the multi-property-guided generation. Here, as an example, SubOptGraph acts as the property predictor, aiming at generating deep-blue molecules with high efficiency.
- The effectiveness and potential are validated by the rediscovery of several experimentally reported deep-blue molecules with high efficiency and two novel synthesized deep-blue moelcules with high photoluminescence quantum yield (PLQY).
- It can be extended to the fields of other complex functional molecules/materials and updated with the development of data, predictors, and so on.


**Environment requirements:**

Main frameworks/packages

Python 3.7.3

Pytorch 1.10.0

RDKit 2020.03.2

We recommend that the above packages and relevant dependencies can be installed through anaconda.

For the settings of SubOptGraph as the optical property predictor, please refer to https://github.com/Jo3690/SOG to get all packages prepared. We have offered the weights of the predictors for emission peaks and PLQYs here in Model_paras/.

**The pretraining of EmiGen-RL:**

First, preprocess the vocabulary files to decompose the molecular graphs to get the substructures and connectivity points.

python get_vocab.py  < data.txt > vocab.txt

Then, get the graph data by using the following command, ready for the (pre)training of the model.

python preprocess.py --train data.txt --vocab vocab.txt --save_dir ./data/

Last, for the (pre)training process:

python train_generator.py --train ./data/ --vocab vocab.txt --save_dir ./ckpt/

**The finetuning of the model:**

For the generation of the complex funcational molecules from specific domain (taking blue-emitting as an example here)

python transfer_train.py --train blue_mols.txt --vocab vocab.txt --save_dir ./finetune_ckpt/ --generative_model ./ckpt/model.ckpt

**The RL process:**

For the multi-property-guided generation, we use reinforcement learning, speficically, proximal policy optimization (PPO), to generate desired complex funcation molecules (taking deep-blue molecules with high PLQY as a case):

python RL_two.py --vocab vocab.txt --save_dir ./RL_dir/ --generative_model ./finetune_ckpt/finetune.ckpt

We provide a checkpoint file under the path of ./ckpt for the weight of pretrained model. You can also train your own model through the data we uploaded or do any modifications as you like.

