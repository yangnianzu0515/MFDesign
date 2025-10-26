# Training
Since our model is built based on [Boltz-1](https://github.com/jwohlwend/boltz), most of the parameters are the same as those of boltz training, which can be found at [Boltz-1 training](https://github.com/jwohlwend/boltz/blob/c53c9f7a86370a35026507a6264288755cf250c8/docs/training.md). You can use the same training script as Boltz-1 like follows:
```bash
python scripts/train/train.py scripts/train/configs/stage_1.yaml
```
We provide all configuration files for the 4 stages described in our paper.

## Modify the configuration file
Compared to Boltz-1, we made some changes in the configuration file by adding some proprietary parameters.

```yaml
data:
  datasets:
    - _target_: boltz.data.module.training.DatasetConfig
      target_dir: ./antibody_data
      msa_dir: ./msa
      prob: 1.0
      sampler:
        _target_: boltz.data.sample.antibody.AntibodySampler
      cropper:
        _target_: boltz.data.crop.antibody.AntibodyCropper
        add_antigen: false
        min_neighborhood: 0
        max_neighborhood: 40
  distinguish_epitope: true
```

For the data module, we replaced both the `Sampler` and `Cropper` of the dataset with antibody-specific designs, consistent with the paper.The `add_antigen` of the `Cropper` is used to determine whether or not to consider retaining the antigen residue. In addition to this,  the `distinguish_epitope` parameter is used to determine whether to introduce antigenic epitope information. 
For specific data, `target_dir` is the preprocessed structure file and `msa_dir` is the preprocessed msa file, both in NPZ file format.

```yaml
model:
  score_model_args:
    sequence_model_args:
      hidden_dim: 768
      vocab_size: 20
      dropout: 0.1
  structure_prediction_training: true
  sequence_prediction_training: true
  confidence_prediction: false
  diffusion_process_args:
    noise_type: discrete_absorb
```

For the model module, we added the sequence diffusion module to `score_model` with the parameters shown in `sequence_model_args`. At the same time, we divide the whole module into three parts `structure`, `sequence`, `confidence`, and use the corresponding `structure_prediction_training`, `sequence_prediction_training` and `confidence_ prediction` parameters to represent which module parameters are involved in the training, only `structure` and `sequence` modules will always be trained in our setting. In the `diffusion_process_args` module we added `noise_type` to control the noise type, which is `discrete_absorb`, `discrete_uniform` and `continuous`.

We provide yaml files for all four stages. They differ in parameters mainly in `max_tokens`, `batch_size` and `Cropper`. In particular, the `Cropper` used in the fourth stage is `MixCropper`, which is a mix of Boltz-1's own `Cropper` and our `AntibodyCropper`, with the `probability` parameter representing the probability of selecting the `AntibodyCropper`.