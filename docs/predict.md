# Prediction
Since our model is built based on [Boltz-1](https://github.com/jwohlwend/boltz), most of the parameters are the same as those of boltz prediction, which can be found at [Boltz-1 prediction](https://github.com/jwohlwend/boltz/blob/c53c9f7a86370a35026507a6264288755cf250c8/docs/prediction.md).

To run predictions with the model, use the following command:
```python
python ./scripts/predict.py --data <INPUT_PATH> --use_msa_server
```
where `<INPUT_PATH>` is a path to the input file or a directory. In our settings we only support YAML format as input. Passing the `--use_msa_server` flag will auto-generate the MSA using the mmseqs2 server, otherwise you can provide a precomputed MSA in CSV format like the file in `examples/msa`.

## YAML format
Regarding the filename of the yaml file, we recommend using the format `<ID>_<H chain ID>_<L chain ID>_<antigen chains ID>`, where the ID can be the corresponding PDB number, and the IDs of the light and heavy chains and antigen chains are the same as in the yaml file. For multiple antigen chains, just splice the chain IDs together as in `8euq_C_D_AB`; for nanobodies, leave the light chain IDs empty as in `8s0n_B__A`.

The schema of the YAML is the following(using `7vgs_D_C_A` as an example):
```yaml
version: 1
sequences:
  - protein:
      id: D
      sequence: EVQLQQSGPELVKPGASMKISCKTSXXXXXXXTMNWVKQSHGKNLEWIGLIXXXXXXTSYNQKFKGKATLTVDKSSSTAYMELLSLTSEDSAVYYCEVXXXXWGQGTLVTVSA
      spec_mask: '00000000000000000000000001111111000000000000000000011111100000000000000000000000000000000000000000111100000000000'
      ground_truth: EVQLQQSGPELVKPGASMKISCKTSGYSFTGYTMNWVKQSHGKNLEWIGLINPYNGDTSYNQKFKGKATLTVDKSSSTAYMELLSLTSEDSAVYYCEVINTYWGQGTLVTVSA
  - protein:
      id: C
      sequence: DIVMTQSPASLAVSLGQRATISCXXXXXXXXXXXXXXXWYQQKPGQPPKLLIYXXXXXXXGIPARFSGSGSGTDFTLNIHPVEEGDAATYYCXXXXXXXXXFGGGTKLEI
      spec_mask: '00000000000000000000000111111111111111000000000000000111111100000000000000000000000000000000111111111000000000'
      ground_truth: DIVMTQSPASLAVSLGQRATISCKASQSIDYDGDNYMNWYQQKPGQPPKLLIYTTSNLESGIPARFSGSGSGTDFTLNIHPVEEGDAATYYCQQNNEDPYTFGGGTKLEI
  - protein:
      id: A
      sequence: TVEELKKLLEQWNLVIGFLFLTWICLLQFAYANRNRFLYIIKLIFLWLLWPVTLACFVLAAVYRINWITGGIAIAMACLVGLMWLSYFIASFRLFARTRSMWSFNPETNILLNVPLHGTILTRPLLESELVIGAVILRGHLRIAGHHLGRCDIKDLPKEITVATSRTLSYYKLGASQRVAGDSGFAAYSRYRIGNY
      spec_mask: '0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000111111110000000000111111111100000111100000000000000000000001111111111011111111111100000000001111111111'
      ground_truth: TVEELKKLLEQWNLVIGFLFLTWICLLQFAYANRNRFLYIIKLIFLWLLWPVTLACFVLAAVYRINWITGGIAIAMACLVGLMWLSYFIASFRLFARTRSMWSFNPETNILLNVPLHGTILTRPLLESELVIGAVILRGHLRIAGHHLGRCDIKDLPKEITVATSRTLSYYKLGASQRVAGDSGFAAYSRYRIGNY
```
For each chain, this is represented by a `protein`, where `id` is the id number of the corresponding chain, which corresponds to the filename; `sequence` is the corresponding amino acid sequence, and for antibody chains, the CDR region is replaced by an `X`, which represents the region to be designed; `spec_mask` is a 01 string used to label the region, and the position of the CDR region is labeled as 1 for antibody chains, and the position of the antigen epitope is labeled as 1 for antigen chains; 

`ground_truth` is the real sequence of the corresponding chain, which is provided for calculating the MSA and the recovery rate of the generated results, and will not be inputted to the model, and the calculated MSA will go through a filter to filt out the sequences with high similarity to the CDR region, so there is no need to worry about data leakage problems. 

Note that `ground_truth` is optional, if `ground_truth` is omitted as an input, the MSA will be computed using the `sequence` containing the mask, and the recovery rate will not be computed in the generated result. However, it is highly recommended to provide `ground_truth` to compute a better quality MSA, which will lead to better quality results. 

**Remark**: We found that if the MSA is not filtered using a **0.2 similarity threshold** against the **original sequence**, the results will be slightly worse than those reported in the paper. This similarity threshold strongly affects the density and distribution of the input MSA (e.g., a 0.8 similarity threshold yields nearly twice as many MSA sequences as a 0.2 threshold). Therefore, changing this input will lead to a slight performance degradation due to the distribution shift.

## Options

| **Option**              | **Type**        | **Default**                 | **Description**                                                                                                                                                                      |
|-------------------------|-----------------|-----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--out_dir`             | `PATH`          | `./`                        | The path where to save the predictions.                                                                                                                                              |
| `--cache`               | `PATH`          | `./model`                  | The directory where to download the data and model.                                                                                                                                  |
| `--checkpoint`          | `PATH`          | `./model/stage_4.ckpt`                        | An optional checkpoint. Uses the provided model by default.                                                                                                                  |
| `--devices`             | `INTEGER`       | `1`                         | The number of devices to use for prediction.                                                                                                                                         |
| `--accelerator`         | `[gpu,cpu,tpu]` | `gpu`                       | The accelerator to use for prediction.                                                                                                                                               |
| `--recycling_steps`     | `INTEGER`       | `3`                         | The number of recycling steps to use for prediction.                                                                                                                                 |
| `--sampling_steps`      | `INTEGER`       | `200`                       | The number of sampling steps to use for prediction.                                                                                                                                  |
| `--diffusion_samples`   | `INTEGER`       | `5`                         | The number of diffusion samples to use for prediction.                                                                                                                               |
| `--step_scale`          | `FLOAT`         | `1.638`                     | The step size is related to the temperature at which the diffusion process samples the distribution. The lower the higher the diversity among structure samples (recommended between 1 and 2). |
| `--temperature`          | `FLOAT`         | `1.0`                     | The temperature parameters of sequence design. The lower the higher the diversity among sequences. |                                                                                                             
| `--num_workers`  | `INTEGER`       | `2`                         | The number of dataloader workers to use for prediction.                                                                                                                              |
| `--override`            | `FLAG`          | `False`                     | Whether to override existing predictions if found.                                                                                                                                   |
| `--use_msa_server`      | `FLAG`          | `False`                     | Whether to use the msa server to generate msa's.                                                                                                                                     |
| `--msa_server_url`      | str             | `https://api.colabfold.com` | MSA server url. Used only if --use_msa_server is set.                                                                                                                                |
| `--msa_pairing_strategy` | str             | `greedy`                    | Pairing strategy to use. Used only if --use_msa_server is set. Options are 'greedy' and 'complete'                                                                                   |
| `--only_process_msa` | `FLAG`             | `False`                    | Whether to only compute and process msa as `npz` file                                                                                   |
| `--msa_filtering_threshold` | `FLOAT`             | `0.2`                    | The threshold for the MSA filtering. Filter the sequences which has the similarity beyond the threshold in MSA.                                                                       |
| `--preprocessed_data_path`          | `PATH`          | `./data/summary.json`                        | Path to preprocessed data. Use this file to do MSA filtering.                                                                                                                  |
| `--msa_dir`          | `PATH`          | `None`                        | Path to MSA file in CSV format.                                                                                                                |
| `--processed_msa_dir`          | `PATH`          | `None`                        | Path to MSA file in NPZ format.                                                                                                                |
| `--structure_inpainting`          | `FLAG`          | `False`                        | Whether to do structure inpainting.                                                                                                               |
| `--ground_truth_structure_dir`          | `PATH`          | `./data/antibody_data/structures`                        | Structure data in NPZ format. Use this to do structure inpainting.                                                                                           |
| `--noise_type`      | `str`          | `discrete_absorb`                     | The noise type of sequence diffusion model, the type can be `discrete_absorb`, `discrete_uniform` or `continuous`, and it should match the checkpoint. |
| `--no_epitope`          | `FLAG`          | `False`                        | Not to use epitope information in input.                                                                                       |
| `--write_full_pae`      | `FLAG`          | `False`                     | Whether to save the full PAE matrix as a file.                                                                                                                                       |
| `--write_full_pde`      | `FLAG`          | `False`                     | Whether to save the full PDE matrix as a file.                                                                                                                                       |

The `summary.json` needs to be prepared to complete the MSA Filtering in the following format:
```json
{
    "9bu6_A_B_CD": {
        "pdb": "9bu6",
        "H_chain_id": "A",
        "L_chain_id": "B",
        "H_chain_seq": "QVQLVQSGAEVKKPGASVKVSCKASGYIFIDYYIHWVRQAPGQGLEWMGWINPNRGGTDYAQKFQGRVTMTSDTSIGTAFLELTRLKSDDTAVYYCARDRIWGGNWNPQKDDYGDRGGDYWGQGTLVTV",
        "L_chain_seq": "EIVLTQSPGTLSLSPGERATLSCRASQSFSSTYLAWYQHKPGQAPRLLIYGSSRRATGIPDRFSGSGSGTDFTLTISRLEPEDFAVYYCQQFGSSPRTFGQGTKLEV",
        "H_chain_masked_seq": "QVQLVQSGAEVKKPGASVKVSCKASXXXXXXXYIHWVRQAPGQGLEWMGWIXXXXXXTDYAQKFQGRVTMTSDTSIGTAFLELTRLKSDDTAVYYCARXXXXXXXXXXXXXXXXXXXXXXWGQGTLVTV",
        "L_chain_masked_seq": "EIVLTQSPGTLSLSPGERATLSCXXXXXXXXXXXXWYQHKPGQAPRLLIYXXXXXXXGIPDRFSGSGSGTDFTLTISRLEPEDFAVYYCXXXXXXXXXFGQGTKLEV",
        "antigen_chain_id": [
            "C",
            "D"
        ],
        "antigen_seq": {
            "C": "GLDKICLGHHAVANGTIVKTLTNEQEEVTNATETVESTGINRLCMKGRKHKDLGNCHPIGMLIGTPACDLHLTGMWDTLIERENAIAYCYPGATVNVEALRQKIMESGGINKISTGFTYGSSINSAGTTRACMRNGGNSFYAELKWLVSKSKGQNFPQTTNTYRNTDTAEHLIMWGIHHPSSTQEKNDLYGTQSLSISVGSSTYRNNFVPVVGARPQVNGQSGRIDFHWTLVQPGDNITFSHNGGLIAPSRVSKLIGRGLGIQSDAPIDNNCESKCFWRGGSINTRLPFQNLSPRTVGQCPKYVNRRSLMLATGMRNVPELI",
            "D": "LFGAIAGFLENGWEGMVDGWYGFRHQNAQGTGQAADYKSTQAAIDQITGKLNRLVEKTNTEFESIESEFSEIEHQIGNVINWTKDSITDIWTYQAELLVAMENQHTIDMADSEMLNLYERVRKQLRQNAEEDGKGCFEIYHACDDSCMESIRNNTYDHSQYREEALLNRLNI"
        },
        "antigen_type": "protein | protein",
        "resolution": 3.65,
        "scfv": false,
        "date": "11/27/24",
        "index_in_summary": 0
    },
    ...
}
```

It contains the basic information of the antibody-antigen complex. MSA Filtering will take the `H_chain_seq`,`L_chain_seq` as the real value to compare for filtering, and it will replace the first line of MSA with the masked sequence `H_chain_masked_seq`,`L_chain_masked_seq`.

In addition, if we want to do structure inpainting, we need to provide additional preprocessed ground truth structure data, which is in the same form as the structure data in the training data.

## Epitope Information Usage

When using the `--no_epitope` flag, the model will ignore epitope information during prediction. This means that you do not need to provide epitope annotations (marked as '1') in the `spec_mask` field for antigen chains in your YAML input file. The `spec_mask` for antigen chains can be set to all zeros ('0') or omitted entirely when this flag is enabled.

We provide models that have been trained both with and without epitope information. When using `--no_epitope`, the system will automatically use the model trained without epitope information, which is specifically designed to perform antibody design without requiring prior knowledge of the epitope regions on the antigen.

## Output
The output file is similar to Boltz-1. The difference is that there is an additional `<input_file>.seq` file in the results directory to represent the generated sequence, which has the following format:

```csv
Rank	Sequence	Total	H	L	H1	H2	H3	L1	L2	L3
0	EVQLQQSGPELVKPGASMKISCKTSEYSFTEYTMNWVKQSHGKNLEWIGLINPYNNDTSYNQKFKGKATLTVDKSSSTAYMELLSLTSEDSAVYYCEVNEAYWGQGTLVTVSADIVMTQSPASLAVSLGQRATISCKASQSIDYDRDNYMHWYQQKPGQPPKLLIYTTSNLESGIPARFSGSGSGTDFTLNIHPVEEGDAATYYCQQDKEDPYAFGGGTKLEI	0.771	0.647	0.839	0.714	0.833	0.250	0.867	1.000	0.667
1	EVQLQQSGPELVKPGASMKISCKTSEYSFTEYTMNWVKQSHGKNLEWIGLINPYNDDTSYNQKFKGKATLTVDKSSSTAYMELLSLTSEDSAVYYCEVKQAYWGQGTLVTVSADIVMTQSPASLAVSLGQRATISCKASQSIDYDRDNYMHWYQQKPGQPPKLLIYTTSNLDSGIPARFSGSGSGTDFTLNIHPVEEGDAATYYCHQNNEDPYAFGGGTKLEI	0.771	0.647	0.839	0.714	0.833	0.250	0.867	0.857	0.778
```

where `Rank` and `Sequence` are the generated sequences and their corresponding numbers. If `ground_truth` is provided as input, the corresponding recovery rate is given later. For nanobodies the results `L1`,`L2`,`L3` will not be provided and the value of `L` will be the same as `H`.
