# CoGround: What Others Chose for Requests Like Yours

Code for the WWW 2027 short paper *What Others Chose for Requests Like Yours: Social Affordance for Conversational Recommendation*.

CoGround scores an item by **relational popularity**, i.e., how often a community chose the item for requests similar to the current one:

$$\mathrm{RP}(i\mid C) = \log \sum_{c'\in N(i)} \exp\big(s(C,c')/\tau\big)$$

Here $N(i)$ are the training episodes (request + chosen items) that chose $i$ and $s$ is the cosine between request embeddings (Qwen3-Embedding-8B). The score is added to a retrieval scorer, and the candidate set is the union of the top-$K$ items under both scores. No parameters are trained on recommendation labels apart from two ridge maps.

## Repository layout

```
coground/      the method: relational popularity, retrieval scorer, dynamic candidates (one entry script per dataset)
common/        shared loaders, ranking/evaluation helpers, bootstrap
experiments/   analyses in Section 4 (controls, kNN vote, requester-confirmed gold, personal history, Figure 2)
baselines/     data conversion and evaluation for ReDial, KBRD, KGSF, UniCRS, UCCR, MemoCRS†, zero-shot LLM
  runners/     example shell scripts that train the baselines with the settings in the paper
  patches/     our changes to CRSLab, UniCRS and UCCR (diffs, new files, example configs)
data/          Reddit-Movie preprocessing and request encoding
configure.sh   sets the base directory for all paths
```

### Names in the code and in the paper

| Code | Paper |
|---|---|
| `A1` | retrieval scorer $a_i$ (Eq. 1) |
| `c1`, `p_C . y_i` | match between the desired profile and the Tag Genome profile |
| `EP_emb` | relational popularity on request embeddings (Eq. 2) |
| `EP_prof`, `EP_exp`, `EP_attr` | relational popularity on tag profiles (all, experiential, property tags) |
| `union K100, tau .02, g 3` | CoGround final setting on Reddit-Movie |
| `fb_labels` | requester-confirmed gold |

## Setup

```bash
pip install -r requirements.txt
bash configure.sh /abs/path/to/workspace
```

Scripts read inputs and write outputs under `<workspace>/www27/` (`data/`, `results/`, `logs/`). A GPU with 24GB or more is enough for the method; the LLM baselines use vLLM.

### Data

| Dataset | Source | Preparation |
|---|---|---|
| Reddit-Movie (large) | release of He et al., *Large Language Models as Zero-Shot Conversational Recommenders* (CIKM 2023) | `python data/prep_reddit.py`, then `python data/enc_reddit.py` |
| ReDial, TG-ReDial, Inspired | CRSLab | downloaded by CRSLab; export samples with `python baselines/export_crslab_samples.py <dataset>` |
| Tag Genome 2021 | GroupLens | unzip to `<workspace>/datasets/tag_genome_2021/` |

## Reproducing the paper

Run the baselines first when a script compares against them (their per-sample rank dumps go to `results/dumps/`).

| Paper | Command | Output (`results/`) |
|---|---|---|
| Table 1, CoGround and retrieval scorer, Reddit-Movie | `python coground/fullcat_reddit_v2.py` | `fullcat_reddit_v2.json` |
| Table 1, HR/MRR/NDCG for all systems, Reddit-Movie | `python experiments/reddit_table_metrics.py` | `reddit_table_metrics.json` |
| Table 1, ReDial, TG-ReDial and Inspired | `python coground/fullcat_small.py redialorig` / `tgredial` / `inspired` | `fullcat_{redialorig,tgredial,inspired}.json` |
| Table 1, baselines | `bash baselines/runners/<script>.sh` (see below) | `dumps/*.jsonl` |
| Table 2, tag relations and candidate set | `python coground/fullcat_reddit_v2.py`, `python coground/fullcat_redial_v2.py` | `fullcat_{reddit,redial}_v2.json` |
| Table 2, shuffled control | `python experiments/shuffle_ep.py reddit` / `redial` | `shuffle_ep_*.json` |
| Table 2, hard vote of nearest requests | `python experiments/knn_baseline.py reddit` / `redial` | `knn_baseline_*.json` |
| kNN vote by similarity quartile | `python experiments/knn_quartile.py reddit` / `redial` | `knn_quartile_*.json` |
| Figure 2a, b and near-duplicate check | `python experiments/dup_check.py reddit` / `redial` | `dup_check_*.json` |
| Figure 2c and personal history by number of past threads | `python experiments/pers_bins_reddit.py` | `pers_bins_reddit.json` |
| Figure 2 (plot) | `python experiments/fig_analysis.py` | `figures/fig_analysis.pdf` |
| Requester-confirmed gold (labels) | `python experiments/fb_label_llm.py` | `fb_labels.json` |
| Requester-confirmed NDCG, near-duplicate removal, CIs | `python experiments/final_sys_checks.py` | `final_sys_checks.json` |
| Personal history weight $\lambda$ | `python common/pers_pop_reddit.py` | `pers_pop_reddit.json` |
| Own choice as a recommender (three seeds) | `python experiments/redial_selflabel_personal.py` | `redial_selflabel_personal.json` |

All settings ($\tau$, $g$, $K$, $\lambda$, baseline checkpoints) are selected on validation and test is evaluated once. Each script's docstring states its design and the date it was fixed.

### Baselines

| System | Script |
|---|---|
| ReDial, KBRD, KGSF on Reddit-Movie | `baselines/build_reddit_crslab.py`, then `runners/run_reddit_seeds_v2.sh`, `runners/run_kgsf_reddit_tune.sh`, `runners/run_redial_reddit_fix3.sh` |
| ReDial, KBRD, KGSF on CRSLab datasets | `runners/run_redial_all.sh`, `runners/run_tg_seeds.sh`, `runners/run_insp_seeds.sh` |
| UniCRS | `baselines/crslab2unicrs*.py`, then `runners/run_unicrs_reddit.sh`, `runners/run_unicrs_*_www27.sh`, `runners/run_unicrs_cl_sel.sh` |
| UCCR | `baselines/build_uccr_reddit.py`, `baselines/register_uccr_reddit.py`, `baselines/patch_uccr_sys.py`, then `runners/run_uccr_reddit.sh` |
| MemoCRS† | `baselines/memocrs_build_reddit.py` |
| Zero-shot LLM | `baselines/llm_zs_reddit.py`, `baselines/llm_zs_crslab.py` (`runners/run_llm_h100_v3.sh`) |
| Scoring of dumps | `baselines/reddit_crs_eval.py`, `baselines/reddit_top_eval.py`, `common/score_dumps.py`, `baselines/unicrs_score_*.py` |

Upstream code is used at these commits, with the changes in `baselines/patches/` applied:

| Repository | Commit | Changes |
|---|---|---|
| [CRSLab](https://github.com/RUCAIBox/CRSLab) | `6497938` | `crslab.diff` (per-sample rank dumps, seeds, ReDial output fix), `config/` (dataset configs), `new_files_crslab/` |
| [UniCRS](https://github.com/RUCAIBox/UniCRS) | `f7d8f95` | `unicrs.diff` (checkpoint selection on validation ranking, dumps), `unicrs_eval_cl.py` |
| UCCR | `ca567b3` | `uccr/redial.py` and `baselines/patch_uccr_sys.py` |

```bash
cd CRSLab && git checkout 6497938 && git apply /path/to/this/repo/baselines/patches/crslab/crslab.diff
cp -r /path/to/this/repo/baselines/patches/crslab/config/* config/
cp -r /path/to/this/repo/baselines/patches/crslab/new_files_crslab/* crslab/
```

## Notes

- Scripts were run on one H100 GPU. Several analysis scripts load the main script's feature build (`exec` of `coground/*.py`), so keep the folder layout.
- Baseline runners contain the GPU indices and thread counts we used; adjust them to your machine.
- Runners and configs are given for one seed; other seeds differ only in the seed value.
- Exploratory scripts that do not produce a reported number are not included.
