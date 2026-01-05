# accel_sensor → displacement_pattern_by_subject 生成フロー（最初から最後まで）

対象: 加速度センサ（`accel_sensor`）を含む実験ログから、最終的に `displacement_pattern_by_subject_*.png` と `displacement_pattern_summary.csv` を出力するまで。

このリポジトリでは「振幅（変位の大きさ）」と「位相（窓位相相関）」は別工程で計算し、最後に `displacement_pattern_analyzer.py` で統合して **パターン別の平均振幅（cm）** と **時間割合（%）** を出します。

---

## 前提（入力データの配置）

各セッション（タイムスタンプ `YYYYMMDD_HHMMSS`）が入ったフォルダに、最低限以下が存在する前提です。

- `*_experiment_log.csv`
- `*_random_dot*_trial_1.csv`（例: `*_random_dot_data_trial_1.csv`）
- `*_accel_*trial_1.csv`（例: `accel_log_trial_1.csv`, `accel_log_serial_trial_1.csv`, `accel_sensor_trial_1.csv`, `YYYYMMDD_HHMMSS_accel_log...trial_1.csv`）

条件により追加で:
- GVS系: `*_dac_output_red.csv` / `*_dac_output_green.csv`
- Audio系: `*_audio...csv` または `audio_red_integrated_analysis.csv` 等

フォルダ名（例: `vis/`, `audio/`, `gvs/`, `only_audio/`, `only_gvs/`）は `analyze_datas.py` が刺激タイプ推定に使います。

---

## 全体の処理順（End-to-End）

最短の実行順は以下です（フォルダを上位から指定すれば再帰処理されます）。

1) 統合データ生成: `analyze_datas.py`
2) 身体動揺（3 Hz LPF）生成: `postural_sway_analyzer.py`
3) 頭部変位（cm）生成: `head_displacement_analyzer.py --sway`
4) 窓位相相関（変位×視覚）生成: `displacement_visualization_extractor.py`
5) パターン別振幅・割合の集計/描画: `displacement_pattern_analyzer.py`

例（被験者Bを一括処理）:

```bash
python analyze_datas.py B/
python postural_sway_analyzer.py B/
python head_displacement_analyzer.py B/ --sway
python displacement_visualization_extractor.py B/ 3 --output B/
python displacement_pattern_analyzer.py B/ --output B_results
```

---

## 推奨ディレクトリ構成例（A/B/C = 被験者）

このプロジェクトでは、被験者ごとにトップフォルダを分け（例: `A/`, `B/`, `C/`）、その配下に刺激条件フォルダを置く構成が扱いやすいです。

```text
A/
  vis/
    2025...._experiment_log.csv
    2025...._random_dot_data_trial_1.csv
    2025...._accel_log_trial_1.csv
  audio/
    2025...._experiment_log.csv
    2025...._random_dot_data_trial_1.csv
    2025...._accel_log_trial_1.csv
    2025...._audio_....csv  (または audio_*_integrated_analysis.csv)
  gvs/
    2025...._experiment_log.csv
    2025...._random_dot_data_trial_1.csv
    2025...._accel_log_trial_1.csv
    2025...._dac_output_red.csv / _dac_output_green.csv
  only_audio/
    (audioのみのセッション一式)
  only_gvs/
    (gvsのみのセッション一式)
```

各刺激フォルダ配下には、解析が進むにつれて以下の生成物が増えます（セッションごと）。

- `*_integrated_analysis.csv`（(1)）
- `*_integrated_analysis_sway_3.0Hz.csv`（(2)）
- `*_sway_3.0Hz_head_displacement.csv`（(3)）
- `displacement_phase/filtered_signals/*.csv|png`（(4)）
- `displacement_phase/correlation_visualizations/*.csv|png`（(4)）

最終的な集計物（`*_results/`）は、被験者フォルダの外側に置くのがおすすめです（例: `A_results/`, `B_results/`, `C_results/`）。

---

## 振幅（amplitude）系：入力/出力とフロー

### 目的
- 加速度由来の姿勢角（roll/pitch）から **頭部変位（cm）** を計算し、
- `displacement_pattern_analyzer.py` で **パターン別の平均振幅（cm）** を算出する。

### フロー

```text
accel_sensor + random_dot + (audio/gvs) + experiment_log
  └─(1) analyze_datas.py
      └─ *_integrated_analysis.csv
           └─(2) postural_sway_analyzer.py (3Hz LPF)
               └─ *_integrated_analysis_sway_3.0Hz.csv
                    └─(3) head_displacement_analyzer.py --sway
                        └─ *_sway_3.0Hz_head_displacement.csv
                             └─(5) displacement_pattern_analyzer.py
                                 └─ displacement_pattern_* (CSV/PNG)
```

### (1) `analyze_datas.py`（統合データ生成）
**入力**（セッションフォルダ内）
- `*_accel_*trial_1.csv`（加速度）
- `*_random_dot*_trial_1.csv`（ランダムドット）
- `*_experiment_log.csv`（条件: red/green 等）
- （条件で）`*_dac_output_*.csv`, `*_audio*.csv`

**出力**（同フォルダ）
- `YYYYMMDD_HHMMSS_integrated_analysis.csv`
- `YYYYMMDD_HHMMSS_integrated_analysis.png`（可視化）

**統合CSVの主要列（例）**
- `psychopy_time`
- `accel_x`, `accel_y`, `accel_z`
- `roll`, `pitch`, `angle_change`, `roll_change`, `pitch_change`
- `red_dot_mean_x`, `red_dot_mean_y`, `green_dot_mean_x`, `green_dot_mean_y`
- `red_dot_x_change`, `green_dot_x_change`
- `gvs_dac_output`（GVS系フォルダ）

### (2) `postural_sway_analyzer.py`（身体動揺抽出）
**入力**
- `*_integrated_analysis.csv`

**処理**
- 3 Hz ローパスで身体動揺成分（`*_sway`列）を作る
- データ切り出し（デフォルト 20s〜300s）が入る

**出力**
- `YYYYMMDD_HHMMSS_integrated_analysis_sway_3.0Hz.csv`
- `correlation_summary_3.0Hz.csv`（フォルダ単位の要約）

**sway CSVの主要列（例）**
- `psychopy_time`
- `accel_x_sway`, `accel_y_sway`, `accel_z_sway`
- `roll_sway`, `pitch_sway`, `roll_change_sway`, `pitch_change_sway`, `angle_change_sway`
- `red_dot_x_change_sway`, `green_dot_x_change_sway`
- `gvs_dac_output_sway`

### (3) `head_displacement_analyzer.py --sway`（頭部変位計算）
**入力**
- `*_integrated_analysis_sway_3.0Hz.csv`（`--sway`時は `_sway_` ファイルのみ対象）

**処理**
- 倒立振子モデルで変位換算（角度→長さ×sin）

**出力**
- `YYYYMMDD_HHMMSS_sway_3.0Hz_head_displacement.csv`
- `*_head_displacement_timeseries.png`
- `*_head_trajectory_2d.png`
- `*_head_density_2d.png`

**head_displacement CSVの列（例）**
- `psychopy_time`
- `roll_sway`, `pitch_sway`, `roll_change_sway`, `pitch_change_sway`
- `displacement_x_cm`, `displacement_y_cm`
- `displacement_x_relative_cm`, `displacement_y_relative_cm`

### (5) `displacement_pattern_analyzer.py`（パターン別振幅/割合の集計）
**入力**
- `displacement_phase/correlation_visualizations/*_displacement_window_correlations_*.csv`（位相相関、後述）
- `*_sway_3.0Hz_head_displacement.csv`（上で生成）

**処理（振幅側）**
- 位相相関パターンごとの区間に対して `displacement_x_cm` の平均振幅（cm）を計算

**出力（指定 `--output` ディレクトリ）**
- `displacement_pattern_detailed.csv`（セッション別）
- `displacement_pattern_summary.csv`（条件集約）
- `displacement_pattern_by_subject_<SUBJECT>.png`（被験者別）
- `displacement_pattern_percentage_stacked_<SUBJECT>.png`（積み上げ割合）
- `displacement_pattern_by_condition.png`
- `displacement_pattern_sample_counts.png`

`displacement_pattern_detailed.csv` 列:
- `subject`, `stimulus_type`, `color_condition`, `session_id`, `relative_path`, ...
- `red_dominant_mean/std/samples`, `green_dominant_*`, `both_high_*`, `both_low_*`

`displacement_pattern_summary.csv` 列:
- `subject`, `stimulus_type`, `color_condition`
- `*_mean_mean`, `*_mean_std`, `*_samples_sum`, `total_valid_samples_sum`

---

## 位相（phase）系：入力/出力とフロー

### 目的
- `displacement_x_cm` と視覚刺激（赤/緑フロー）の信号から瞬時位相を求め、
- 窓（例: 10s）ごとの位相相関を計算し、
- `displacement_pattern_analyzer.py` で **相関パターン（赤のみ高/緑のみ高/両方高/両方低）** を決める。

### フロー

```text
*_integrated_analysis_sway_3.0Hz.csv  +  *_sway_3.0Hz_head_displacement.csv
  └─(4) displacement_visualization_extractor.py
      ├─ displacement_phase/filtered_signals/*_displacement_filtered_signals_3.0Hz.csv
      └─ displacement_phase/correlation_visualizations/*_displacement_window_correlations_3.0Hz.csv
           └─(5) displacement_pattern_analyzer.py でパターン分類に使用
```

### (4) `displacement_visualization_extractor.py`（変位×視覚の窓位相相関）
**入力**
- `*_integrated_analysis_sway_3.0Hz.csv`
- `*_sway_3.0Hz_head_displacement.csv`

**処理（位相側）**
- 3 Hz LPF済み信号を正規化
- ヒルベルト変換で瞬時位相を計算
- 窓位相相関を算出

**出力**（各セッションフォルダ内に `displacement_phase/` を作成）
- `displacement_phase/filtered_signals/`
  - `*_displacement_filtered_signals_3.0Hz.png`
  - `*_displacement_filtered_signals_3.0Hz.csv`
  - `*_displacement_normalization_info_3.0Hz.csv`
- `displacement_phase/correlation_visualizations/`
  - `*_displacement_window_correlations_3.0Hz.png`
  - `*_displacement_window_correlations_3.0Hz.csv`
  - `*_displacement_window_correlations_normalization_info_3.0Hz.csv`

`*_displacement_filtered_signals_3.0Hz.csv` 列:
- `psychopy_time`
- `displacement_x_filtered`
- `red_dot_x_change_filtered`
- `green_dot_x_change_filtered`

`*_displacement_window_correlations_3.0Hz.csv` 列:
- `psychopy_time`
- `phase_correlation_displacement_red_dot`
- `phase_correlation_displacement_green_dot`

---

## `displacement_pattern_analyzer.py` のパターン定義（位相相関→分類）

閾値 `CORRELATION_THRESHOLD = 0.5` を用いて排他的に分類します。

- **赤のみ高い**: red ≥ 0.5 かつ green < 0.5
- **緑のみ高い**: green ≥ 0.5 かつ red < 0.5
- **両方高い**: red ≥ 0.5 かつ green ≥ 0.5
- **両方低相関**: red < 0.5 かつ green < 0.5

分類した区間ごとに、(a) サンプル数（時間割合%へ換算）と (b) `displacement_x_cm` の平均振幅（cm）を計算します。

---

## よくある詰まりポイント（only_gvs が出ない等）

- `displacement_visualization_extractor.py` は **head_displacement を生成しません**。
  - 先に `head_displacement_analyzer.py --sway` を実行して `*_head_displacement.csv` を作る必要があります。
- `displacement_pattern_analyzer.py` は `*_displacement_window_correlations_*.csv` と対応する `*_head_displacement.csv` が両方揃わないと、そのセッションをスキップします。

---

## 最終成果物（displacement_pattern_by_subject）

`python displacement_pattern_analyzer.py <被験者フォルダ> --output <出力フォルダ>` の出力:
- `displacement_pattern_by_subject_<SUBJECT>.png`
  - 棒の中: **時間割合（%）**
  - 棒の上: **平均振幅（cm）**
- `displacement_pattern_summary.csv` / `displacement_pattern_detailed.csv`

---

## 実行例（A/B/Cを最初から最後まで一括）

被験者フォルダごとに同じコマンド列を実行します。

```bash
# A
python analyze_datas.py A/
python postural_sway_analyzer.py A/
python head_displacement_analyzer.py A/ --sway
python displacement_visualization_extractor.py A/ 3 --output A/
python displacement_pattern_analyzer.py A/ --output A_results

# B
python analyze_datas.py B/
python postural_sway_analyzer.py B/
python head_displacement_analyzer.py B/ --sway
python displacement_visualization_extractor.py B/ 3 --output B/
python displacement_pattern_analyzer.py B/ --output B_results

# C
python analyze_datas.py C/
python postural_sway_analyzer.py C/
python head_displacement_analyzer.py C/ --sway
python displacement_visualization_extractor.py C/ 3 --output C/
python displacement_pattern_analyzer.py C/ --output C_results
```

---

## 全被験者平均（A/B/C → 1つの平均結果）

被験者別の `*_results/displacement_pattern_summary.csv` が揃ったら、平均化は次で実行します。

```bash
python displacement_pattern_average_analyzer.py A/ B/ C/ --output result_averaged
```

出力先 `result_averaged/` に、被験者平均のサマリCSVと図（平均版）が生成されます。

