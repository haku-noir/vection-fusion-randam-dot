# 📋 6. CSVファイル仕様

## 6.1 実験出力ファイル

### experiment_log.csv

実験の設定・条件を記録するログファイル。

| カラム名 | 型 | 説明 |
|---------|---|------|
| `trial` | int | 試行番号 |
| `panning_mode` | str | 音響パンニング方式（volume/itd/both） |
| `scrolling_mode` | bool | スクロールモードON/OFF |
| `condition` | str | 実験条件（red/green） |
| `response` | str | 被験者応答 |
| `RT` | float | 反応時間 |
| `stimulus_start_time` | str | 刺激開始時刻 |
| `stimulus_start_timestamp` | str | タイムスタンプ |
| `single_color_dot` | bool | 単色ドットモード |
| `visual_reverse` | bool | 視覚刺激反転フラグ |
| `audio_reverse` | bool | 音響刺激反転フラグ |
| `gvs_reverse` | bool | GVS刺激反転フラグ |
| `audio_source_mode` | str | 音源モード（simulation/mp3） |
| `audio_sync_type` | str | 音響同期タイプ |
| `audio_file_used` | str | 使用音源ファイル |
| `win_width` | int | 画面幅 [px] |
| `win_height` | int | 画面高さ [px] |
| `n_dots` | int | ドット数 |
| `dot_size` | int | ドットサイズ [px] |
| `fall_speed` | int | 落下速度 [px/s] |
| `dot_osc_freq` | float | 横揺れ周波数 [Hz] |
| `dot_osc_amp` | int | 横揺れ振幅 [px] |

---

### random_dot_trial_1.csv

ランダムドットの位置データ（フレームごと）。

| カラム名 | 型 | 単位 | 説明 |
|---------|---|------|------|
| `psychopy_time` | float | 秒 | PsychoPy時間 |
| `red_dot_mean_x` | float | px | 赤ドット群の平均X座標 |
| `red_dot_mean_y` | float | px | 赤ドット群の平均Y座標 |
| `green_dot_mean_x` | float | px | 緑ドット群の平均X座標 |
| `green_dot_mean_y` | float | px | 緑ドット群の平均Y座標 |

- **サンプリング**: ~60Hz（フレームレート依存）
- **座標系**: 画面中心が原点

---

### accel_log_serial_trial_1.csv

加速度センサとドット位置の同期データ。

| カラム名 | 型 | 単位 | 説明 |
|---------|---|------|------|
| `psychopy_time` | float | 秒 | PsychoPy時間 |
| `accel_time` | float | 秒 | 加速度センサ時間 |
| `accel_x` | float | m/s² | X軸加速度 |
| `accel_y` | float | m/s² | Y軸加速度 |
| `accel_z` | float | m/s² | Z軸加速度 |
| `red_dot_mean_x` | float | px | 赤ドットX座標 |
| `red_dot_mean_y` | float | px | 赤ドットY座標 |
| `green_dot_mean_x` | float | px | 緑ドットX座標 |
| `green_dot_mean_y` | float | px | 緑ドットY座標 |

- **サンプリング**: ~120Hz（M5Stack依存）

---

### audio_trial_1.csv

音響パンニングデータ。

| カラム名 | 型 | 単位 | 説明 |
|---------|---|------|------|
| `psychopy_time` | float | 秒 | PsychoPy時間 |
| `angle_change` | float | 度 | パンニング角度 |

---

## 6.2 共通データファイル

### dac_output_red.csv / dac_output_green.csv

GVS出力パターン（事前生成）。

| カラム名 | 型 | 単位 | 説明 |
|---------|---|------|------|
| `time_sec` | float | 秒 | 時間 |
| `sine_value_internal` | float | - | 内部正弦波値 |
| `dac25_output` | int | - | DAC PIN25出力（+方向） |
| `dac26_output` | int | - | DAC PIN26出力（-方向） |

- **サンプリング**: ~1000Hz
- **GVS出力計算**: `gvs_output = dac25_output - dac26_output`

---

### audio_red_integrated_analysis.csv / audio_green_integrated_analysis.csv

音響パターン（解析用参照データ）。

| カラム名 | 型 | 単位 | 説明 |
|---------|---|------|------|
| `psychopy_time` | float | 秒 | 時間 |
| `accel_x` | float | m/s² | X軸加速度（ダミー） |
| `accel_y` | float | m/s² | Y軸加速度（ダミー） |
| `accel_z` | float | m/s² | Z軸加速度（ダミー） |
| `angle_change` | float | 度 | 角度変化（＝パンニング角度） |
| `roll` | float | 度 | ロール角 |
| `pitch` | float | 度 | ピッチ角 |
| `roll_change` | float | 度 | ロール変化 |
| `pitch_change` | float | 度 | ピッチ変化 |
| `red_dot_mean_x` | float | px | 赤ドットX座標 |
| `red_dot_mean_y` | float | px | 赤ドットY座標 |
| `green_dot_mean_x` | float | px | 緑ドットX座標 |
| `green_dot_mean_y` | float | px | 緑ドットY座標 |
| `red_dot_x_change` | float | px | 赤ドットX変化 |
| `green_dot_x_change` | float | px | 緑ドットX変化 |
| `audio_angle_change` | float | 度 | 音響角度変化 |

---

## 6.3 解析出力ファイル

### integrated_analysis.csv

統合解析結果（analyze_datas.py出力）。

| カラム名 | 型 | 単位 | 説明 |
|---------|---|------|------|
| `psychopy_time` | float | 秒 | 統一時間軸 |
| `accel_x` | float | m/s² | X軸加速度 |
| `accel_y` | float | m/s² | Y軸加速度 |
| `accel_z` | float | m/s² | Z軸加速度 |
| `angle_change` | float | 度 | 角度変化（ロール） |
| `roll` | float | 度 | ロール角 |
| `pitch` | float | 度 | ピッチ角 |
| `roll_change` | float | 度 | ロール変化 |
| `pitch_change` | float | 度 | ピッチ変化 |
| `red_dot_mean_x` | float | px | 赤ドットX座標 |
| `red_dot_mean_y` | float | px | 赤ドットY座標 |
| `green_dot_mean_x` | float | px | 緑ドットX座標 |
| `green_dot_mean_y` | float | px | 緑ドットY座標 |
| `red_dot_x_change` | float | px | 赤ドットX変化 |
| `green_dot_x_change` | float | px | 緑ドットX変化 |
| `gvs_dac_output` | float | - | GVS出力値（gvs/all条件） |
| `audio_angle_change` | float | 度 | 音響角度変化（audio/all条件） |

- **サンプリング**: 60Hz（統一）

---

### integrated_analysis_sway_3.0Hz.csv

身体動揺解析結果（postural_sway_analyzer.py出力）。

| カラム名 | 型 | 単位 | 説明 |
|---------|---|------|------|
| `psychopy_time` | float | 秒 | 時間 |
| `accel_x_sway` | float | m/s² | X軸加速度（3Hzフィルタ済） |
| `accel_y_sway` | float | m/s² | Y軸加速度（3Hzフィルタ済） |
| `accel_z_sway` | float | m/s² | Z軸加速度（3Hzフィルタ済） |
| `roll_sway` | float | 度 | ロール（3Hzフィルタ済） |
| `pitch_sway` | float | 度 | ピッチ（3Hzフィルタ済） |
| `angle_change_sway` | float | 度 | 角度変化（3Hzフィルタ済） |
| `red_dot_mean_x_sway` | float | px | 赤ドットX（3Hzフィルタ済） |
| `green_dot_mean_x_sway` | float | px | 緑ドットX（3Hzフィルタ済） |
| `red_dot_x_change_sway` | float | px | 赤ドットX変化（3Hzフィルタ済） |
| `green_dot_x_change_sway` | float | px | 緑ドットX変化（3Hzフィルタ済） |
| `correlation_angle_red_dot` | float | - | 角度-赤ドット窓相関 |
| `correlation_angle_green_dot` | float | - | 角度-緑ドット窓相関 |
| `accel_x` | float | m/s² | X軸加速度（元データ） |
| `accel_y` | float | m/s² | Y軸加速度（元データ） |
| `accel_z` | float | m/s² | Z軸加速度（元データ） |
| `angle_change` | float | 度 | 角度変化（元データ） |
| `red_dot_mean_x` | float | px | 赤ドットX（元データ） |
| `green_dot_mean_x` | float | px | 緑ドットX（元データ） |

---

### phase_analysis_3.0Hz.csv

位相解析結果（phase_correlation_analyzer.py出力）。

| カラム名 | 型 | 単位 | 説明 |
|---------|---|------|------|
| `psychopy_time` | float | 秒 | 時間 |
| `angle_change_filtered` | float | 度 | 角度変化（フィルタ済） |
| `red_dot_mean_x_filtered` | float | px | 赤ドットX（フィルタ済） |
| `green_dot_mean_x_filtered` | float | px | 緑ドットX（フィルタ済） |
| `red_dot_x_change_filtered` | float | px | 赤ドットX変化（フィルタ済） |
| `green_dot_x_change_filtered` | float | px | 緑ドットX変化（フィルタ済） |
| `angle_change_phase` | float | rad | 角度変化の位相 |
| `red_dot_phase` | float | rad | 赤ドットの位相 |
| `green_dot_phase` | float | rad | 緑ドットの位相 |
| `red_dot_phase_diff` | float | rad | 角度-赤ドット位相差 |
| `green_dot_phase_diff` | float | rad | 角度-緑ドット位相差 |
| `phase_correlation_angle_red_dot` | float | - | 角度-赤ドット位相相関 |
| `phase_correlation_angle_green_dot` | float | - | 角度-緑ドット位相相関 |

---

### window_correlations_3.0Hz.csv

窓相関解析結果（phase_correlation_analyzer.py出力）。

| カラム名 | 型 | 単位 | 説明 |
|---------|---|------|------|
| `psychopy_time` | float | 秒 | 窓中心時刻 |
| `phase_correlation_angle_red_dot` | float | - | 角度-赤ドット位相相関（窓平均） |
| `phase_correlation_angle_green_dot` | float | - | 角度-緑ドット位相相関（窓平均） |

---

### correlation_summary_3.0Hz.csv

相関統計サマリー（postural_sway_analyzer.py出力）。

| カラム名 | 型 | 説明 |
|---------|---|------|
| `session_id` | str | セッションID |
| `folder_type` | str | フォルダタイプ（vis/audio/gvs/all） |
| `condition` | str | 実験条件（red/green） |
| `single_color_dot` | bool | 単色ドットモード |
| `visual_reverse` | bool | 視覚反転フラグ |
| `audio_reverse` | bool | 音響反転フラグ |
| `gvs_reverse` | bool | GVS反転フラグ |
| `angle_mean` | float | 角度変化平均 |
| `angle_std` | float | 角度変化標準偏差 |
| `angle_var` | float | 角度変化分散 |
| `angle_q1` | float | 角度変化第1四分位数 |
| `angle_median` | float | 角度変化中央値 |
| `angle_q3` | float | 角度変化第3四分位数 |
| `angle_min` | float | 角度変化最小値 |
| `angle_max` | float | 角度変化最大値 |
| `red_window_corr_mean` | float | 赤ドット窓相関平均 |
| `red_window_corr_std` | float | 赤ドット窓相関標準偏差 |
| `green_window_corr_mean` | float | 緑ドット窓相関平均 |
| `green_window_corr_std` | float | 緑ドット窓相関標準偏差 |
| `total_correlation_red` | float | 赤ドット全体相関係数 |
| `total_correlation_green` | float | 緑ドット全体相関係数 |

---

## 6.4 カラム命名規則

### 接尾辞

| 接尾辞 | 意味 |
|-------|------|
| `_sway` | 3Hzローパスフィルタ適用済み |
| `_filtered` | フィルタ適用済み |
| `_change` | 初期値からの変化量 |
| `_norm` | 正規化済み |
| `_phase` | 位相値 |
| `_diff` | 差分値 |

### 接頭辞

| 接頭辞 | 意味 |
|-------|------|
| `accel_` | 加速度関連 |
| `red_dot_` | 赤ドット関連 |
| `green_dot_` | 緑ドット関連 |
| `audio_` | 音響関連 |
| `gvs_` | GVS関連 |
| `phase_` | 位相関連 |
| `correlation_` | 相関関連 |

---

*前ページ: [解析プログラム](./05-解析プログラム.md) | 次ページ: [コマンドライン引数](./07-コマンドライン引数.md)*
