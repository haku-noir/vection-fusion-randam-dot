#!/bin/bash

# 一連の解析プログラム実行スクリプト
# 使用法: ./run_analysis.sh [オプション] フォルダ名1 [フォルダ名2 ...]
# オプション:
#   -p : postural_sway_analyzerからスタート
#   -c : correlation_visualizerからスタート
#   (オプションなし) : analyze_datasからフル実行

set -e  # エラー時に停止

# デフォルト設定
START_FROM="full"
FOLDERS=()

# ヘルプ表示関数
show_help() {
    echo "使用法: $0 [オプション] フォルダ名1 [フォルダ名2 ...]"
    echo ""
    echo "オプション:"
    echo "  -p          postural_sway_analyzerからスタート"
    echo "  -c          correlation_visualizerからスタート"
    echo "  -h, --help  このヘルプを表示"
    echo ""
    echo "例:"
    echo "  $0 data1                    # data1フォルダをフル実行"
    echo "  $0 -p data1 data2           # 複数フォルダをpostural_sway_analyzerから実行"
    echo "  $0 -c data1 data2 data3     # 複数フォルダをcorrelation_visualizerから実行"
    echo ""
    echo "実行される処理:"
    echo "  フル実行: analyze_datas.py → postural_sway_analyzer.py → correlation_visualizer.py"
    echo "  -p指定:   postural_sway_analyzer.py → correlation_visualizer.py"
    echo "  -c指定:   correlation_visualizer.py"
}

# プログラム実行関数
run_analyze_datas() {
    local folder="$1"
    echo "━━━ analyze_datas.py実行中: $folder ━━━"
    if python3 analyze_datas.py "$folder"; then
        echo "✓ analyze_datas.py完了: $folder"
    else
        echo "✗ analyze_datas.pyエラー: $folder" >&2
        return 1
    fi
}

run_postural_sway_analyzer() {
    local folder="$1"
    echo "━━━ postural_sway_analyzer.py実行中: $folder ━━━"
    if python3 postural_sway_analyzer.py "$folder"; then
        echo "✓ postural_sway_analyzer.py完了: $folder"
    else
        echo "✗ postural_sway_analyzer.pyエラー: $folder" >&2
        return 1
    fi
}

run_correlation_visualizer() {
    local folder="$1"
    echo "━━━ correlation_visualizer.py実行中: $folder ━━━"
    if python3 correlation_visualizer.py "$folder"; then
        echo "✓ correlation_visualizer.py完了: $folder"
    else
        echo "✗ correlation_visualizer.pyエラー: $folder" >&2
        return 1
    fi
}

# 単一フォルダの処理
process_folder() {
    local folder="$1"
    local start_from="$2"

    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════════════════════╗"
    echo "║ フォルダ処理開始: $folder"
    echo "║ 開始ポイント: $start_from"
    echo "╚════════════════════════════════════════════════════════════════════════════════════════╝"

    # フォルダの存在確認
    if [ ! -d "$folder" ]; then
        echo "✗ エラー: フォルダが存在しません: $folder" >&2
        return 1
    fi

    local start_time=$(date +%s)

    case "$start_from" in
        "full")
            run_analyze_datas "$folder" && \
            run_postural_sway_analyzer "$folder" && \
            run_correlation_visualizer "$folder"
            ;;
        "postural")
            run_postural_sway_analyzer "$folder" && \
            run_correlation_visualizer "$folder"
            ;;
        "correlation")
            run_correlation_visualizer "$folder"
            ;;
        *)
            echo "✗ 不正な開始ポイント: $start_from" >&2
            return 1
            ;;
    esac

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ フォルダ処理完了: $folder (実行時間: ${minutes}分${seconds}秒)"
    else
        echo ""
        echo "❌ フォルダ処理失敗: $folder (実行時間: ${minutes}分${seconds}秒)"
        return 1
    fi
}

# コマンドライン引数の解析
while [[ $# -gt 0 ]]; do
    case $1 in
        -p)
            START_FROM="postural"
            shift
            ;;
        -c)
            START_FROM="correlation"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -*)
            echo "✗ 不正なオプション: $1" >&2
            echo "ヘルプを表示するには: $0 --help"
            exit 1
            ;;
        *)
            FOLDERS+=("$1")
            shift
            ;;
    esac
done

# フォルダが指定されていない場合のエラー
if [ ${#FOLDERS[@]} -eq 0 ]; then
    echo "✗ エラー: フォルダが指定されていません" >&2
    echo ""
    show_help
    exit 1
fi

# メイン処理
echo "════════════════════════════════════════════════════════════════════════════════════════════"
echo "解析プログラム実行スクリプト"
echo "開始時刻: $(date '+%Y-%m-%d %H:%M:%S')"
echo "開始ポイント: $START_FROM"
echo "対象フォルダ数: ${#FOLDERS[@]}"
echo "対象フォルダ: ${FOLDERS[*]}"
echo "════════════════════════════════════════════════════════════════════════════════════════════"

TOTAL_START_TIME=$(date +%s)
SUCCESS_COUNT=0
FAILURE_COUNT=0
FAILED_FOLDERS=()

# 各フォルダを処理
for folder in "${FOLDERS[@]}"; do
    if process_folder "$folder" "$START_FROM"; then
        ((SUCCESS_COUNT++))
    else
        ((FAILURE_COUNT++))
        FAILED_FOLDERS+=("$folder")
    fi
done

TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))
TOTAL_MINUTES=$((TOTAL_DURATION / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

# 最終結果
echo ""
echo "════════════════════════════════════════════════════════════════════════════════════════════"
echo "全体処理完了"
echo "終了時刻: $(date '+%Y-%m-%d %H:%M:%S')"
echo "総実行時間: ${TOTAL_MINUTES}分${TOTAL_SECONDS}秒"
echo "処理フォルダ数: ${#FOLDERS[@]}"
echo "成功: $SUCCESS_COUNT"
echo "失敗: $FAILURE_COUNT"

if [ $FAILURE_COUNT -gt 0 ]; then
    echo "失敗したフォルダ: ${FAILED_FOLDERS[*]}"
    echo "════════════════════════════════════════════════════════════════════════════════════════════"
    exit 1
else
    echo "すべての処理が正常に完了しました！"
    echo "════════════════════════════════════════════════════════════════════════════════════════════"
    exit 0
fi
