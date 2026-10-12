#!/usr/bin/env bash
set -e

echo "Enter the path to where the collectiveapi repo (https://github.com/astra-sim/collectiveapi) is cloned."
read -p "Press Enter to use default path (\$HOME/collectiveapi), or type 'abort' or 'x' to exit: " COLLECTIVEAPI_PATH

if [ -z "$COLLECTIVEAPI_PATH" ]; then
    COLLECTIVEAPI_PATH="$HOME/collectiveapi"
fi
if [ "$COLLECTIVEAPI_PATH" = "abort" ] || [ "$COLLECTIVEAPI_PATH" = "x" ]; then
    echo "Aborted by user."
    exit 1
fi
echo "COLLECTIVEAPI_PATH: $COLLECTIVEAPI_PATH"

SCRIPT_DIR=$(dirname "$(realpath "$0")")
# 产出物命名/落点与仓内已提交示例及 custom_collective.json 的引用保持一致：
#   examples/system/custom_collectives/custom_ring_allreduce_8npus_1MB/custom_allreduce.<0..7>.et
# （2026-09-26 修复批前本脚本曾把 allreduce_ring_8ranks.* 散落生成在本目录、
#   与已提交示例不一致，且调用 `python`——镜像仅保证 python3。）
OUTPUT_DIR="$SCRIPT_DIR/custom_ring_allreduce_8npus_1MB"
OUTPUT_BASE="$OUTPUT_DIR/custom_allreduce"
mkdir -p "$OUTPUT_DIR"

echo "Generating XML representation of custom allreduce algorithm from MSCCLang..."
python3 "$COLLECTIVEAPI_PATH/msccl-tools/examples/mscclang/allreduce_a100_ring.py" 8 1 1 > "$OUTPUT_BASE.xml"

echo "Converting XML to Chakra ET representation..."
python3 "$COLLECTIVEAPI_PATH/chakra_converter/et_converter.py" --input_filename "$OUTPUT_BASE.xml" --output_filename "$OUTPUT_BASE" --coll_size 1048576 --collective allreduce

read -p "Optional: Do you want to convert the Chakra ET representation to JSON? [y/n] " CONVERT_TO_JSON

if [ "$CONVERT_TO_JSON" = "y" ] || [ -z "$CONVERT_TO_JSON" ]; then
    echo "Converting Chakra ET representation to JSON..."
    for i in {0..7}; do
        chakra_jsonizer \
            --input_filename "$OUTPUT_BASE.$i.et" \
            --output_filename "$OUTPUT_BASE.$i.json"
    done
fi
