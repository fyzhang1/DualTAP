#!/bin/bash
# API评估示例脚本

# 设置颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== 对抗样本迁移性评估示例 ===${NC}\n"

# 检查点路径（请根据实际情况修改）
CHECKPOINT="./checkpoints/generator_epoch_50.pth"

# 如果检查点不存在，给出提示
if [ ! -f "$CHECKPOINT" ]; then
    echo "警告: 检查点文件不存在: $CHECKPOINT"
    echo "请先训练模型或修改CHECKPOINT变量为正确的路径"
    exit 1
fi

# 创建输出目录
mkdir -p ./eval_results

echo -e "${GREEN}1. 使用本地模型评估（Qwen2.5-VL-7B）${NC}"
echo "命令: python evaluate.py --checkpoint $CHECKPOINT --output ./eval_results/eval_local.json"
echo ""

# 如果想实际运行，取消下面的注释
# python evaluate.py --checkpoint $CHECKPOINT --output ./eval_results/eval_local.json

echo -e "${GREEN}2. 使用OpenAI GPT-4o API评估${NC}"
echo "首先设置API密钥:"
echo "  export OPENAI_API_KEY=''"
echo ""
echo "然后运行:"
echo "  python evaluate.py --checkpoint $CHECKPOINT \\"
echo "    --output ./eval_results/eval_gpt4o.json \\"
echo "    --use-api --api-type openai --api-model gpt-4o"
echo ""

# 如果设置了OPENAI_API_KEY，可以运行
# if [ ! -z "$OPENAI_API_KEY" ]; then
#     python evaluate.py --checkpoint $CHECKPOINT \
#         --output ./eval_results/eval_gpt4o.json \
#         --use-api --api-type openai --api-model gpt-4o
# fi

echo -e "${GREEN}3. 使用Claude API评估${NC}"
echo "首先设置API密钥:"
echo "  export ANTHROPIC_API_KEY='your-api-key-here'"
echo ""
echo "然后运行:"
echo "  python evaluate.py --checkpoint $CHECKPOINT \\"
echo "    --output ./eval_results/eval_claude.json \\"
echo "    --use-api --api-type claude"
echo ""

echo -e "${GREEN}4. 使用Gemini API评估${NC}"
echo "首先设置API密钥:"
echo "  export GEMINI_API_KEY='your-api-key-here'"
echo ""
echo "然后运行:"
echo "  python evaluate.py --checkpoint $CHECKPOINT \\"
echo "    --output ./eval_results/eval_gemini.json \\"
echo "    --use-api --api-type gemini"
echo ""

echo -e "${BLUE}=== 提示 ===${NC}"
echo "1. 详细使用说明请查看: API_EVALUATION_GUIDE.md"
echo "2. 使用API前请确保安装对应的库:"
echo "   pip install openai anthropic google-generativeai"
echo "3. 注意API调用会产生费用"
echo ""

