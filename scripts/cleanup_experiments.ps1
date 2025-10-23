# 统一入口：清理实验目录
# 用法：
#   干跑预览：./scripts/cleanup_experiments.ps1 --dry-run
#   移至回收站：./scripts/cleanup_experiments.ps1
#   永久删除：./scripts/cleanup_experiments.ps1 --mode delete

python "$PSScriptRoot/../cleanup_experiments.py" $args