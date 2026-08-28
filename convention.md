# Code Conventions

## 文件结构

- 转换逻辑放 `transforms/`，测试放 `tests/`
- 不使用 `__init__.py`；在文件顶部直接 import 并按需重命名

## Import

```python
# 直接引用模块文件
from transforms.real_world_transforms import apply_real_world_transform as apply_transform
```

## 函数

- 公开函数加类型注解和 docstring
- 私有辅助函数以 `_` 开头（如 `_load_rgb`、`_clip`）
- 参数用 `*` 强制 keyword-only（如 `value=`, `seed=`）

## 命名

| 类型 | 规范 |
|------|------|
| 函数/变量 | `snake_case` |
| 类型别名 | `PascalCase` |
| 常量 | `UPPER_SNAKE_CASE` |


## 其他

- Python 3.10+，使用 `from __future__ import annotations`
- 行长不超过 100 字符
- 不提交 `__pycache__/`、`.venv/`、`.DS_Store`
