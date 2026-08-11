# 领域契约

本目录不依赖 PyTorch 或模拟器，只允许 Python 标准库。

| 文件 | 作用 |
| --- | --- |
| `rules.py` | 水果等级、名称、半径、质量、生成和合并计分 |
| `state.py` | 非 Tensor 状态与结果 dataclass |
| `__init__.py` | 公共导出 |

- 规则值以 `rules.py` 为准，不在模拟器或 UI 复制另一套来源。
- 改公共类型或导出时检查消费者；先用符号名搜索，不浏览整个项目。
- 改名称、半径或等级时额外核对 `assets/fruits` 映射和模拟器规则表。

最小测试：

```powershell
$env:PYTHONPATH='src'
conda run -n python-torch python -m unittest tests.test_domain_contracts -v
```
