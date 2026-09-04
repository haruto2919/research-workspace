# Skill Benchmark: notion-material-archive

**Model**: gpt-5
**Date**: 2026-07-19T06:46:46Z
**Evals**: 1, 2, 3 (1 run each per configuration)

## Summary

| Metric | With Skill | Without Skill | Delta |
|--------|------------|---------------|-------|
| Pass Rate | 100% ± 0% | 79% ± 7% | +0.21 |
| Time | 0.0s ± 0.0s | 0.0s ± 0.0s | +0.0s |
| Tokens | 4458 ± 2000 | 3437 ± 984 | +1021 |

## Notes

- With-skillは全19判定を通過し、without-skillは15/19だった。
- 最大の差は画像処理で、without-skillは一時HTTPS URLを最終pageへ残す計画だったが、with-skillはNotionのfile-upload blockへ置換した。
- 一般的な安全判定は両方で通りやすいため、helper、staged-copy checksum、cleanupを比較判定に含めた。
- 実行時間と真のtoken数は通知から取得できず、timeは0、token欄はoutput文字数のproxyである。
