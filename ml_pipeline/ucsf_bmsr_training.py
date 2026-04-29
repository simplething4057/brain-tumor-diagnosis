# UCSF-BMSR MET Training Notebook
# 최종 수정 버전 - 2026-04-16
# 변경사항:
#   - cache_dir: Drive → 로컬 /content/persistent_cache (Drive 용량 문제 해결)
#   - init_filters: 32 유지 (A100 선택 시), 16으로 변경 가능 (T4 사용 시)
#   - VAL_EVERY: 1 (매 epoch 검증 + 체크포인트)
#   - 중복 코드 제거, 변수명 통일

