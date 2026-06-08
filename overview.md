# LCA CAG 3DGS Research Dashboard Overview

본 대시보드는 좌관상동맥 조영술(LCA CAG) 이미지 기반의 **2차원 다중 작업 혈관 세그멘테이션(Multi-Task DUNet)** 모델 연구 결과와, 이를 미분 가능한 **3차원 가우시안 스플래팅(3DGS)** 최적화와 결합하여 혈관의 3D 볼륨(내경 반경 및 중심 경로)을 복원하는 PoC 연구의 핵심 결과물들을 종합적으로 제시합니다.

---

## 연구 핵심 프레임워크 (Framework Overview)

<div style="display: flex; gap: 24px; align-items: stretch; margin: 32px 0; flex-wrap: wrap;">
  <div style="flex: 1; min-width: 320px; background: rgba(255,255,255,0.02); border: 1px solid var(--card-border); border-radius: 12px; padding: 20px; display: flex; flex-direction: column; justify-content: space-between;">
    <div>
      <h4 style="margin-top: 0; margin-bottom: 12px; text-align: center; color: var(--text-main); font-size: 1.1rem; display: flex; align-items: center; justify-content: center; gap: 8px;">
        Multi-Task DUNet 모델 아키텍처
      </h4>
      <p style="font-size: 0.9rem; color: var(--text-muted); margin-bottom: 16px; line-height: 1.5; text-align: justify;">
        Deformable Convolution 블록과 인코더-디코더 스킵 커넥션을 이용해 토폴로지 연결성이 유지된 혈관 마스크를 추출하고, 다중 분지 헤드를 통해 LAD, LCx, LM 및 전체 혈관 마스크를 통합 학습하는 신경망 구조입니다.
      </p>
    </div>
    <img src="/Users/home/.gemini/antigravity/brain/af8691ec-7f2d-4229-a6a4-e8f4635b5bd9/correct_dunet_vector_architecture_1780909356736.png" alt="Multi-Task DUNet 아키텍처" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); cursor: zoom-in; margin: 0;">
  </div>
  <div style="flex: 1; min-width: 320px; background: rgba(255,255,255,0.02); border: 1px solid var(--card-border); border-radius: 12px; padding: 20px; display: flex; flex-direction: column; justify-content: space-between;">
    <div>
      <h4 style="margin-top: 0; margin-bottom: 12px; text-align: center; color: var(--text-main); font-size: 1.1rem; display: flex; align-items: center; justify-content: center; gap: 8px;">
        3DGS 연계 볼륨 복원 파이프라인
      </h4>
      <p style="font-size: 0.9rem; color: var(--text-muted); margin-bottom: 16px; line-height: 1.5; text-align: justify;">
        2차원 예측 마스크와 다방향 포즈 기하정보를 이용한 클래스 제약 삼각측량으로 3차원 초기 시드 좌표를 도출하고, 미분 가능한 3D 가우시안 렌더러 기반 최적화 루프를 통해 3차원 관상동맥의 실질적인 굵기와 볼륨을 복원하는 기법입니다.
      </p>
    </div>
    <img src="/Users/home/.gemini/antigravity/brain/af8691ec-7f2d-4229-a6a4-e8f4635b5bd9/dunet_3dgs_reconstruction_pipeline_1780909757909.png" alt="전체 파이프라인 흐름도" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); cursor: zoom-in; margin: 0;">
  </div>
</div>

---

## PoC 주요 연구 성과 요약 (Executive Summary)

*   **2D Segmentation Accuracy**: 클래스 가중치 Focal Loss와 Dice Loss를 결합하여 **LCA Test Set 기준 76.55%**의 통합 혈관 Dice 성능을 확보.
*   **3D Projection Agreement**: 양방향 C-Arm 기하 투영 오차를 최소화하는 3DGS 역전파 최적화를 통해 투영 일치도 **Mean Projection Dice 84.96%** 달성.
*   **Volumetric Radius Profiling**: LAD 및 LM 분지별 추적 거리에 따른 물리적 내경 직경(Diameter, $D = 2r$) 변화를 정량화(0.8mm~2.7mm 범위 표출).
*   **Memory Efficiency**: 3DGS 최적화 OOM 문제를 극복하고자 1/2 포인트 다운샘플링 및 1/4 해상도 최적화 연산 설계를 도입하여 L4 GPU 환경에서 **메모리 8배 절약 (2.5 GiB 구동 가능)** 성공.

---

## 대시보드 문서 바로가기

아래의 사이드바 메뉴를 통해 세부 단계별 상세 연구 결과를 조회하실 수 있습니다:

1.  **초기 탐색 단계 보고서**: 기본 U-Net 및 Deformable U-Net(DUNet) 모델 설계 및 2D 세그멘테이션 기본 실험 결과 분석.
2.  **메인 단계 보고서**: Multi-Task 세그멘테이션 헤드 결합과 Focal Loss 적용에 따른 성능 대조 분석 및 평가.
3.  **1차 결과 워크스루**: 삼각측량 기반 기하 구조 초기화 및 초기 3D 뼈대 복원 모델 성과.
4.  **2차 결과 워크스루**: 미분 가능한 3D 가우시안 렌더러 수식 모델 및 역전파 학습에 따른 3D 혈관 볼륨 복원 및 직경 프로파일 정량 분석 결과.
5.  **전체 연구 로드맵**: 움직임 보정(Dynamic 3DGS) 및 임상 전개를 위한 중장기 연구 발전 로드맵 제시.
