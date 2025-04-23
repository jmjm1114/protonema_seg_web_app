
#%%
import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import pandas as pd
from typing import List, Tuple
#%%
def read_images_from_upload(uploaded_file) -> List[Tuple[str, np.ndarray]]:
    """
    업로드된 파일이 단일 이미지이면 하나만,
    ZIP 파일이면 내부 이미지 전부를 추출하여 (파일명, img) 리스트로 반환한다.
    """
    images = []
    filename = uploaded_file.name.lower()

    # 파일 내용 바이트로 읽기
    file_bytes = uploaded_file.read()

    if filename.endswith('.zip'):
        # ZIP 파일로 처리
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            for name in zf.namelist():
                # 내부 파일 중 이미지 확장자만 시도
                if any(name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']):
                    # 파일 읽기
                    img_data = zf.read(name)
                    # OpenCV로 디코딩
                    np_data = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
                    img = ensure_3ch(img)  # 그레이/4채널 → 3채널 변환
                    if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
                        images.append((name, img))
    else:
        # 단일 이미지로 처리
        np_data = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
        img = ensure_3ch(img)  # 그레이/4채널 → 3채널 변환
        if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
            images.append((uploaded_file.name, img))

    return images

def ensure_3ch(img: np.ndarray) -> np.ndarray:
    """
    입력 이미지를 항상 3채널 BGR로 맞춰준다.
    - 그레이(2D) → BGR
    - 4채널(BGRA) → BGR
    """
    if img is None:
        return None
    if len(img.shape) == 2:
        # Grayscale인 경우
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        # BGRA인 경우
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def find_scale_bar(img, expected_bar_width_mm=1.0):
    """
    (간단 예시) 우측 하단에서 스케일바 검출 후
    픽셀 길이를 측정해 mm당 픽셀 환산비를 리턴.
    """
    h, w = img.shape[:2]
    # 안전장치: h나 w가 매우 작으면 바로 None 반환
    if h < 80 or w < 200:
        return None

    roi = img[h-80:h, w-200:w].copy()

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_width = 0
    for cnt in contours:
        x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
        if w_cnt > max_width:
            max_width = w_cnt

    if max_width > 0:
        mm_per_pixel = expected_bar_width_mm / max_width
    else:
        mm_per_pixel = None
    return mm_per_pixel

def segment_protonema_by_excluding_bg(
    img: np.ndarray,
    known_background_bgr=(204, 216, 152),
    k=2,
    attempts=10
):
    """
    K-means로 이미지 분할 후,
    배경 색에 가장 가까운 클러스터를 제외한 나머지를 '프로토네마'라 간주.
    """
    # 1) LAB 변환
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    H, W = lab.shape[:2]

    # 2D로 만들기
    pixel_vals = lab.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)

    # 2) K-means 실행
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixel_vals, k, None, criteria, attempts,
        cv2.KMEANS_RANDOM_CENTERS
    )

    # 3) '배경' 클러스터 찾기
    known_bg_lab = cv2.cvtColor(
        np.uint8([[known_background_bgr]]), cv2.COLOR_BGR2LAB
    )[0, 0, :]
    distances = [np.linalg.norm(c - known_bg_lab) for c in centers]
    background_cluster_idx = np.argmin(distances)

    # 4) 마스크 생성 (배경이 아닌 것만)
    labels_2d = labels.reshape(H, W)
    mask = (labels_2d != background_cluster_idx).astype(np.uint8)

    # 5) 노이즈 제거
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # (선택) 가장 큰 연결 요소만 남기기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        new_mask = np.zeros_like(mask)
        cv2.drawContours(new_mask, [largest_contour], -1, 1, cv2.FILLED)
        mask = new_mask

    # 6) 면적 계산
    total_pixels = H * W
    protonema_area = cv2.countNonZero(mask)
    protonema_percentage = (protonema_area / total_pixels) * 100

    # 7) 결과 시각화
    result_img = img.copy()
    result_img[mask > 0] = [0, 255, 0]  # 초록색 표시

    return mask, result_img, protonema_area, protonema_percentage

def main():
    st.title("Protonema Segmentation v.2.4")
    st.write("여러 이미지를 한 번에 업로드 가능, zip 파일 처리 가능.")
    st.write("K값은 3 추천. 처리 잘 안될 시 배경 RGB값 조정 필요.")
    st.write("스케일바 기능 추가. 이미지별 스케일바 길이 조정 가능.")
    st.write("결과는 CSV로 다운로드 가능.(선택사항)")

    k = st.sidebar.slider("클러스터 개수 (k)", 2, 6, 3)
    attempts = st.sidebar.slider("K-means 반복 횟수 (attempts)", 1, 20, 10)

    bg_r = st.sidebar.number_input("배경 R값 (0~255)", 0, 255, 240)
    bg_g = st.sidebar.number_input("배경 G값 (0~255)", 0, 255, 240)
    bg_b = st.sidebar.number_input("배경 B값 (0~255)", 0, 255, 240)
    known_background_bgr = (bg_b, bg_g, bg_r)  # OpenCV: BGR 순서

    # ---- 결과 저장을 위한 리스트 ----
    results_list = []

    uploaded_files = st.file_uploader(
        "이미지 또는 ZIP 파일을 업로드하세요 (다중 가능)",
        type=['jpg','jpeg','png','tif','tiff','zip'],
        accept_multiple_files=True
    )

    if uploaded_files:
        for up_file in uploaded_files:
            images_data = read_images_from_upload(up_file)
            if not images_data:
                st.warning(f"'{up_file.name}'에서 유효한 이미지를 찾지 못했습니다.")
                continue

            for (fname, img) in images_data:
                st.subheader(f"파일명: {fname}")

                # 이미지별 스케일바 길이 설정
                scale_bar_length = st.sidebar.number_input(
                    f"스케일바 길이(mm) [{fname}]",
                    min_value=0.1,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                    key=f"scale_bar_length_{fname}"
                )

                mm_per_pixel = find_scale_bar(img, expected_bar_width_mm=scale_bar_length)
                mask, result_img, area_px, ratio_pct = segment_protonema_by_excluding_bg(
                    img,
                    known_background_bgr=known_background_bgr,
                    k=k,
                    attempts=attempts
                )

                h, w = img.shape[:2]
                total_px = h * w

                st.write(f"• 프로토네마 픽셀 수: {area_px}")
                st.write(f"• 전체 픽셀 수: {total_px}")
                st.write(f"• 프로토네마 비율: {ratio_pct:.2f}%")

                if mm_per_pixel:
                    area_mm2 = area_px * (mm_per_pixel ** 2)
                    px_per_mm = int(1 / mm_per_pixel) if mm_per_pixel != 0 else 0
                    st.write(f"• 추정 스케일바 길이(pixels): 1 mm ≈ {px_per_mm} px")
                    st.write(f"• 프로토네마 실제 면적 추정: {area_mm2:.4f} mm²")
                else:
                    st.warning("스케일바를 찾지 못했습니다. (면적 계산 불가)")
                    area_mm2 = None

                col1, col2 = st.columns(2)
                with col1:
                    st.image(
                        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                        caption="원본 이미지",
                        use_container_width=True
                    )
                with col2:
                    st.image(
                        cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                        caption="배경 제외 후 프로토네마 표시",
                        use_container_width=True
                    )

                # ---- 결과를 리스트에 추가 ----
                results_list.append({
                    "파일명": fname,
                    "프로토네마 픽셀 수": area_px,
                    "전체 픽셀 수": total_px,
                    "프로토네마 비율(%)": ratio_pct,
                    "스케일바 길이(mm)": scale_bar_length,
                    "1 mm당 픽셀(px/mm)": (int(1 / mm_per_pixel) if mm_per_pixel else None),
                    "프로토네마 면적(mm^2)": area_mm2
                })

        # ---- 모든 파일 처리 후, CSV 다운로드 버튼 ----
        st.write("---")
        st.subheader("CSV 다운로드 (선택 기능)")

        if st.button("CSV 생성하기"):
            if len(results_list) == 0:
                st.warning("분석 결과가 없습니다.")
            else:
                df = pd.DataFrame(results_list)
                csv_data = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="CSV 다운로드",
                    data=csv_data,
                    file_name="protonema_results.csv",
                    mime="text/csv"
                )

#%%
if __name__ == "__main__":
    main()