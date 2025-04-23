
#%%
import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
#%%
def segment_protonema_by_excluding_bg(
    img, 
    k=2,
    attempts=10, 
    known_background_bgr=(204, 216, 152)
):
    """
    K-means로 분할한 뒤, 배경 색에 가장 가까운 클러스터를 제외하고 나머지를
    '프로토네마'로 간주하는 예시
    """
    # 1) LAB 변환
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    pixel_vals = lab.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)
    
    # 2) K-means 실행
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
    
    # 3) 클러스터링 결과 -> 이미지
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(img.shape)
    
    # 4) '배경' 클러스터 찾기
    known_bg_lab = cv2.cvtColor(np.uint8([[known_background_bgr]]), cv2.COLOR_BGR2LAB)[0,0,:]
    distances = [np.linalg.norm(center - known_bg_lab) for center in centers]
    background_cluster_idx = np.argmin(distances)  # 배경과 가장 가까운 클러스터
    
    # 5) 마스크 생성: 배경이 아닌 것만 1
    mask = (labels != background_cluster_idx).astype(np.uint8) * 255
    
    # 6) 노이즈 제거 (모폴로지 연산)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # (선택) 가장 큰 연결 요소만 남기고 싶다면:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        new_mask = np.zeros_like(mask)
        cv2.drawContours(new_mask, [largest_contour], -1, 255, cv2.FILLED)
        mask = new_mask
    
    # 7) 결과 시각화
    result_img = img.copy()
    result_img[mask > 0] = [0, 255, 0]  # 초록색
    
    return mask, segmented_image, result_img

def find_scale_bar(img, expected_bar_width_mm=1.0):
    """
    (간단 예시) 우측 하단에서 스케일바 검출 후 픽셀 길이를 측정해 mm 당 픽셀 환산비를 리턴
    """
    h, w = img.shape[:2]
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

def main():
    st.title("Protonema Segmentation v.2.1")
    st.write("배경 색을 알려주고, 해당 색과 가장 가까운 클러스터를 배경으로 보고 제외합니다.")

    k = st.sidebar.slider("클러스터 개수 (k)", 2, 6, 2)
    attempts = st.sidebar.slider("K-means 반복 횟수 (attempts)", 1, 20, 10)
    
    # 배경 대표색(BGR) 사용자 입력(선택)
    bg_r = st.sidebar.number_input("배경 R값 (0~255)", 0, 255, 204)
    bg_g = st.sidebar.number_input("배경 G값 (0~255)", 0, 255, 216)
    bg_b = st.sidebar.number_input("배경 B값 (0~255)", 0, 255, 152)
    known_background_bgr = (bg_b, bg_g, bg_r)  # OpenCV는 BGR 순서!
    
    uploaded_files = st.file_uploader(
        "이미지 파일들을 여러 장 선택하세요", 
        type=['jpg','jpeg','png','tif','tiff'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.subheader(f"파일명: {uploaded_file.name}")
            
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img is None:
                st.error("이미지를 읽을 수 없습니다.")
                continue
            
            mm_per_pixel = find_scale_bar(img, expected_bar_width_mm=1.0)
            
            mask, segmented, result_img = segment_protonema_by_excluding_bg(
                img, k, attempts, known_background_bgr
            )
            
            protonema_pixels = cv2.countNonZero(mask)
            total_pixels = img.shape[0] * img.shape[1]
            protonema_ratio = (protonema_pixels / total_pixels) * 100
            
            if mm_per_pixel:
                area_in_mm2 = protonema_pixels * (mm_per_pixel ** 2)
                st.write(f"• 프로토네마 픽셀 수: {protonema_pixels}")
                st.write(f"• 전체 픽셀 수: {total_pixels}")
                st.write(f"• 프로토네마 비율: {protonema_ratio:.2f}%")
                st.write(f"• 추정 스케일바 길이(pixels): 1 mm ≈ {int(1/mm_per_pixel)} px")
                st.write(f"• 프로토네마 실제 면적 추정: {area_in_mm2:.4f} mm²")
            else:
                st.write(f"• 프로토네마 픽셀 수: {protonema_pixels}")
                st.write(f"• 전체 픽셀 수: {total_pixels}")
                st.write(f"• 프로토네마 비율: {protonema_ratio:.2f}%")
                st.warning("스케일바를 찾지 못했습니다. (면적 계산 불가)")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                        caption="원본 이미지",
                        use_column_width=True)
            with col2:
                st.image(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB),
                        caption="K-means 분할된 이미지",
                        use_column_width=True)
            with col3:
                st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                        caption="배경 제외 후",
                        use_column_width=True)

if __name__ == "__main__":
    main()