
#%%
import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
#%%
def find_scale_bar(img, expected_bar_width_mm=1.0):
    """
    아주 단순하게 이미지 하단 우측 영역에서 스케일바(검은 막대)를 찾고
    그 길이(픽셀)를 반환하는 함수.
    - 실제 이미지에 따라 전처리가 달라질 수 있음.
    """
    h, w = img.shape[:2]
    
    # 우측 하단 영역 ROI (예: 높이 하단 80px, 너비 우측 200px 정도)
    roi = img[h-80:h, w-200:w].copy()
    
    # 그레이 변환 & 이진화
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    # 검정 막대가 희미하거나, 다른 요소들과 섞여 있을 수 있으니
    # 필요하다면 모폴로지 연산으로 노이즈 제거
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # 컨투어 찾기
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_width = 0
    for cnt in contours:
        x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
        # 막대는 가로로 길다고 가정
        if w_cnt > max_width:
            max_width = w_cnt
    
    # max_width 픽셀이 현재 1.0 mm에 해당한다고 가정
    if max_width > 0:
        mm_per_pixel = expected_bar_width_mm / max_width
    else:
        mm_per_pixel = None
    
    return mm_per_pixel

def segment_protonema_kmeans(img, k=3, attempts=10, known_protonema_bgr=(204, 216, 152)):
    """
    K-means로 이미지 색상을 군집화하여 프로토네마를 분리하는 함수
    """
    # LAB 변환
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    pixel_vals = lab.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)
    
    # K-means 실행
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
    
    # 클러스터링 결과 이미지
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(img.shape)
    
    # 프로토네마 클러스터 찾기 (임의의 대표값과 가장 가까운 클러스터)
    known_protonema_lab = cv2.cvtColor(np.uint8([[known_protonema_bgr]]), cv2.COLOR_BGR2LAB)[0,0,:]
    distances = [np.linalg.norm(center - known_protonema_lab) for center in centers]
    protonema_cluster_idx = np.argmin(distances)
    
    # 마스크 생성
    mask = (labels == protonema_cluster_idx).reshape(img.shape[:2])
    mask = (mask * 255).astype(np.uint8)
    
    # 노이즈 제거
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 결과 시각화
    result_img = img.copy()
    result_img[mask > 0] = [0, 255, 0]  # 초록색 칠하기
    
    return mask, segmented_image, result_img

def main():
    st.title("protonema segmentation with K-means")
    st.write("여러 이미지를 선택하면, 각 이미지를 K-means로 분할한 뒤, 스케일바를 추정하여 실제 면적을 계산합니다.")

    # 사이드바 파라미터
    k = st.sidebar.slider("클러스터 개수 (k)", 2, 5, 3)
    attempts = st.sidebar.slider("K-means 반복 횟수 (attempts)", 1, 20, 10)

    uploaded_files = st.file_uploader("이미지 파일들을 여러 개 선택하세요", type=['jpg', 'jpeg', 'png', 'tif', 'tiff'], accept_multiple_files=True)

    if uploaded_files is not None and len(uploaded_files) > 0:
        for uploaded_file in uploaded_files:
            # 파일명 표시
            st.subheader(f"파일명: {uploaded_file.name}")
            
            # 이미지 읽기
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img is None:
                st.error("이미지를 읽을 수 없습니다.")
                continue
            
            # 스케일바에서 mm_per_pixel 추정
            mm_per_pixel = find_scale_bar(img, expected_bar_width_mm=1.0)
            
            # 세그멘테이션 수행
            mask, segmented, result_img = segment_protonema_kmeans(img, k, attempts)
            
            # 프로토네마 면적 계산
            protonema_pixels = cv2.countNonZero(mask)
            total_pixels = img.shape[0] * img.shape[1]
            protonema_ratio = (protonema_pixels / total_pixels) * 100
            
            # mm_per_pixel이 성공적으로 계산되었다면 실제 면적(㎟) 환산
            if mm_per_pixel:
                # 픽셀 하나의 면적 = (mm_per_pixel)^2 ㎟
                area_in_mm2 = protonema_pixels * (mm_per_pixel ** 2)
                st.write(f"• 프로토네마 픽셀 수: {protonema_pixels}")
                st.write(f"• 전체 픽셀 수: {total_pixels}")
                st.write(f"• 프로토네마 비율: {protonema_ratio:.2f}%")
                st.write(f"• 추정 스케일바 길이(pixels): 1 mm ≈ {int(1/mm_per_pixel)} px")
                st.write(f"• 프로토네마 실제 면적 추정: {area_in_mm2:.4f} mm²")
            else:
                # 스케일바 검출 실패 시
                st.write(f"• 프로토네마 픽셀 수: {protonema_pixels}")
                st.write(f"• 전체 픽셀 수: {total_pixels}")
                st.write(f"• 프로토네마 비율: {protonema_ratio:.2f}%")
                st.warning("스케일바를 찾지 못했습니다. (면적 계산 불가)")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="원본 이미지", use_column_width=True)
            with col2:
                st.image(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB), caption="분할된 이미지", use_column_width=True)
            with col3:
                st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="프로토네마 검출 결과", use_column_width=True)
#%%
if __name__ == "__main__":
    main()