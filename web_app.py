
#%%
import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
#%%
def segment_protonema_kmeans(uploaded_file, k=3, attempts=10):
    """
    K-means로 이미지 색상을 군집화하여 프로토네마를 분리하는 함수
    """
    # 업로드된 파일을 numpy 배열로 변환
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        st.error("이미지를 읽을 수 없습니다.")
        return None
    
    # LAB 변환
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    pixel_vals = lab.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)
    
    # K-means 실행
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
    
    # 클러스터링 결과 처리
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(img.shape)
    
    # 프로토네마 클러스터 찾기
    known_protonema_bgr = np.uint8([[[204, 216, 152]]])
    known_protonema_lab = cv2.cvtColor(known_protonema_bgr, cv2.COLOR_BGR2LAB)[0,0,:]
    
    distances = [np.linalg.norm(center - known_protonema_lab) for center in centers]
    protonema_cluster_idx = np.argmin(distances)
    
    # 마스크 생성
    mask = (labels == protonema_cluster_idx).reshape(img.shape[:2])
    mask = (mask * 255).astype(np.uint8)
    
    # 노이즈 제거
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 면적 계산
    total_pixels = img.shape[0] * img.shape[1]
    protonema_area = cv2.countNonZero(mask)
    protonema_percentage = (protonema_area / total_pixels) * 100
    
    # 결과 시각화
    result_img = img.copy()
    result_img[mask > 0] = [0, 255, 0]
    
    return {
        'original': cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        'segmented': cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB),
        'mask': mask,
        'result': cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
        'percentage': protonema_percentage
    }
#%%
def main():
    st.title("프로토네마 세그멘테이션 웹 앱")
    st.write("이미지를 업로드하면 K-means 클러스터링으로 프로토네마 영역을 분리합니다.")
    
    # 사이드바 파라미터
    k = st.sidebar.slider("클러스터 개수 (k)", 2, 5, 3)
    attempts = st.sidebar.slider("K-means 반복 횟수", 1, 20, 10)
    
    # 파일 업로더
    uploaded_file = st.file_uploader("이미지 파일 선택", type=['jpg', 'jpeg', 'png', 'tif', 'tiff'])
    
    if uploaded_file is not None:
        # 이미지 처리
        results = segment_protonema_kmeans(uploaded_file, k, attempts)
        
        if results:
            # 결과 표시
            col1, col2 = st.columns(2)
            with col1:
                st.image(results['original'], caption="원본 이미지")
                st.image(results['segmented'], caption="분할된 이미지")
            with col2:
                st.image(results['mask'], caption="프로토네마 마스크")
                st.image(results['result'], caption="검출 결과")
            
            # 분석 결과
            st.write(f"프로토네마 비율: {results['percentage']:.2f}%")
#%%
if __name__ == "__main__":
    main() 
