# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 16:13:55 2024

@author: Lenovo
"""

import cv2
import mediapipe as mp

#kamera kaydı açma
cap =cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

#Bu kodlar, MediaPipe kütüphanesini kullanarak yüz tanıma işlemleri gerçekleştirmek için gerekli modülleri içerir ve bu modülleri kullanmaya hazır hale getirir. Bu kodlar genellikle yüz tespiti ve çizim işlemleri gibi görsel tanıma uygulamalarında kullanılır.

mpFaceDetection=mp.solutions.face_detection #mp.solutions.face_detection modülü, MediaPipe kütüphanesinde bulunan yüz tanıma özelliğini sağlar. Bu modül, yüz tanıma modelini yükler ve görüntülerdeki yüzleri tespit etmek için kullanılır.
#mpFaceDetection=mp.solutions.face_detection kodu, mp.solutions.face_detection modülünü mpFaceDetection adında bir değişkene atar. Bu şekilde, bu modülü daha sonra kolayca kullanabilirsiniz.

faceDetection=mpFaceDetection.FaceDetection() #yüz tanıma modelini başlatır. Bu kod, yüz tespiti yapmak için varsayılan yapılandırmalarla bir yüz tespit modeli oluşturur. Bu model, ardından görüntülerdeki yüzleri tespit etmek için kullanılabilir.

myDraw=mp.solutions.drawing_utils #ediaPipe kütüphanesinin çizim yardımcı programlarını içeren drawing_utils modülünü myDraw adlı bir değişkene atar. Bu modül, tespit edilen yüzleri veya diğer nesneleri çizmek için kullanılabilir.


while True: #Bu döngü, programın sonsuza kadar çalışmasını sağlar ve video akışından sürekli olarak kareler alınmasını sağlar.
    success,img=cap.read() # Bu satır, bir sonraki kareyi başarıyla alıp alamadığınızı kontrol eder ve kareyi img değişkenine atar.
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #Bu satır, OpenCV'nin varsayılan BGR (Mavi, Yeşil, Kırmızı) renk uzayından RGB (Kırmızı, Yeşil, Mavi) renk uzayına dönüşüm yapar.
    results=faceDetection.process(imgRGB) #Bu satır, yüz tespitini gerçekleştirmek için önceden eğitilmiş bir model (muhtemelen MediaPipe Face Detection modeli) kullanarak imgRGB görüntüsünü işler.
    
    num_faces=0 #sayaç
    
    print(results.detections) #Bu satır, tespit edilen yüzlerin bilgilerini içeren results değişkenini yazdırır.
    
    if results.detections: #Bu satır, en az bir yüz tespit edilmişse içeriye girer.
        for id, detection in enumerate(results.detections): #Bu satır, tespit edilen her yüz için bir döngü başlatır ve yüzlerin indislerini (id) ve tespitlerini (detection) alır.
            bboxC=detection.location_data.relative_bounding_box #Bu satır, tespit edilen yüzün sınırlayıcı kutusunu (bounding box) bboxC değişkenine atar. Bu sınırlayıcı kutu, yüzün görüntü içindeki konumunu ve boyutunu tanımlar.
            #print(bboxC)
            h, w,_=img.shape #img.shape: Bu satır, img görüntüsünün yüksekliği (h) ve genişliği (w) bilgisini alır.
            bbox=int(bboxC.xmin*w), int(bboxC.ymin*h),int(bboxC.width*w),int(bboxC.height*h) # Bu satır, sınırlayıcı kutunun piksel cinsinden koordinatlarını hesaplar.
            #print(bbox)
            cv2.rectangle(img, bbox, (0,255,255),2) #Bu satır, sınırlayıcı kutu çevresine bir dikdörtgen çizer. img görüntüsünde, bbox koordinatlarına sahip bir dikdörtgen çizilir. (0, 255, 255) üçlüsü, çizilen dikdörtgenin rengini temsil eder (mavi-yeşil renk tonu). Son olarak, 2, çizginin kalınlığını belirtir.
            
            num_faces +=1 #yüz tespit edildikçe sayaç artırılır
            
    cv2.putText(img, f"num faces: {num_faces}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == ord('r'):
        break

cap.release()
cv2.destroyAllWindows()
    