import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import math
import os

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def main():
    # Carregar modelo principal para rastrear (detecta tudo)
    model = YOLO("yolov8n.pt")
    
    # Carregar modelo de pose para o esqueleto (substitui MediaPipe)
    pose_model = YOLO("yolov8n-pose.pt")
    
    # Video Setup
    video_path = "robot runing.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return

    # Video Writer setup
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    os.makedirs('resultados', exist_ok=True)
    out_path = 'resultados/robot_human_tracking.mp4'
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # Histórico para os rastros
    track_history = defaultdict(lambda: [])
    
    # IDs de rastreamento mapeados para 'Humano' ou 'Robo'
    id_classification = {}
    
    frame_counter = 0
    collision_threshold = 100 # pixels de distância para considerar colisão
    
    # Criar uma janela redimensionável para não estourar a resolução da tela
    cv2.namedWindow("Tracking do Robo e Humano", cv2.WINDOW_NORMAL)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame_counter += 1
        
        # Executar rastreamento com YOLO usando ByteTrack (imgsz=320 para mais velocidade)
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, imgsz=320)
        
        human_center = None
        robot_center = None
        
        # Obter os resultados do rastreamento
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.int().cpu().numpy()
            
            for box, track_id, conf, cls in zip(boxes, track_ids, confidences, classes):
                x1, y1, x2, y2 = map(int, box)
                
                # Filtrar caixas muito pequenas que podem ser ruído
                if (x2 - x1) * (y2 - y1) < 1500:
                    continue
                    
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                track_history[track_id].append(center)
                
                # Limitar o tamanho do rastro
                if len(track_history[track_id]) > 30:
                    track_history[track_id].pop(0)
                    
                # Diferenciação usando YOLO-Pose
                crop_y1, crop_y2 = max(0, y1), min(height, y2)
                crop_x1, crop_x2 = max(0, x1), min(width, x2)
                
                label = id_classification.get(track_id, "Desconhecido")
                
                if crop_y2 > crop_y1 and crop_x2 > crop_x1:
                    crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    
                    if track_id not in id_classification or label == "Humano":
                        # Testar pose na área recortada (imgsz=160 pois é só um recorte, muito mais rápido)
                        pose_results = pose_model(crop_img, verbose=False, imgsz=160)
                        has_skeleton = False
                        
                        if pose_results[0].keypoints is not None:
                            kp_conf = pose_results[0].keypoints.conf
                            # Se encontrou keypoints com confiança média aceitável
                            if kp_conf is not None and kp_conf.numel() > 0 and kp_conf[0].mean() > 0.4:
                                has_skeleton = True
                        
                        if has_skeleton:
                            id_classification[track_id] = "Humano"
                            label = "Humano"
                            
                            # Obter a imagem com o esqueleto desenhado (sem as caixas extra)
                            annotated_crop = pose_results[0].plot(boxes=False, labels=False)
                            # Colocar o recorte desenhado de volta no frame original
                            frame[crop_y1:crop_y2, crop_x1:crop_x2] = annotated_crop
                        elif track_id not in id_classification:
                            id_classification[track_id] = "Robo"
                            label = "Robo"
                
                color = (0, 255, 0) if label == "Humano" else (0, 0, 255) if label == "Robo" else (255, 255, 255)
                
                if label == "Humano":
                    human_center = center
                elif label == "Robo":
                    robot_center = center
                    
                # Desenhar Bounding Box e Label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Desenhar rastro (trail)
                points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)
                
        # Calcular Distância e Colisão
        if human_center and robot_center:
            dist = calculate_distance(human_center, robot_center)
            

            
            # Alerta de Colisão
            if dist < collision_threshold:
                alert_text = "COLISAO DETECTADA!"
                (tw, th), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
                alert_x = (width - tw) // 2
                alert_y = 100
                cv2.putText(frame, alert_text, (alert_x, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                

        
        # Exibir na tela (opcional)
        cv2.imshow("Tracking do Robo e Humano", frame)
        
        # Salvar no vídeo
        out.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processamento concluido! Video salvo em: {out_path}")

if __name__ == "__main__":
    main()
