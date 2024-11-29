from ultralytics import YOLO

# 事前学習済みモデルをロード
model = YOLO('yolo11n.pt')  # yolo11n.pt の場所を指定

# トレーニングデータセットを指定
if __name__ == '__main__':
    model.train(
        data='C:\Users\hojo0\OneDrive\program\Python\paper QQQQQ.v3i.yolov11\data.yaml',  # データセットの設定ファイル (下記参照)
        epochs=100,                         # エポック数
        imgsz=640,                         # 入力画像サイズ
        batch=16,                          # バッチサイズ
        device=0                           # GPU (0 = GPUを使用, 'cpu' = CPU)
    )