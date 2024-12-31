# MMT_final
## 功能
  依照遊戲截圖中的UI分辨是來自哪款遊戲的截圖
## 實作
### Training side
  1. 一次將同個遊戲的數個不同截圖（必需是同類型畫面，如主頁面、戰鬥畫面、角色頁面等）集合成一個Training data set並輸入標籤（如：「遊戲：Master Duel, 畫面：主頁」）
  2. 程式偵測出這些截圖中重疊的相同部份（即UI）
  3. 根據2. 的結果生成一張Mask和一張UI Image，與標籤一起儲存
### Detecting side
  1. 使用者輸入一張截圖
  2. 程式將輸入套上各個Mask後與UI Image比對
  3. 輸出相似度最高的結果（如：這張是 Master Duel 的 主頁 的截圖）
## 用法
### Training side
  1. 將一系列用於訓練的截圖放到「training_input」中
  2. 執行
'''
detector.train_from_folder("training_input", <遊戲名稱（如"Arknights"）>, <頁面名稱（如"Home"或"Battle"）>)
'''
### Detecting side
  1. 將一系列想辨識的截圖放到「detecting_input」中
  2. 在main_part執行detector.load_train_results()
  3. 執行
'''
detecting_input_dir = "detecting_input"
detecting_images = [os.path.join(detecting_input_dir, fname) for fname in os.listdir(detecting_input_dir) if fname.endswith(('.png', '.jpg', '.jpeg'))]

if detecting_images:
    for image_path in detecting_images:  # Iterate through all images in the folder
        result = detector.detect(image_path)
        print(f"Image: {image_path}, Detected: {result}")
else:
    print("No images found in detecting_input directory.")
'''
