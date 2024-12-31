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
