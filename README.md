# Histogram_of_Oriented_Gradients
Histogram of Oriented Gradients (HOG) 方向梯度直方圖

紀錄一下在學校修數位影像處理這門課的期末報告,本來想要用英文寫出來,但是本人英文程度沒有很好,覺得用英文硬寫出來,母語不是英文的人看不懂以外,母語為英文的人可能也看不懂我的單字跟文法(台式英文),既然這樣想說就先用中文寫,等自己英文有點程度後再加上英文也不遲(反正也沒甚麼人會看)

## 編寫環境
使用Anaconda![](https://i.imgur.com/2wDYM8c.png)
的Spyder![](https://i.imgur.com/cCOap9R.png)執行程式

* Python版本為 3.8.8

* Opencv版本為 4.0.1

Anaconda環境檔案 -> hog.yaml

匯入環境檔可上網查詢 **Anaconda環境匯入**

## 輸入/輸出
* 輸入: input_image_path(輸入影像路徑)

影像(跑者) 建議輸入形狀長寬比為2:1

* 輸出: hog_vector

1X3780的特徵向量

```python
if __name__ == '__main__': 

    input_image_path = (r'影像路徑\檔名.png')
    
    #執行Histogram_of_Oriented_Gradients 
    hog_vector = Histogram_of_Oriented_Gradients()
    #輸出為hog_vector
    
```

## 執行步驟
* 第一步-Image_Pretreatment影像預處理

影像灰階化

調整影像大小(此程式調整為64 * 128大小)

其他處理(可做可不做)
1. 強度除以影像最大值
2. 強度除以255
3. gamma函式

* 第二步-計算Sobel

這部分使用opencv處理

**這部分可以上網查更詳細的步驟**

```python=83
gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)
gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)

mag, angle = cv2.cartToPolar(gradient_values_x, gradient_values_y, angleInDegrees=True)
    
#angle範圍會為0~360之間,但我們只要0~180度
for i in range(angle.shape[0]):
    for j in range(angle.shape[1]):
        if(angle[i][j] > 180):
            angle[i][j] = angle[i][j] - 180
```

* 第三步-計算Cell

設定Cell大小為8 * 8,步長為8,把輸入且做完預處理的影像用Cell分割,會得到8 * 16個Cell

設定bin為1 * 9 的格子

0度到180度之間平分九等分

| 0 | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160 |

將每個Cell內的angle對應的mag存放在bin裡面(依照比例放)
(0度跟180度的放在| 0 |裡面)

**這部分可以上網查更詳細的步驟**
```python=69
cell_gradient_box = np.zeros(((int)(128 / cell_size), (int)(64 / cell_size), bin_size)) 
#輸入為128*64大小的影像應該會被分為->(16,8,9)
for i in range(cell_gradient_box.shape[0]): #先算cell左上角的格子的直方圖，左至右、上到下
    for j in range(cell_gradient_box.shape[1]):
        #找第0格~第8格的幅值
        cell_mag = mag[i * cell_size:(i + 1) * cell_size,j * cell_size:(j + 1) * cell_size]
        #找第0格~第8格的角度值
        cell_angle = angle[i * cell_size:(i + 1) * cell_size,j * cell_size:(j + 1) * cell_size]
        #計算直方圖
        cell_gradient_box[i][j] = Cell_Gradient(cell_mag, cell_angle)
            
```

* 第四步-計算Block 第五步-串接Block

設定Block大小為四個Cell組合成正方形的大小 (16 * 16) 步長為8,所以會有(64/8)-1) * ((128/8)-1) = 7 * 15 = 105個Block

將Block內的四個Cell資料依序左上、右上、左下、右下串接成長度為36的特徵向量

對特徵向量做L2norm處理(36個資料 除以 36個資料平方後相加再開根號的數值)

再將每個Block依序串接可得長度為(bin數) * (Block內的Cell數) * ((64/8)-1) * ((128/8)-1) = 9 * 4 * 7 * 15 = 3780


```python
k=0
hog_vector = np.zeros((bin_size*4*(cell_gradient_box.shape[0] - 1)*(cell_gradient_box.shape[1] - 1)))
for i in range(cell_gradient_box.shape[0] - 1):
    for j in range(cell_gradient_box.shape[1] - 1):
        histogram_block = np.concatenate([cell_gradient_box[i][j],cell_gradient_box[i][j + 1],cell_gradient_box[i+1][j],cell_gradient_box[i+1][j + 1]])           
            
        #做L2範數           
        L2_norm = histogram_block * histogram_block
        L2_norm = L2_norm.sum()
        L2_norm = np.power(L2_norm,0.5)
        extre_min = np.power(0.0001,2) #創一個極小值 怕L2_norm為零 分母為零
        L2_norm = L2_norm + extre_min
            
        histogram_block = histogram_block / L2_norm
           
        #把histogram_block串接起來
        hog_vector[36*k : 36*(k+1)] = histogram_block
        k=k+1
```
---

## 整個步驟流程

```python
def Histogram_of_Oriented_Gradients():
    #讀取灰階圖
    img = cv2.imread(input_image_path,0)
    
    #Image_Pretreatment影像預處理
    img_finshed = Image_Pretreatment(img)
    
    #計算Sobel
    mag, angle = Compute_Sobel(img_finshed)
    
    #計算Cell
    cell_gradient_box = Computer_Cell(mag, angle)
    
    #計算Block
    hog_vector = Compute_Block(cell_gradient_box)
    
    return hog_vector
```

## 結束