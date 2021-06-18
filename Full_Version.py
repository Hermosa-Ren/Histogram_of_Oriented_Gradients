import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
def Compute_Block(cell_gradient_box):
    k=0
    hog_vector = np.zeros((bin_size*4*(cell_gradient_box.shape[0] - 1)*(cell_gradient_box.shape[1] - 1)))
    for i in range(cell_gradient_box.shape[0] - 1):
        for j in range(cell_gradient_box.shape[1] - 1):
            histogram_block = np.concatenate([cell_gradient_box[i][j],cell_gradient_box[i][j + 1],cell_gradient_box[i+1][j],cell_gradient_box[i+1][j + 1]])
            
            #顯示圖片
            #x = np.arange(1,37,1)
            #plt.title('histogram_block')
            #plt.bar(x,histogram_block)
            #plt.savefig(r'路徑\檔名.png')
            #plt.show()
            
            #做L2範數           
            L2_norm = histogram_block * histogram_block
            L2_norm = L2_norm.sum()
            L2_norm = np.power(L2_norm,0.5)
            extre_min = np.power(0.0001,2) #用一個極小值 怕L2_norm為零 分母為零
            L2_norm = L2_norm + extre_min
            #歸一化
            histogram_block = histogram_block / L2_norm
            
            #顯示圖片
            #x = np.arange(1,37,1)
            #plt.title('histogram_block_L2')
            #plt.bar(x,histogram_block)
            #plt.savefig(r'路徑\檔名.png')
            #plt.show()
            
            #把histogram_block串接起來
            hog_vector[36*k : 36*(k+1)] = histogram_block
            k=k+1
            
    return hog_vector
#計算直方圖
def Cell_Gradient(cell_mag, cell_angle):
    histogram_cell = np.zeros(bin_size)  # 0 20 40 60 80 100 120 140 160   
    for k in range(cell_size):
        for l in range(cell_size):
            cell_mag_catch = cell_mag[k][l] #讀取[0,0]幅值
            cell_angle_catch = cell_angle[k][l]#讀取[0,0]角度值
            if(cell_angle_catch % 20 == 0): #如果角度是0 20 40 60 80 100 120 140 160 180直接丟值進去
                bin_number = int(cell_angle_catch / 20) % bin_size #有%bin_size是因為180要丟進0的裡面設計的
                histogram_cell[bin_number] += cell_mag_catch
            
            else:#其他角度要將幅值分配
                bin_number_small = int(cell_angle_catch / 20) % bin_size    
                bin_number_big = (bin_number_small + 1) % bin_size #有%bin_size是因為假如bin_number_small為8的話再加1會變9也就是要放進第0格裡面
                ratio = cell_angle_catch % 20 #依照比例丟進bin_number_small與bin_number_big
                histogram_cell[bin_number_small] += (cell_mag_catch * (1 - (ratio / 20)))
                histogram_cell[bin_number_big] += (cell_mag_catch * (ratio / 20))  
            
            #顯示直方圖          
            #x = np.arange(0,180,20)
            #plt.xlabel("angle")
            #plt.ylabel("mag")
            #plt.title("Histogram of Gradient")
            #plt.bar(x,histogram_cell,width = 3)
            #plt.savefig(r'路徑\檔名.png')
            #plt.show()
            
    return histogram_cell
def Computer_Cell(mag, angle):
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
            
    return cell_gradient_box

def Compute_Sobel(img):
    gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)
    gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)

    mag, angle = cv2.cartToPolar(gradient_values_x, gradient_values_y, angleInDegrees=True)
    
    #angle範圍會為0~360之間,但我們只要0~180度
    for i in range(angle.shape[0]):
        for j in range(angle.shape[1]):
            if(angle[i][j] > 180):
                angle[i][j] = angle[i][j] - 180
    
    '''
    #顯示Compute_Sobel後的影像
    while True:
        abs_x = abs(gradient_values_x)
        abs_x = np.uint8(abs_x)
        cv2.namedWindow("gradient_values_x",0)
        cv2.resizeWindow("gradient_values_x", 256, 512)
        cv2.imshow("gradient_values_x",abs_x)
        
        abs_y = abs(gradient_values_y)
        abs_y = np.uint8(abs_y)
        cv2.namedWindow("gradient_values_y",0)
        cv2.resizeWindow("gradient_values_y", 256, 512)
        cv2.imshow("gradient_values_y",abs_y)
        
        mag_uint8 = np.uint8(mag)
        cv2.namedWindow("mag",0)
        cv2.resizeWindow("mag", 256, 512)
        cv2.imshow("mag",mag_uint8)
        
        k = cv2.waitKey(0)
        if k == 27:
            #按Esc
            cv2.destroyAllWindows()
            break
    '''
    
    return mag, angle
#Image_Pretreatment影像預處理
def Image_Pretreatment(img):
    
    #resize調整大小
    img_resize = cv2.resize(img, (64,128), interpolation=cv2.INTER_CUBIC)
    img_resize_32 = np.float32(img_resize)
    '''
    #顯示影像
    cv2.namedWindow("Resize",0)
    cv2.resizeWindow("Resize", 256, 512)
    cv2.imshow("Resize",img_resize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    
    #預處理(一)強度除以最大值
    #img_after = (img_resize_32/np.max(img_resize_32))
    
    #預處理(二)強度除以255
    #img_after = (img_resize_32/255)
    
    #預處理(三)gamma函式
    #img_after = np.power(img_resize_32,0.9)
    
    '''
    img_after_uint8 = np.uint8(img_after)
    cv2.namedWindow("img_after",0)
    cv2.resizeWindow("img_after", 256, 512)
    cv2.imshow("img_after", img_after_uint8)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    
    #return img_after
    return img_resize_32
   

#Histogram_of_Oriented_Gradients梯度方向直方圖
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

if __name__ == '__main__':
    #讀取當前資料夾位置   
    #input_image_path = (r'路徑\檔名.png')
    
    this_file_path = os.getcwd()
    #input_image_path = (r'{}\running_man_1.png'.format(this_file_path))
    input_image_path = (r'{}\running_man_2.png'.format(this_file_path))
    #input_image_path = (r'{}\running_man_3.png'.format(this_file_path))
    #input_image_path = (r'{}\running_man_4.png'.format(this_file_path))
    #input_image_path = (r'{}\running_man_5.png'.format(this_file_path))
    #input_image_path = (r'{}\running_man_6.png'.format(this_file_path))
    #input_image_path = (r'{}\running_man_7.png'.format(this_file_path))
    #input_image_path = (r'{}\landscape.png'.format(this_file_path))
    
    #參數
    bin_size = 9
    cell_size = 8
    
    #執行程式
    hog_vector = Histogram_of_Oriented_Gradients() #輸出為hog_vector
    
    #print輸出長度
    print ("輸出HOG長度為{}".format(hog_vector.shape[0]))
    
    #將HOG輸出向量可視化    
    x = np.arange(hog_vector.shape[0])
    plt.title('HOG')
    plt.bar(x,hog_vector,color='red') 
    #plt.savefig(r'{}\running_man_1_result.png'.format(this_file_path))
    #plt.savefig(r'路徑\檔名.png')
    plt.show()