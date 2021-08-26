import predict_funcs as f
from openpyxl import load_workbook
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow.keras
import numpy as np
import os


if __name__ == '__main__':
    os.chdir(r'C:\Users\David Lee\Desktop\science fair bs\Testing')
    model = tensorflow.keras.models.load_model('keras_model1+3.h5')
    wb = load_workbook(filename = 'Book1.xlsx')
    ws = wb.worksheets[0]
    for name in range(1,101):
        path = str(name) +'.jpg'    

        img = Image.open(path)
        img_array, img = f.image_to_array(img)

        np.set_printoptions(suppress=True)
        prediction,prob = f.predict_single(img_array,model)

    # predt_OneImg()

        for i in range(0,prob.shape[1]):
            print('Class{}:'.format(i)+ '{:.3f}'. format(prob[0][i]))
        print(' ')
        f.show_plot(img, prediction,name)

        f.to_excel(wb, ws, name, prediction[0],prob)


