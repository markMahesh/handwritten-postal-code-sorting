print("Hello World")
from tkinter import *
import cv2,os,shutil
import numpy as np
from PIL import ImageGrab, Image, ImageTk
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model
from geopy.geocoders import Nominatim
 
# Using Nominatim Api
geolocator = Nominatim(user_agent="geoapiExercises")


model = load_model('saveModel.h5')
image_folder = "img/"

root = Tk()
root.resizable(0, 0)
root.title("Handwritten Multiple Digit Recognition")

try : 
    shutil.rmtree("img")
except:
    pass

os.mkdir("img")

lastx, lasty = None, None
image_number = 0

# cv = Canvas(root, width=640, height=480, bg='white')
cv = Canvas(root, width=1200, height=600, bg='white')
cv.grid(row=0, column=0, pady=2, sticky=W, columnspan=3)



def clear_widget():
    global cv
    cv.delete('all')
    Label_num.config(text='PinCode = ')
    Label_num2.config(text='Location = ')


def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=3, fill='black', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y


def activate_event(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y


cv.bind('<Button-1>', activate_event)


def Recognize_Digit():
    global image_number
    filename = f'img_{image_number}.png'

    ans = 0
    widget = cv

    x = root.winfo_rootx() + widget.winfo_rootx()
    y = root.winfo_rooty() + widget.winfo_rooty()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()
    # print(x, y, x1, y1)

    # get image and save
    ImageGrab.grab().crop((x, y, x1, y1)).save(image_folder + filename)

    image = cv2.imread(image_folder + filename, cv2.IMREAD_COLOR)
    # image = cv2.imread('pinCode2.jpeg',cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    # contours = sorted(contours)
    # print(contours)
    dic = {}
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        print(x)
        # make a rectangle box around each curve
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)

        # Cropping out the digit from the image corresponding to the current contours in the for loop
        digit = th[y:y + h, x:x + w]

        # Resizing that digit to (18, 18)
        resized_digit = cv2.resize(digit, (18, 18))

        # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

        digit = padded_digit.reshape(1, 28, 28, 1)
        digit = digit / 255.0

        pred = model.predict([digit])[0]
        final_pred = np.argmax(pred)
        dic[x] = final_pred
        # ans+=str(final_pred)
        data = str(final_pred) + ' ' + "{:.2f}".format(max(pred) * 100)+ '%'
        # print(data)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 1
        cv2.putText(image, data, (x, y - 5), font, fontScale, color, thickness)

    im = Image.fromarray(image)
    root.one = one = ImageTk.PhotoImage(image=im)
    cv.create_image(0, 0, image=one, anchor=NW)

    cv2.imwrite(os.path.join(image_folder , "img_"+str(image_number)+"_pred.png"),image)
    image_number+=1
    # for val in dic.values():
    #     ans+=str(val)
    for i in sorted (dic) :
        ans = ans*10 + (dic[i])
    print(dic)
    Label_num.config(text=("Number: "+str(ans)))
    Label_num2.config(text=("Location: "+str(geolocator.geocode(ans))))
    cv2.waitKey(0)

def Video_Recognize_Digit():
    video=cv2.VideoCapture(0)
    a=1 
    while True:
        a+=1
        check ,frame=video.read()
        #print(frame)
        frame = cv2.resize(frame,(600,600))
        # gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        # contours = sorted(contours)
        # print(contours)
        dic = {}
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            print(x)
            # make a rectangle box around each curve
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

            # Cropping out the digit from the image corresponding to the current contours in the for loop
            digit = th[y:y + h, x:x + w]

            # Resizing that digit to (18, 18)
            resized_digit = cv2.resize(digit, (18, 18))

            # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
            padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

            digit = padded_digit.reshape(1, 28, 28, 1)
            digit = digit / 255.0

            pred = model.predict([digit])[0]
            final_pred = np.argmax(pred)
            dic[x] = final_pred
            # ans+=str(final_pred)
            data = str(final_pred) + ' ' + "{:.2f}".format(max(pred) * 100)+ '%'
            # print(data)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (255, 0, 0)
            thickness = 1
            cv2.putText(frame, data, (x, y - 5), font, fontScale, color, thickness)

        im = Image.fromarray(frame)
        root.one = one = ImageTk.PhotoImage(image=im)
        cv.create_image(0, 0, image=one, anchor=NW)

        # cv2.imwrite(os.path.join(image_folder , "img_"+str(image_number)+"_pred.png"),image)
        # image_number+=1
        # for val in dic.values():
        #     ans+=str(val)
        for i in sorted (dic) :
            ans+=str(dic[i])
        print(dic)
        Label_num.config(text=("Number: "+ans))
        
        # faces=face_cascade.detectMultiScale(gray,1.1,3)
        # for x,y,w,h in faces:
        #     cv2.rectangle(frame, (x,y),(x+w , y+h), (255,0,0),3)
        cv2.imshow("capturing",frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
            break
    print("no. of frames captured :",a)
    video.release()
    cv2.destroyAllWindows()
    



btn_save = Button(text='Get PinCode', command=Recognize_Digit)
btn_save.grid(row=2, column=0, pady=1, padx=1)
btn_save = Button(text='Video Capture', command=Video_Recognize_Digit)
btn_save.grid(row=2, column=1, pady=1, padx=1)
button_clear = Button(text='Clear', command=clear_widget)
button_clear.grid(row=2, column=3, pady=1, padx=1)
Label_num = Label(text='PinCode = ')
Label_num.grid(row=2, column=2, pady=1, padx=1)
Label_num2 = Label(text='Location =')
Label_num2.grid(row=1, column=1, pady=1, padx=1)

root.mainloop()
