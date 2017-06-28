from flask import Flask, request
from flask import json
from flask import send_file
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps
import numpy as np
from PIL import Image
from numpy.fft import fft2
import math
from numpy import histogram as nphist
import codecs
import sys
import cv2
from werkzeug import secure_filename
reload(sys)  
sys.setdefaultencoding('utf8')

cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

#Create a engine for connecting to SQLite3.
#Assuming salaries.db is in your app root folder

e = create_engine('sqlite:///salaries.db')

app = Flask(__name__)
api = Api(app)

def INPUT_IMAGE_NAME():
    return "input_image.jpg"

def OUTPUT_IMAGE_NAME():
    return "output_image.jpg"

def save_image(request):
    f0 = request.files['file']
    f0.save(secure_filename(INPUT_IMAGE_NAME()))

def return_image():
    return send_file(OUTPUT_IMAGE_NAME(), mimetype='image/jpeg') 

def add_hat(x,y,w,h, hat_type, background, foreground):
    (width_b, height_b) = background.size   
    (width_f, height_f) = foreground.size

    if(hat_type == "FedoraHat"):
        hat_point_left = (187,314)
        hat_point_right = (638,314)
        hat_real_width = 638-187
    elif (hat_type == "SantaHat"):
        hat_point_left = (18,195)
        hat_point_right = (295,195)
        hat_real_width = 295-18
    elif (hat_type == "Glasses"):
        hat_point_left = (0, 157)
        hat_point_right = (442,157)
        hat_real_width = 442 - 0


    if(hat_type == "FedoraHat" or hat_type == "SantaHat"):
        width_face = w
        proportion_width = (width_face)/(hat_real_width*1.0)
        new_height =(int) (height_f*proportion_width)
        new_width = (int) (width_f*proportion_width);
        new_hat_point_left = ((int)(hat_point_left[0]*proportion_width),(int)(hat_point_left[1]*proportion_width))
        resized_foreground = foreground.resize((new_width,new_height))
        background.paste(resized_foreground, (x - new_hat_point_left[0], y - new_hat_point_left[1] ), resized_foreground)
    elif(hat_type == "Glasses"):
        width_face = w
        proportion_width = (width_face)/(hat_real_width*1.0)
        new_height =(int) (height_f*proportion_width)
        new_width = (int) (width_f*proportion_width);
        new_hat_point_left = ((int)(hat_point_left[0]*proportion_width),(int)(hat_point_left[1]*proportion_width))
        resized_foreground = foreground.resize((new_width,new_height))
        background.paste(resized_foreground, (x - new_hat_point_left[0], y + h/2 + h/10 - new_hat_point_left[1] ), resized_foreground)
    return background

def iacircle(s, r, c):
        
    rows, cols = s[0], s[1]
    rr0,  cc0  = c[0], c[1] 
    rr, cc = np.meshgrid(range(rows), range(cols), indexing='ij')
    g = (rr - rr0)**2 + (cc - cc0)**2 <= r**2
    return g

def iapercentile(f, p=1):
    k = (f.size-1) * p/100.
    dw = np.floor(k).astype(int)
    up = np.ceil(k).astype(int)
    g  = np.sort(f.ravel())
    d  = g[dw]
    d0 =   d   * (up-k)
    d1 = g[up] * (k -dw)
    return np.where(dw==up, d, d0+d1)

def iacos(s, t, theta, phi):
    r, c = np.indices(s)
    tc = t / np.cos(theta)
    tr = t / np.sin(theta)
    f = np.cos(2*3.1415*(r/tr + c/tc) + phi)
    return f

def normalize(f):
    arr = np.array(f)
    arr = arr.astype('float')
    # Do not touch the alpha channel
    minval = arr.min()
    maxval = arr.max()
    if minval != maxval:
        arr -= minval
        arr *= (255.0/(maxval-minval))
    return arr

def ruidoAltura(gray_img):
    height,width = gray_img.size
    s = (width,height)
    t = height/60
    theta = 0.01 * 3.1415/180
    noise_1 = iacos(s, t, theta, 0)
    return Image.blend(Image.fromarray(normalize(gray_img).astype('uint8')), Image.fromarray(normalize(noise_1).astype('uint8')), 0.5)
    #noisy_1 = normalize(gray_img).astype('uint8') + normalize(noise_1).astype('uint8')
    #return Image.fromarray(normalize(noisy_1).astype('uint8'))

def iaidft(F):
    #F = Image.fromarray(img)
    s = F.shape
    if len(F.shape) == 1: F = F[np.newaxis,np.newaxis,:]
    if len(F.shape) == 2: F = F[np.newaxis,:,:] 
    (p,m,n) = F.shape
    A = iadftmatrix(m)
    B = iadftmatrix(n)
    C = iadftmatrix(p)
    Faux = np.dot(np.conjugate(A),F)
    Faux = np.dot(Faux,np.conjugate(B))
    f = np.dot(np.conjugate(C),Faux)/(np.sqrt(p)*np.sqrt(m)*np.sqrt(n))
        
    return f.reshape(s)

def ruidoLargura(gray_img):
    height,width = gray_img.size
    s = (width,height)
    t = height/60
    theta = 90 * 3.1415/180
    noise_2 = iacos(s, t, theta, 0)
    return Image.blend(Image.fromarray(normalize(gray_img).astype('uint8')), Image.fromarray(normalize(noise_2).astype('uint8')), 0.5)
   # noisy_2 = gray_img + normalize(noise_2).astype('uint8'))
   # return Image.fromarray(normalize(noisy_2).astype('uint8'))

def ruidoAmbos(gray_img):
    height,width = gray_img.size
    s = (width,height)
    t = height/60
    theta1 = 0.01 * 3.1415/180
    theta2 = 90 * 3.1415/180
    noise_1 = iacos(s, t, theta1, 0)
    noise_2 = iacos(s, t, theta2, 0)
    noise_3 = normalize(noise_1).astype('uint8') + normalize(noise_2).astype('uint8')
    return Image.blend(Image.fromarray(normalize(gray_img).astype('uint8')), Image.fromarray(normalize(noise_3).astype('uint8')), 0.5)
    #noisy_3 = gray_img + normalize(noise_3).astype('uint8')
    #return Image.fromarray(normalize(noisy_3).astype('uint8'))

def iadftmatrix(N):
    x = np.arange(N).reshape(N,1)
    u = x 
    Wn = np.exp(-1j*2*3.1415/N)
    A = (1./np.sqrt(N)) * (Wn ** np.dot(u, x.T))
    return A    
    
    
def iaptrans(f,t):
    g = np.empty_like(f) 
    if f.ndim == 1:
        W = f.shape[0]
        col = arange(W)
        g[:] = f[(col-t)%W]
    elif f.ndim == 2:
        H,W = f.shape
        rr,cc = t
        row,col = np.indices(f.shape)
        g[:] = f[(row-rr)%H, (col-cc)%W]
    elif f.ndim == 3:
        Z,H,W = f.shape
        zz,rr,cc = t
        z,row,col = np.indices(f.shape)
        g[:] = f[(z-zz)%Z, (row-rr)%H, (col-cc)%W]
    return g


def ianormalize(f, range=[0,255]):
    f = np.asarray(f)
    range = np.asarray(range)
    if f.dtype.char in ['D', 'F']:
        raise Exception('error: cannot normalize complex data')
    faux = np.ravel(f).astype(float)
    minimum = faux.min()
    maximum = faux.max()
    lower = range[0]
    upper = range[1]
    if upper == lower:
        g = np.ones(f.shape) * maximum
    if minimum == maximum:
        g = np.ones(f.shape) * (upper + lower) / 2.
    else:
        g = (faux-minimum)*(upper-lower) / (maximum-minimum) + lower
    g = np.reshape(g, f.shape)   
    return g

def iargb2gray(f):
    width, height = f.size
    gray = Image.new('L', (width, height))
    for x in xrange(width):
        for y in xrange(height):
            r, g, b = f.getpixel((x, y))
            value = r * 299.0/1000 + g * 587.0/1000 + b * 114.0/1000
            value = int(value)
            gray.putpixel((x, y), value)
    return gray    


class Departments_Meta(Resource):
    def post(self):
        save_image(request)
        img = Image.open(INPUT_IMAGE_NAME())
        img = iargb2gray(img)
        imgdft = fft2(img)
        mag = abs(imgdft)
        pct25 = iapercentile(mag , 25)
        pct50 = iapercentile(mag , 50)
        pct75 = iapercentile(mag , 75)
        pct95 = iapercentile(mag , 95)
        filter25 = (mag > pct25)
        filter50 = (mag > pct50)
        filter75 = (mag > pct75)
        filter95 = (mag > pct95)
        filtered95 = imgdft*filter95
        filtered75 = imgdft*filter75
        img75 = Image.fromarray(normalize(np.real(fft2(filtered75))).astype('uint8'))
        #img90 = Image.fromarray(np.asarray(ianormalize(np.real(fft2(filtered95)))))
        img75 = img75.transpose(Image.FLIP_LEFT_RIGHT)
        img75.save('hue.png')
            
        return return_image()

def iadftview(F):
    FM = iafftshift(np.log10(abs(F)+1))
    return normalize(FM).astype('uint8')

def iafftshift(f):
    f = np.asarray(f)
    return iaptrans(f, np.array(f.shape)/2)

class LarguraRuido(Resource):
    def post(self):
        save_image(request)
        img = Image.open(INPUT_IMAGE_NAME())
        img = img.convert('RGB')
        img = iargb2gray(img)
        img = ruidoLargura(img)
        img.save(OUTPUT_IMAGE_NAME())
        return return_image()

class AlturaRuido(Resource):
    def post(self):
        save_image(request)
        img = Image.open(INPUT_IMAGE_NAME())
        img = img.convert('RGB')
        img = iargb2gray(img)
        img = ruidoAltura(img)
        img.save(OUTPUT_IMAGE_NAME())
        return return_image()

class AmbosRuido(Resource):
    def post(self):
        save_image(request)
        img = Image.open(INPUT_IMAGE_NAME())
        img = img.convert('RGB')
        img = iargb2gray(img)
        img = ruidoAmbos(img)
        img.save(OUTPUT_IMAGE_NAME())
        return return_image()

class Mascara(Resource):
    def post(self):
        save_image(request)
        img = Image.open(INPUT_IMAGE_NAME())
        fftimg = fft2(img)
        img = iadftview(fftimg)
        auxImg = Image.fromarray(img)
        width, height = auxImg.size
        mask = iacircle(img.shape,35,[height/2,width/2])
        mask = iaptrans(mask, np.array(mask.shape)/2).astype(bool)
        filtered = fftimg*mask
        img = iadftview(filtered)
        img = Image.fromarray(img)
        img.save(OUTPUT_IMAGE_NAME())
        return return_image()


class Mf(Resource):
    def post(self):
        save_image(request)
        img = Image.open(INPUT_IMAGE_NAME())
        #img = img.convert('RGB')
        #img = iargb2gray(img)
        fftimg = fft2(img)
        img = iadftview(fftimg)
        auxImg = Image.fromarray(img)
        width, height = auxImg.size
        mask = iacircle(img.shape,35,[height/2,width/2])
        mask = iaptrans(mask, np.array(mask.shape)/2).astype(bool)
        filtered = fftimg*mask
        #img = normalize(filtered).astype('uint8')
        img = iaidft(filtered)
        img = normalize(np.abs(img)).astype('uint8')
        img = Image.fromarray(img)
        img.save(OUTPUT_IMAGE_NAME())
        return return_image()


class Filtragem(Resource):
    def post(self):
        save_image(request)
        img = Image.open(INPUT_IMAGE_NAME())
        img = normalize(img).astype('uint8')
        img = iaidft(img)
        img = normalize(np.abs(img)).astype('uint8')
        img = Image.fromarray(img)
        img.save(OUTPUT_IMAGE_NAME())
        return return_image()

class Notch(Resource):
    def post(self):
        save_image(request)
        img = Image.open(INPUT_IMAGE_NAME())
        #img = img.convert('RGB')
        #img = iargb2gray(img)
        fftimg = fft2(img)
        img = iadftview(fftimg)
        auxImg = Image.fromarray(img)
        width, height = auxImg.size


        notch_mask = img
        for j in range (1,width):
            for n in range(1,height):
                notch_mask[j,n] = 255;

        w = width
        h = height

        size_w = width/4
        size_h = 5         

        for n in range (int(-size_w/2),int(size_w/2)):
            for j in range(int(-size_h/2),int(size_h/2)):
                notch_mask[n+int(w/4),j+int(h/2)] = 0;
                
        #for n in range (int(-size_h/2),int(size_h/2)):
        #    for j in range(int(-size_w/2),int(size_w/2)):
        #        notch_mask[j-int((h/2)),n+3*int((w/4))] = 0;
                

        # size_w = 5
        # size_h = height*1.5/4   

        # for n in range (int(-size_w/2),int(size_w/2)):
            # for j in range(int(-size_h/2),int(size_h/2)):
                # notch_mask[j+int(h/4),n+int(w/2)] = 0;
                
        # for n in range (int(-size_w/2),int(size_w/2)):
            # for j in range(int(-size_h/2),int(size_h/2)):
                # notch_mask[j+3*int((h/4)),n-int(w/2)] = 0;

        filtered = fftimg*notch_mask

        img = iadftview(filtered)
        img = Image.fromarray(img)
        img.save(OUTPUT_IMAGE_NAME())
        return return_image()

        #img = normalize(filtered).astype('uint8')
        img = iaidft(filtered)
        img = normalize(np.abs(img)).astype('uint8')
        img = Image.fromarray(img)
        img.save(OUTPUT_IMAGE_NAME())
        return return_image()

class SantaHat(Resource):
    def post(self):
        save_image(request)
        # Read the image
        image = cv2.imread(INPUT_IMAGE_NAME())
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
            #flags = cv2.CV_HAAR_SCALE_IMAGE
        )
        background = Image.open(INPUT_IMAGE_NAME())
        foreground = Image.open("christmas_hat.png")

        for (x, y, w, h) in faces:
            img = add_hat(x,y,w,h,"SantaHat", background, foreground);
        img.save(OUTPUT_IMAGE_NAME())
        return return_image()

class FedoraHat(Resource):
    def post(self):
        save_image(request)
        # Read the image
        image = cv2.imread(INPUT_IMAGE_NAME())
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
            #flags = cv2.CV_HAAR_SCALE_IMAGE
        )
        background = Image.open(INPUT_IMAGE_NAME())
        foreground = Image.open('Fedora.png')


        for (x, y, w, h) in faces:
            img = add_hat(x,y,w,h,"FedoraHat", background, foreground);
        img.save(OUTPUT_IMAGE_NAME())
        return return_image()            

class Glasses(Resource):
    def post(self):
        save_image(request)

        # Read the image
        image = cv2.imread(INPUT_IMAGE_NAME())
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
            #flags = cv2.CV_HAAR_SCALE_IMAGE
        )
        background = Image.open(INPUT_IMAGE_NAME())
        foreground = Image.open('sunglass.png')

        for (x, y, w, h) in faces:
            img = add_hat(x,y,w,h,"Glasses", background, foreground);
        img.save(OUTPUT_IMAGE_NAME())
        return return_image()

class DrawRectangles(Resource):
    def post(self):
        save_image(request)
        # Read the image
        image = cv2.imread(INPUT_IMAGE_NAME())
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
            #flags = cv2.CV_HAAR_SCALE_IMAGE
        )
        background = Image.open(INPUT_IMAGE_NAME())
        foreground = Image.open('sunglass.png')

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(image, (x, y), (x+w, y+(h/2)), (0, 255, 0), 2)
        cv2.imwrite(OUTPUT_IMAGE_NAME(), image)
        return return_image()
        
api.add_resource(Departments_Meta, '/compress')
api.add_resource(AlturaRuido, '/altura')
api.add_resource(LarguraRuido, '/largura')
api.add_resource(AmbosRuido, '/ambos')
api.add_resource(Filtragem, '/filtragem')
api.add_resource(Notch, '/notch')
api.add_resource(SantaHat, '/santahat')
api.add_resource(FedoraHat, '/fedorahat')
api.add_resource(Glasses, '/glasses')
api.add_resource(DrawRectangles, '/drawrectangles')

if __name__ == '__main__':
      app.run(host='0.0.0.0', port=80)

