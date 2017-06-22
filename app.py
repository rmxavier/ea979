from flask import Flask, request
from flask import json
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
reload(sys)  
sys.setdefaultencoding('utf8')

#Create a engine for connecting to SQLite3.
#Assuming salaries.db is in your app root folder

e = create_engine('sqlite:///salaries.db')

app = Flask(__name__)
api = Api(app)

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
        raise Exception, 'error: cannot normalize complex data'
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
        f = codecs.open('imageT.jpg', 'wb')
        f.write(request.data)
        f.close()
        img = Image.open('imageT.jpg')
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
	        
        return 'ok'

def iadftview(F):
    FM = iafftshift(np.log10(abs(F)+1))
    return normalize(FM).astype('uint8')

def iafftshift(f):
    f = np.asarray(f)
    return iaptrans(f, np.array(f.shape)/2)

class LarguraRuido(Resource):
    def post(self):
	f = codecs.open('imageT.jpg', 'wb')
        f.write(request.data)
        f.close()
        img = Image.open('imageT.jpg')
        img = img.convert('RGB')
        img = iargb2gray(img)
        img = ruidoLargura(img)
        img.save('ahu.bmp')
        return 'ok'

class AlturaRuido(Resource):
    def post(self):
        f = codecs.open('imageT.jpg', 'wb')
        f.write(request.data)
        f.close()
        img = Image.open('imageT.jpg')
        img = img.convert('RGB')
        img = iargb2gray(img)
        img = ruidoAltura(img)
        img.save('ahu.bmp')
        return 'ok'

class AmbosRuido(Resource):
    def post(self):
        f = codecs.open('imageT.jpg', 'wb')
        f.write(request.data)
        f.close()
        img = Image.open('imageT.jpg')
        img = img.convert('RGB')
        img = iargb2gray(img)
        img = ruidoAmbos(img)
        img.save('ambos.jpg')
        return 'ok'

class Mascara(Resource):
    def post(self):
        f = codecs.open('imageT.jpg', 'wb')
        f.write(request.data)
        f.close()
        img = Image.open('imageT.jpg')
	fftimg = fft2(img)
	img = iadftview(fftimg)
	auxImg = Image.fromarray(img)
	width, height = auxImg.size
	mask = iacircle(img.shape,35,[height/2,width/2])
	mask = iaptrans(mask, np.array(mask.shape)/2).astype(bool)
	filtered = fftimg*mask
	img = iadftview(filtered)
	img = Image.fromarray(img)
        img.save('mask.jpg')
        return 'ok'


class (Resource):
    def post(self):
        f = codecs.open('imageT.jpg', 'wb')
        f.write(request.data)
        f.close()
        img = Image.open('imageT.jpg')
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
        img.save('mf.jpg')
        return 'ok'


class Filtragem(Resource):
    def post(self):
        f = codecs.open('imageT.jpg', 'wb')
        f.write(request.data)
        f.close()
        img = Image.open('imageT.jpg')
	img = normalize(img).astype('uint8')
	img = iaidft(img)
	img = normalize(np.abs(img)).astype('uint8')
        img = Image.fromarray(img)
        img.save('ahu-filtered.jpg')
        return 'ok'

class Notch(Resource):
    def post(self):
		f = codecs.open('imageT.jpg', 'wb')
		f.write(request.data)
		f.close()
		img = Image.open('imageT.jpg')
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
		#	for j in range(int(-size_w/2),int(size_w/2)):
		#		notch_mask[j-int((h/2)),n+3*int((w/4))] = 0;
				

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
		img.save('mask.jpg')
		return 'ok'

		#img = normalize(filtered).astype('uint8')
		img = iaidft(filtered)
		img = normalize(np.abs(img)).astype('uint8')
		img = Image.fromarray(img)
		img.save('notch.jpg')
		return 'ok'
		
api.add_resource(Departments_Meta, '/compress')
api.add_resource(AlturaRuido, '/altura')
api.add_resource(LarguraRuido, '/largura')
api.add_resource(AmbosRuido, '/ambos')
api.add_resource(Mascara, '/mascara')
api.add_resource(Filtragem, '/filtragem')
api.add_resource(Mf, '/mf')
api.add_resource(Notch, '/notch')

if __name__ == '__main__':
      app.run(host='0.0.0.0', port=7501)

