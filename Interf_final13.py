import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as waves
import pyaudio
import wave
import codecs
import random
import os
import scipy.signal as signal
import scipy.stats as stats
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore, QtGui, QtWidgets
import sounddevice as sd
import soundfile as sf

archivo = ""

##############################################################################################################################
############################### CLASE FILE #####################################################################
class App_File(QWidget):

    def __init__(self):
        super().__init__()
        global archivo
        self.title = 'PyQt5 file dialogs - pythonspot.com'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        archivo = self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        return self.openFileNameDialog()


    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            return fileName

    def openFileNamesDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self,"QFileDialog.getOpenFileNames()", "","All Files (*);;Python Files (*.py)", options=options)
        if files:
            print(files)

    def saveFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","All Files (*);;Text Files (*.txt)", options=options)
        if fileName:
            print(fileName)

##############################################################################################################################

class Ui_MainWindow(object):

    def triangular_2(self,A0, T, d, x):
        senal = signal.sawtooth((2*np.pi*T*x)+d,0.5)
        res=np.array([])
        for i in senal:
            if (i>0):
                res=np.concatenate([res,np.array([i])])
            else:
                res=np.concatenate([res,np.array([0])])
        return res*A0

    def triangular_3(self,A0, N, d, x):
        senal = signal.sawtooth((2*np.pi*(1/N)*x)+d,0.5)
        res=np.array([])
        for i in senal:
            if (i>0):
                res=np.concatenate([res,np.array([i])])
            else:
                res=np.concatenate([res,np.array([0])])
        return res*A0

    def abs_2(self,A0, T, d, x):
        sin=np.sin((2*3.1416*x*T)+d)
        res=np.array([])
        for i in sin:
            if (i>0):
                res=np.concatenate([res,np.array([i])])
            else:
                res=np.concatenate([res,np.array([i*-1])])
        return res*A0

    def rec_2(self,A0, T, d, x):
        sin=np.sin((2*3.1416*x*T)+d)
        res=np.array([])
        for i in sin:
            if (i>0):
                res=np.concatenate([res,np.array([i])])
            else:
                res=np.concatenate([res,np.array([0])])
        return res*A0

    def tren_2(self,A0, T, d, x):
        cos=np.cos((2*3.1416*x*T)+d)
        res=np.array([])
        for i in cos:
            if (i>0):
                res=np.concatenate([res,np.array([1])])
            else:
                res=np.concatenate([res,np.array([0])])
        return res*A0

    def sinc_2(self,A0, n, d, x):
        sincardinal = np.sinc(((n*x)/2)+d)
        res = (A0)*sincardinal
        return res

    def sin_2(self,A0, f0, d, x):
        sincardinal = np.sin((2*3.1416*f0*x)+d)
        res = A0*sincardinal
        return res

    def cos_2(self,A0, N, d, x):
        sincardinal = np.cos((2*3.1416*x*(1/N))+d)
        res = A0*sincardinal
        return res

    def cos_3(self,A0, f0, d, x):
        sincardinal = np.cos((2*3.1416*x*f0)+d)
        res = A0*sincardinal
        return res

    def exp_2(self,A0, f0, d, x):
        val = np.exp((1/f0)*x+d)
        res = val*A0
        return res

    def graficar_Sonido(self,nombre):
        archivo = nombre+'.wav'
        muestreo, sonido = waves.read(archivo)

        if (self.a0.toPlainText()):
            sonido = int(self.a0.toPlainText())* sonido

        self.informacion.setText('archivo de sonido grabado...')
        waves.write('Sonido_Graficado'+nombre+'.wav', muestreo, sonido)
        plt.plot(sonido)
        plt.xlabel('tiempo (s)')
        plt.ylabel('Amplitud')
        plt.show()
        plt.savefig("grafica_Sonido"+nombre+".png")


    def graficar_Parte(self,nombre,inicia,termina):
        archivo = nombre+'.wav'
        muestreo, sonido = waves.read(archivo)

        tamano = np.shape(sonido)
        muestras = tamano[0]
        m = len(tamano)
        canales = 1
        if (m > 1):
            canales = tamano[1]

        if (canales > 1):
            canal = 0
            uncanal = sonido[:, canal]
        else:
            uncanal = sonido

        if(self.muestras.isChecked):
            self.informacion.setText("Graficando Parte por segundos datos "+str(segundos))
            a = int(inicia*muestreo)
            b = int(termina*muestreo)
        else:
            self.informacion.setText("Graficando Parte por muestra datos "+str(muestra))
            a = inicia
            b = termina

        parte = uncanal[a:b]

        amplificar = input("Amplificar sonido?:[S/N]: ")
        if (self.a0.toPlainText()):
            parte = int(self.a0.toPlainText())*parte

        self.informacion.setText('archivo de parte[] grabado...')
        waves.write('Parte_Graficada_'+nombre+'.wav', muestreo, parte)

        plt.plot(parte)
        plt.xlabel('tiempo (s)')
        plt.ylabel('Amplitud')
        plt.show()
        plt.savefig("grafica_Parte_"+nombre+".png")


    def record(self,nombre,segundos):
        file_name = nombre+'.wav'

        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 8000
        RECORD_SECONDS = int(segundos)
        self.informacion.setText("Comienza la grabación con "+str(segundos))
        print("** Grabando **")
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK)

        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(file_name, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        os.system('cls')
        self.informacion.setText('Grabacion terminada , nombre de archivo wav: '+nombre)

    def abs_graficar(self,nombre,A0,f0,d,t1,t2,fm):
        for i in range(1, 2):
            N = f0/fm
            muestras=(t2-t1)*(f0/N)

            y = np.linspace(t1, t2, 8000)
            plt.plot(y,self.abs_2(A0,f0,d,y),'b')
            plt.title("Tren continuo")
            plt.ylabel("Amplitud")
            plt.xlabel("Tiempo")
            plt.axhline(0, color="black")
            plt.axvline(0, color="black")
            plt.savefig("grafica_tren_continuo"+str(i)+".png")
            plt.show()

            y = np.linspace(t1, t2,muestras)
            print('Graficando')
            plt.plot(y,self.abs_2(A0,f0,d,y),'c.')
            plt.title("Tren discreto")
            plt.ylabel("Amplitud")
            plt.xlabel("Tiempo")
            plt.axhline(0, color="black")
            plt.axvline(0, color="black")
            plt.savefig("grafica_tren_discreto"+str(i)+".png")
            plt.show()

            y = np.linspace(t1, t2,(t2-t1)*f0)
            arr = np.array(self.abs_2(A0,f0,d,y), dtype=np.float32)
            waves.write(nombre+".wav", int(f0), arr)

    def rec_graficar(self,nombre,A0,f0,d,t1,t2,fm):
        for i in range(1, 2):
            N = f0/fm
            muestras=(t2-t1)*(f0/N)

            y = np.linspace(t1, t2, 8000)
            plt.plot(y,self.rec_2(A0,f0,d,y),'b')
            plt.title("Sin recortado continuo")
            plt.ylabel("Amplitud")
            plt.xlabel("Tiempo")
            plt.axhline(0, color="black")
            plt.axvline(0, color="black")
            plt.savefig("grafica_tren_continuo"+str(i)+".png")
            plt.show()

            y = np.linspace(t1, t2,muestras)
            print('Graficando')
            plt.plot(y,self.rec_2(A0,f0,d,y),'c.')
            plt.title("Sin recortado discreto")
            plt.ylabel("Amplitud")
            plt.xlabel("Tiempo")
            plt.axhline(0, color="black")
            plt.axvline(0, color="black")
            plt.savefig("grafica_tren_discreto"+str(i)+".png")
            plt.show()

            y = np.linspace(t1, t2,(t2-t1)*f0)
            arr = np.array(self.rec_2(A0,f0,d,y), dtype=np.float32)
            waves.write(nombre+".wav", int(f0), arr)

    def tren_triangular_graficar(self,nombre,A0,f0,d,t1,t2,fm):
        for i in range(1, 2):
            N = f0/fm
            muestras=(t2-t1)*(f0/N)

            y = np.linspace(t1, t2, 8000)
            plt.plot(y,self.triangular_2(A0,f0, d, y),'b')
            plt.title("Tren continuo")
            plt.ylabel("Amplitud")
            plt.xlabel("Tiempo")
            plt.axhline(0, color="black")
            plt.axvline(0, color="black")
            plt.savefig("grafica_tren_triangular_continuo"+str(i)+".png")
            plt.show()

            y = np.linspace(t1,t2,muestras)
            print('Graficando')
            plt.plot(y,self.triangular_2(A0,f0,d,y),'c.')
            plt.title("Tren discreto")
            plt.ylabel("Amplitud")
            plt.xlabel("Tiempo")
            plt.axhline(0, color="black")
            plt.axvline(0, color="black")
            plt.savefig("grafica_tren_triangular_discreto"+str(i)+".png")
            plt.show()

            y = np.linspace(t1, t2,(t2-t1)*f0)
            arr = np.array(self.tren_2(A0,f0, d, y), dtype=np.float32)
            waves.write(nombre+".wav", int(f0), arr)

    def tren_graficar(self,nombre,A0,f0,d,t1,t2,fm):
        for i in range(1, 2):
            N = f0/fm
            muestras=(t2-t1)*(f0/N)

            y = np.linspace(t1, t2, 8000)
            plt.plot(y,self.tren_2(A0,f0, d, y),'b')
            plt.title("Tren continuo")
            plt.ylabel("Amplitud")
            plt.xlabel("Tiempo")
            plt.axhline(0, color="black")
            plt.axvline(0, color="black")
            plt.savefig("grafica_tren_continuo"+str(i)+".png")
            plt.show()

            y = np.linspace(t1,t2,muestras)
            print('Graficando')
            plt.plot(y,self.tren_2(A0,f0,d,y),'c.')
            plt.title("Tren discreto")
            plt.ylabel("Amplitud")
            plt.xlabel("Tiempo")
            plt.axhline(0, color="black")
            plt.axvline(0, color="black")
            plt.savefig("grafica_tren_discreto"+str(i)+".png")
            plt.show()

            y = np.linspace(t1, t2,(t2-t1)*f0)
            arr = np.array(self.tren_2(A0,f0, d, y), dtype=np.float32)
            waves.write(nombre+".wav", int(f0), arr)


    def sinc_graficar(self,nombre,A0,f0,d,t1,t2,fm):
        for i in range(1, 2):
            N=f0/fm
            muestras=(t2-t1)*(f0/N)

            y = np.linspace(t1, t2, (t2-t1)*f0)
            plt.plot(y,self.sinc_2(A0,(2*np.pi*f0), d, y),'g.')
            plt.title("Sinc continuo")
            plt.ylabel("Amplitud")
            plt.xlabel("Tiempo")
            plt.axhline(0, color="black")
            plt.axvline(0, color="black")
            plt.savefig("grafica_sinc_continuo"+str(i)+".png")
            plt.show()

            y = np.linspace(t1, t2,muestras)
            print('Graficando')
            plt.plot(y,self.sinc_2(A0,(2*np.pi*f0), d, y),'b.')
            plt.title("Sinc discreto")
            plt.ylabel("Amplitud")
            plt.xlabel("Tiempo")
            plt.axhline(0, color="black")
            plt.axvline(0, color="black")
            plt.savefig("grafica_sinc_discreto"+str(i)+".png")
            plt.show()

            y = np.linspace(t1, t2,(t2-t1)*f0)
            arr = np.array(self.sinc_2(A0,f0, d, y), dtype=np.float32)
            waves.write(nombre+".wav", int(f0), arr)

    def espiral_graficar(self,nombre,A0,f0,d,t1,t2,fm):
        for i in range(1, 2):
            N = fm/f0
            T0 = 1/f0
            muestras=(t2-t1)*N*(1/T0)

            x = np.linspace(t1, t2, 8000)
            c, s = self.cos_3(A0, f0, d, x), self.sin_2(A0,f0, d, x)
            x = np.array([x])
            c = np.array([c])
            s = np.array([s])
            fig = pl.figure()
            ax1 = fig.add_subplot(111,projection='3d')
            ax1.plot_wireframe(x, c, s, color = 'green')
            pl.show()

            x = np.linspace(t1, t2, muestras)
            c, s = self.cos_3(A0, f0, d, x), self.sin_2(A0,f0, d, x)
            x = np.array([x])
            c = np.array([c])
            s = np.array([s])
            fig = pl.figure()
            ax1 = fig.add_subplot(111,projection='3d')
            ax1.scatter(x, c, s, c = 'r')
            pl.show()

            #y = np.linspace(t1, t2, (t2-t1)*f0)
            #arr = np.array(self.sin_2(A0,f0, d, y), dtype=np.float32)
            #waves.write(nombre+".wav", int(f0), arr)

    def sin_graficar(self,nombre,A0,f0,d,t1,t2,fm):
        for i in range(1, 2):
            N = fm/f0
            T0 = 1/f0
            muestras=(t2-t1)*N*(1/T0)


            y = np.linspace(t1, t2, 8000)
            plt.plot(y, self.sin_2(A0,f0, d, y),'c')
            plt.title("Sin continuo")
            plt.ylabel("Amplitud")
            plt.xlabel("Tiempo")
            plt.axhline(0, color="black")
            plt.axvline(0, color="black")
            plt.savefig("grafica_sin_continuo"+str(i)+".png")
            plt.show()

            y = np.linspace(t1, t2, muestras)
            print('Graficando')
            plt.plot(y, self.sin_2(A0, f0, d, y),'m.')
            plt.title("Sin discreto")
            plt.ylabel("Amplitud")
            plt.xlabel("Tiempo")
            plt.axhline(0, color="black")
            plt.axvline(0, color="black")
            plt.savefig("grafica_sin_discreto"+str(i)+".png")
            plt.show()


            arr = np.array(self.sin_2(A0, f0, d, y), dtype=np.float32)
            waves.write(nombre+".wav", int(f0*N), arr)


    def cos_graficar(self,nombre,A0,f0,d,t1,t2,fm):
        for i in range(1, 2):
            N = f0/fm
            muestras=(t2-t1)*(f0/N)

            y = np.linspace(t1, t2, 8000)
            plt.plot(y, self.cos_3(A0, f0, d, y),'m')
            plt.title("Coseno continuo")
            plt.ylabel("Amplitud")
            plt.xlabel("Tiempo")
            plt.axhline(0, color="black")
            plt.axvline(0, color="black")
            plt.savefig("grafica_cos_continuo"+str(i)+".png")
            plt.show()

            y = np.linspace(t1, t2,muestras)
            print('Graficando')
            plt.plot(y, self.cos_3(A0,f0, d, y),'r.')
            plt.title("Coseno discreto")
            plt.ylabel("Amplitud")
            plt.xlabel("Tiempo")
            plt.axhline(0, color="black")
            plt.axvline(0, color="black")
            plt.savefig("grafica_cos_discreto"+str(i)+".png")
            plt.show()

            y = np.linspace(t1, t2, (t2-t1)*f0)
            arr = np.array(self.cos_3(A0, f0, d, y), dtype=np.float32)
            waves.write(nombre+".wav", int(f0), arr)


    def exp_graficar(self,nombre,A0,f0,d,t1,t2,fm):
        muestras=(t2-t1)/(1/f0)
        F0=f0/fm
        y = np.linspace(t1, t2, muestras)
        arr = np.vectorize(self.exp_2(A0, f0, d, y))
        print('Graficando')
        plt.plot(y, self.exp_2(A0, 2*3.1416*F0, d, y),'r.')
        plt.title("Exponencial discreto con "+str(muestras))
        plt.ylabel("Amplitud")
        plt.xlabel("Tiempo")
        plt.axhline(0, color="black")
        plt.axvline(0, color="black")
        plt.savefig("grafica_exp_discreto"+".png")
        plt.show()
        y = np.linspace(t1, t2, 8000)
        plt.plot(y, self.exp_2(A0, 2*3.1416*f0*(1/fm), d, y),'g.')
        plt.title("Exponencial continuo")
        plt.ylabel("Amplitud")
        plt.xlabel("Tiempo")
        plt.axhline(0, color="black")
        plt.axvline(0, color="black")
        plt.savefig("grafica_exp_continuo"+".png")
        plt.show()
        arr = np.array(self.exp_2(A0, 2*3.1416*f0*(1/fm), d, y), dtype=np.float32)
        waves.write(nombre+".wav", 8000, arr)

    def reproduc(self,nombre):
        chunk=1240
        print('Reproduciendo')
        App_File()
        data, fs = sf.read(archivo, dtype='float32')
        sd.play(data, fs)
        status = sd.wait()

        #f=wave.open(archivo,'rb')
        #p=pyaudio.PyAudio()
        #stream=p.open(format=p.get_format_from_width(f.getsampwidth()),channels=f.getnchannels(),rate=f.getframerate(),output=True)
        #data=f.readframes(chunk)
        #while data:
        #    stream.write(data)
        #    data=f.readframes(chunk)
        #stream.stop_stream()
        #stream.close()
        #p.terminate()


    ########################################################################################################BOT0NES
    def sound(self):
        self.informacion.setText('Reproducuioendo sonido')
        if(self.operacion.isChecked):
            mytext = self.plainTextEdit.toPlainText()
            if(mytext and mytext!='Introduce la informacion pertinente'):
                print('Personalizado con el texto '+mytext)
                try:
                    self.reproduc(mytext)
                except ValueError:
                    self.informacion.setText('Hubo un error en la introducción de datos :s')
            else:
                try:
                    self.informacion.setText('No se introdujo el nombre del archivo')
                except ValueError:
                    self.informacion.setText('Hubo un error en la introducción de datos :s')

    def espiralb(self):
        print('Operando exponencial compleja')
        self.informacion.setText('Bienvenido al exponencial compleja')
        if(self.operacion.isChecked):
            mytext = self.plainTextEdit.toPlainText()
            if(mytext and mytext!='Introduce la informacion pertinente'):
                print('Personalizado con el texto '+mytext)
                try:
                    self.espiral_graficar(mytext,float(self.a0.toPlainText()),float(self.f0.toPlainText()),float(self.desplazamiento.toPlainText()),float(self.t1.toPlainText()),float(self.t2.toPlainText()),float(self.fm.toPlainText()))
                except ValueError:
                    self.informacion.setText('Hubo un error en la introducción de datos :s')
            else:
                print('Default')
                try:
                    self.espiral_graficar('ESPIRAL_APP',float(self.a0.toPlainText()),float(self.f0.toPlainText()),float(self.desplazamiento.toPlainText()),float(self.t1.toPlainText()),float(self.t2.toPlainText()),float(self.fm.toPlainText()))
                except ValueError:
                    self.informacion.setText('Hubo un error en la introducción de datos :s')

    def sincb(self):
        print('Operando el sinc')
        self.informacion.setText('Bienvenido a la función Sinc')
        if(self.operacion.isChecked):
            mytext = self.plainTextEdit.toPlainText()
            if(mytext and mytext!='Introduce la informacion pertinente'):
                print('Personalizado con el texto '+mytext)
                try:
                    self.sinc_graficar(mytext,float(self.a0.toPlainText()),float(self.f0.toPlainText()),float(self.desplazamiento.toPlainText()),float(self.t1.toPlainText()),int(self.t2.toPlafloatext()),float(self.fm.toPlainText()))
                except ValueError:
                    self.informacion.setText('Hubo un error en la introducción de datos :s')
            else:
                print('Default')
                try:
                    self.sinc_graficar('SINC_APP',float(self.a0.toPlainText()),float(self.f0.toPlainText()),float(self.desplazamiento.toPlainText()),float(self.t1.toPlainText()),float(self.t2.toPlainText()),float(self.fm.toPlainText()))
                except ValueError:
                    self.informacion.setText('Hubo un error en la introducción de datos :s')
    def sinb(self):
        print('Operando el sin')
        self.informacion.setText('Bienvenido a la función Sin')
        if(self.operacion.isChecked):
            mytext = self.plainTextEdit.toPlainText()
            if(mytext and mytext!='Introduce la informacion pertinente'):
                print('Personalizado con el texto '+mytext)
                try:
                    self.sin_graficar(mytext,float(self.a0.toPlainText()),float(self.f0.toPlainText()),float(self.desplazamiento.toPlainText()),float(self.t1.toPlainText()),float(self.t2.toPlainText()),float(self.fm.toPlainText()))
                except ValueError:
                    self.informacion.setText('Hubo un error en la introducción de datos :s')
            else:
                print('Default')
                try:
                    self.sin_graficar('SIN_APP',float(self.a0.toPlainText()),float(self.f0.toPlainText()),float(self.desplazamiento.toPlainText()),float(self.t1.toPlainText()),float(self.t2.toPlainText()),float(self.fm.toPlainText()))
                except ValueError:
                    self.informacion.setText('Hubo un error en la introducción de datos :s')
    def cosb(self):
        print('Operando el cos')
        self.informacion.setText('Bienvenido a la función Cos')
        if(self.operacion.isChecked):
            mytext = self.plainTextEdit.toPlainText()
            if(mytext and mytext!='Introduce la informacion pertinente'):
                print('Personalizado con el texto '+mytext)
                try:
                    self.cos_graficar(mytext,float(self.a0.toPlainText()),float(self.f0.toPlainText()),float(self.desplazamiento.toPlainText()),float(self.t1.toPlainText()),float(self.t2.toPlainText()),float(self.fm.toPlainText()))
                except ValueError:
                    self.informacion.setText('Hubo un error en la introducción de datos :s')
            else:
                print('Default')
                try:
                    self.cos_graficar('COS_APP',float(self.a0.toPlainText()),float(self.f0.toPlainText()),float(self.desplazamiento.toPlainText()),float(self.t1.toPlainText()),float(self.t2.toPlainText()),float(self.fm.toPlainText()))
                except ValueError:
                    self.informacion.setText('Hubo un error en la introducción de datos :s')
    def grabando(self):
        print('Grabando')
        self.informacion.setText('Bienvenido a la grabacion')
        if(self.operacion.isChecked):
            mytext = self.plainTextEdit.toPlainText()
            if(mytext and mytext!='Introduce la informacion pertinente'):
                print('Personalizado con el texto '+mytext)
                self.nombre=mytext
                try:
                    self.record(mytext,float(self.segundosini.toPlainText()))
                except ValueError:
                    self.informacion.setText('Hubo un error en la introducción de datos :s')
            else:
                print('Default')
                self.nombre='GRABACION_APP'
                try:
                    self.record('GRABACION_APP',float(self.segundosini.toPlainText()))
                except ValueError:
                    self.informacion.setText('Hubo un error en la introducción de datos :s')

    def expb(self):
        print('Operando exponencial')
        self.informacion.setText('Bienvenido a la función Exponencial')
        if(self.operacion.isChecked):
            mytext = self.plainTextEdit.toPlainText()
            if(mytext and mytext!='Introduce la informacion pertinente'):
                print('Personalizado con el texto '+mytext)
                try:
                    self.exp_graficar(mytext,float(self.a0.toPlainText()),float(self.f0.toPlainText()),float(self.desplazamiento.toPlainText()),float(self.t1.toPlainText()),float(self.t2.toPlainText()),float(self.fm.toPlainText()))
                except ValueError:
                    self.informacion.setText('Hubo un error en la introducción de datos :s')
            else:
                print('Default')
                try:
                    self.exp_graficar('EXPONENCIAL_APP',float(self.a0.toPlainText()),float(self.f0.toPlainText()),float(self.desplazamiento.toPlainText()),float(self.t1.toPlainText()),float(self.t2.toPlainText()),float(self.fm.toPlainText()))
                except ValueError:
                    self.informacion.setText('Hubo un error en la introducción de datos :s')
    def tren_triangular(self):
        print('Tren de pulsos triangular')
        self.informacion.setText('Bienvenido a la función Tren de pulsos triangular')
        if(self.operacion.isChecked):
            mytext = self.plainTextEdit.toPlainText()
            if(mytext and mytext!='Introduce la informacion pertinente'):
                print('Personalizado con el texto '+mytext)
                try:
                    self.tren_triangular_graficar(mytext,float(self.a0.toPlainText()),float(self.f0.toPlainText()),float(self.desplazamiento.toPlainText()),float(self.t1.toPlainText()),float(self.t2.toPlainText()),float(self.fm.toPlainText()))
                except ValueError:
                    self.informacion.setText('Hubo un error en la introducción de datos :s')
            else:
                print('Default')
                try:
                    self.tren_triangular_graficar('EXPONENCIAL_APP',float(self.a0.toPlainText()),float(self.f0.toPlainText()),float(self.desplazamiento.toPlainText()),float(self.t1.toPlainText()),float(self.t2.toPlainText()),float(self.fm.toPlainText()))
                except ValueError:
                    self.informacion.setText('Hubo un error en la introducción de datos :s')
    def tren(self):
        print('Tren de pulsos')
        self.informacion.setText('Bienvenido a la función Tren de pulsos')
        if(self.operacion.isChecked):
            mytext = self.plainTextEdit.toPlainText()
            if(mytext and mytext!='Introduce la informacion pertinente'):
                print('Personalizado con el texto '+mytext)
                try:
                    self.tren_graficar(mytext,float(self.a0.toPlainText()),float(self.f0.toPlainText()),float(self.desplazamiento.toPlainText()),float(self.t1.toPlainText()),float(self.t2.toPlainText()),float(self.fm.toPlainText()))
                except ValueError:
                    self.informacion.setText('Hubo un error en la introducción de datos :s')
            else:
                print('Default')
                try:
                    self.tren_graficar('EXPONENCIAL_APP',float(self.a0.toPlainText()),float(self.f0.toPlainText()),float(self.desplazamiento.toPlainText()),float(self.t1.toPlainText()),float(self.t2.toPlainText()),float(self.fm.toPlainText()))
                except ValueError:
                    self.informacion.setText('Hubo un error en la introducción de datos :s')
    def abs(self):
        print('Abs')
        self.informacion.setText('Bienvenido a la función Sino absoluto')
        if(self.operacion.isChecked):
            mytext = self.plainTextEdit.toPlainText()
            if(mytext and mytext!='Introduce la informacion pertinente'):
                print('Personalizado con el texto '+mytext)
                try:
                    self.abs_graficar(mytext,float(self.a0.toPlainText()),float(self.f0.toPlainText()),float(self.desplazamiento.toPlainText()),float(self.t1.toPlainText()),float(self.t2.toPlainText()),float(self.fm.toPlainText()))
                except ValueError:
                    self.informacion.setText('Hubo un error en la introducción de datos :s')
            else:
                print('Default')
                try:
                    self.abs_graficar('SIN_ABSOLUTO_APP',float(self.a0.toPlainText()),float(self.f0.toPlainText()),float(self.desplazamiento.toPlainText()),float(self.t1.toPlainText()),float(self.t2.toPlainText()),float(self.fm.toPlainText()))
                except ValueError:
                    self.informacion.setText('Hubo un error en la introducción de datos :s')
    def rec(self):
        print('Abs')
        self.informacion.setText('Bienvenido a la función Sino recortado')
        if(self.operacion.isChecked):
            mytext = self.plainTextEdit.toPlainText()
            if(mytext and mytext!='Introduce la informacion pertinente'):
                print('Personalizado con el texto '+mytext)
                try:
                    self.rec_graficar(mytext,float(self.a0.toPlainText()),float(self.f0.toPlainText()),float(self.desplazamiento.toPlainText()),float(self.t1.toPlainText()),float(self.t2.toPlainText()),float(self.fm.toPlainText()))
                except ValueError:
                    self.informacion.setText('Hubo un error en la introducción de datos :s')
            else:
                print('Default')
                try:
                    self.rec_graficar('SIN_ABSOLUTO_APP',float(self.a0.toPlainText()),float(self.f0.toPlainText()),float(self.desplazamiento.toPlainText()),float(self.t1.toPlainText()),float(self.t2.toPlainText()),float(self.fm.toPlainText()))
                except ValueError:
                    self.informacion.setText('Hubo un error en la introducción de datos :s')


    def graficar_func(self):
        print('Graficar')
        self.informacion.setText('Bienvenido ')
        if(self.operacion.isChecked):
            mytext = self.plainTextEdit.toPlainText()
            if(mytext and mytext!='Introduce la informacion pertinente'):
                self.nombre= self.plainTextEdit.toPlainText()
                if(self.graficar.toPlainText()=='Parte' or self.graficar.toPlainText=='p' or self.graficar.toPlainText=='parte'):
                    print('Personalizado con el texto '+mytext)
                    try:
                        self.graficar_Parte(mytext,int(self.segundosini.toPlainText()),float(self.segundosfin.toPlainText()))
                    except ValueError:
                        self.informacion.setText('Hubo un error en la introducción de datos :s')
                else:
                    print('Default')
                    try:
                        self.graficar_Sonido(mytext)
                    except ValueError:
                        self.informacion.setText('Hubo un error en la introducción de datos :s')

            else:
                print('Default')
                if(self.graficar.toPlainText()=='Parte' or self.graficar.toPlainText=='p' or self.graficar.toPlainText=='parte'):
                    try:
                        self.graficar_Parte(self.nombre,int(self.segundosini.toPlainText()),float(self.segundosfin.toPlainText()))
                    except ValueError:
                        self.informacion.setText('Hubo un error en la introducción de datos :s')
                else:
                    print('Default')
                    try:
                        self.graficar_Sonido(self.nombre)
                    except ValueError:
                        self.informacion.setText('Hubo un error en la introducción de datos :s')

    ###############################################################################################################UI

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(803, 754)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/fondo/escudounam_rojo.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setAutoFillBackground(True)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setAutoFillBackground(False)
        self.centralwidget.setStyleSheet("")
        self.centralwidget.setObjectName("centralwidget")
        self.informacion = QtWidgets.QTextBrowser(self.centralwidget)
        self.informacion.setGeometry(QtCore.QRect(20, 70, 761, 161))
        font = QtGui.QFont()
        font.setFamily("Vivaldi")
        font.setPointSize(20)
        font.setItalic(True)
        self.informacion.setFont(font)
        self.informacion.setObjectName("informacion")
        self.Titulo = QtWidgets.QLabel(self.centralwidget)
        self.Titulo.setGeometry(QtCore.QRect(20, 0, 711, 81))
        font = QtGui.QFont()
        font.setFamily("Script")
        font.setPointSize(36)
        font.setBold(True)
        font.setWeight(75)
        self.Titulo.setFont(font)
        self.Titulo.setObjectName("Titulo")
        self.Titulo.setGeometry(150,0,600,80)
        self.sin = QtWidgets.QPushButton(self.centralwidget)
        self.sin.setGeometry(QtCore.QRect(70, 350, 201, 71))
        font = QtGui.QFont()
        font.setFamily("SWComp")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.sin.setFont(font)
        self.sin.setObjectName("sin")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit.setGeometry(QtCore.QRect(20, 250, 761, 31))
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.cos = QtWidgets.QPushButton(self.centralwidget)
        self.cos.setGeometry(QtCore.QRect(290, 350, 201, 71))
        font = QtGui.QFont()
        font.setFamily("SWComp")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.cos.setFont(font)
        self.cos.setObjectName("cos")
        self.sinc = QtWidgets.QPushButton(self.centralwidget)
        self.sinc.setGeometry(QtCore.QRect(510, 350, 201, 71))
        font = QtGui.QFont()
        font.setFamily("SWComp")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.sinc.setFont(font)
        self.sinc.setObjectName("sinc")
        ########################################################################################################3
        self.sinc.clicked.connect(self.sincb)
        #########################################################################################################################
        self.exponencial = QtWidgets.QPushButton(self.centralwidget)
        self.exponencial.setGeometry(QtCore.QRect(70, 450, 201, 71))
        font = QtGui.QFont()
        font.setFamily("SWComp")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.sincom = QtWidgets.QPushButton(self.centralwidget)
        self.sincom.setGeometry(QtCore.QRect(350, 630, 321, 71))
        self.sincom.setFont(font)
        self.sincom.setObjectName("sincom")
        self.sincom.setText(" Senoidal a valores complejos")
        self.sinrec = QtWidgets.QPushButton(self.centralwidget)
        self.sinrec.setGeometry(QtCore.QRect(119, 630, 201, 71))
        self.sinrec.setFont(font)
        self.sinrec.setObjectName("sinrec")
        self.sinrec.setText("Sin recortado")
        self.exponencial.setFont(font)
        self.exponencial.setObjectName("exponencial")
        self.sonido = QtWidgets.QPushButton(self.centralwidget)
        self.sonido.setGeometry(QtCore.QRect(515, 550, 201, 71))
        self.sonido.setFont(font)
        self.sonido.setObjectName("sonido")
        self.sonido.setText("Reproducir")
        self.sinabs = QtWidgets.QPushButton(self.centralwidget)
        self.sinabs.setGeometry(QtCore.QRect(290, 550, 201, 71))
        self.sinabs.setFont(font)
        self.sinabs.setObjectName("sinabs")
        self.sinabs.setText("| Sin |")

        self.triang = QtWidgets.QPushButton(self.centralwidget)
        self.triang.setGeometry(QtCore.QRect(70, 550, 201, 71))
        self.triang.setFont(font)
        self.triang.setObjectName("triangular")
        self.triang.setText("Pulso Triangular")
        self.grabar = QtWidgets.QPushButton(self.centralwidget)
        self.grabar.setGeometry(QtCore.QRect(290, 450, 201, 71))
        font = QtGui.QFont()
        font.setFamily("SWComp")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.grabar.setFont(font)
        self.grabar.setObjectName("grabar")
        self.reproducir = QtWidgets.QPushButton(self.centralwidget)
        self.reproducir.setGeometry(QtCore.QRect(510, 450, 201, 71))
        font = QtGui.QFont()
        font.setFamily("SWComp")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.reproducir.setFont(font)
        self.reproducir.setObjectName("reproducir")
        self.operacion = QtWidgets.QPushButton(self.centralwidget)
        self.operacion.setGeometry(QtCore.QRect(420, 290, 101, 41))
        font = QtGui.QFont()
        font.setFamily("SWComp")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.operacion.setFont(font)
        self.operacion.setObjectName("operacion")
        self.operacion.setText("Graficar")

        self.a0 = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.a0.setGeometry(QtCore.QRect(20, 300, 61, 31))
        self.a0.setPlainText("")
        self.a0.setObjectName("a0")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 280, 81, 16))
        self.label.setObjectName("label")
        self.label.setText("<font color='white'>A0/Amplificacion<front>")
        self.f0 = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.f0.setGeometry(QtCore.QRect(90, 300, 61, 31))
        self.f0.setPlainText("")
        self.f0.setObjectName("f0")
        self.desplazamiento = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.desplazamiento.setGeometry(QtCore.QRect(160, 300, 61, 31))
        self.desplazamiento.setPlainText("")
        self.desplazamiento.setObjectName("w0")
        self.t1 = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.t1.setGeometry(QtCore.QRect(230, 300, 61, 31))
        self.t1.setPlainText("")
        self.t1.setObjectName("muestras")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(120, 280, 21, 16))
        self.label_2.setObjectName("label_2")
        self.label_2.setText("<font color='white'>f0<front>")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(150, 280, 81, 16))
        self.label_3.setObjectName("label_3")
        self.label_3.setText("<font color='white'>Desplazamiento<front>")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(255, 280, 51, 16))
        self.label_4.setObjectName("label_4")
        self.label_4.setText("<font color='white'>T1<front>")
        self.graficar = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.graficar.setGeometry(QtCore.QRect(530, 300, 100, 31))
        self.graficar.setPlainText("")
        self.graficar.setObjectName("graficar")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(525, 280, 121, 16))
        self.label_5.setObjectName("label_5")
        self.label_5.setText("<font color='white'>Graficar Parte/Completo<front>")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(320, 280, 121, 16))
        self.label_6.setObjectName("label_6")
        self.label_6.setText("<font color='white'>T2<front>")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(750, 280, 121, 16))
        self.label_7.setObjectName("label_7")
        self.label_7.setText("<font color='white'>Fin<front>")
        self.segundosfin = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.segundosfin.setGeometry(QtCore.QRect(730, 300, 51, 31))
        self.segundosfin.setPlainText("")
        self.segundosfin.setObjectName("segundos")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(650, 280, 121, 16))
        self.label_8.setObjectName("label_8")
        self.label_8.setText("<font color='white'>Inicio o duracion<front>")
        self.segundosini = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.segundosini.setGeometry(QtCore.QRect(650, 300, 51, 31))
        self.segundosini.setPlainText("")
        self.segundosini.setObjectName("segundos")
        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox.setGeometry(QtCore.QRect(600, 330, 70, 17))
        self.checkBox.setObjectName("checkBox")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(620, 330, 121, 16))
        self.label_9.setObjectName("label_9")
        self.label_9.setText("<font color='white'>Muestras<front>")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(380, 280, 121, 16))
        self.label_10.setObjectName("label_9")
        self.label_10.setText("<font color='white'>fm<front>")
        self.fm = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.fm.setGeometry(QtCore.QRect(360, 300, 51, 31))
        self.fm.setPlainText("")
        self.fm.setObjectName("fm")


        self.t2 = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.t2.setGeometry(QtCore.QRect(300, 300, 51, 31))
        self.t2.setPlainText("")
        self.t2.setObjectName("segundos")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 799, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.sinc.clicked.connect(self.sincb)
        self.sin.clicked.connect(self.sinb)
        self.cos.clicked.connect(self.cosb)
        self.exponencial.clicked.connect(self.expb)
        self.grabar.clicked.connect(self.grabando)
        self.operacion.clicked.connect(self.graficar_func)
        self.reproducir.clicked.connect(self.tren)
        self.sinabs.clicked.connect(self.abs)
        self.sonido.clicked.connect(self.sound)
        ##################################################################################################################################
        self.sinrec.clicked.connect(self.rec)
        self.sincom.clicked.connect(self.espiralb)
        self.triang.clicked.connect(self.tren_triangular)
        ###################################################################################################################################
        self.informacion.setText('Introduce el nombre del archivo si se desea peronalizarlo , introduce los datos en el \n siguiente formato y posteriormente elige la opción . . . \n\t    Funciones: Archivo,A0,F0,Desplazamiento,T1 y T2\n\t Graficar,Reproducir y grabar: Archivo,segundos de inicio y fin\n\t   ,valida muestras e introduce parte si deseas segmentar el audio')


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Titulo.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#000000;\">Señales y Sistemas App</span></p></body></html>"))
        self.sin.setText(_translate("MainWindow", "Sin"))
        self.plainTextEdit.setPlainText(_translate("MainWindow", "Introduce la informacion pertinente"))
        self.cos.setText(_translate("MainWindow", "Cos"))
        self.sinc.setText(_translate("MainWindow", "Sinc"))
        self.exponencial.setText(_translate("MainWindow", "Exponencial"))
        self.grabar.setText(_translate("MainWindow", "Grabar"))
        self.reproducir.setText(_translate("MainWindow", "Tren de pulsos"))
        self.operacion.setText(_translate("MainWindow", "Graficar"))
import fondo_iu_rc


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.setStyleSheet("QMainWindow{background-image:url(:/fondo/Fondo.PNG)}")
    MainWindow.show()
    sys.exit(app.exec_())
