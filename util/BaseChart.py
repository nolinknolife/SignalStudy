# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as signal
from scipy.stats import multivariate_normal


def showPlot(data, title=''):
    plt.close('all')
    data_len = len(data)
    plt.plot(range(data_len), data)
    plt.title(title)
    plt.show()
    plt.close('all')


def savePlot(data, file_path, title=''):
    plt.close('all')
    data_len = len(data)
    plt.plot(range(data_len), data)
    plt.title(title)
    plt.savefig(file_path)
    plt.close('all')


def saveSpectrum(sig, file_path,  fftpt=1024, slidpt=0, w=0, title=''):
    in_data = np.zeros(fftpt, 'complex')
    if len(sig) > fftpt:
        in_data = sig[:fftpt]
    else:
        in_data[:len(sig)] = sig
    fftdt = np.fft.fft(in_data)/np.sqrt(fftpt)
    fftdt = np.fft.fftshift(fftdt)
    fftdt2_lin = np.abs(fftdt)**2
    plt.plot(range(len(fftdt2_lin)), 10*np.log10(fftdt2_lin))
    plt.savefig(file_path)
    plt.close('all')


def showSpectrum(sig, fftpt=1024, y_min=0, y_max=0, slidpt=0, w=0, title=''):
    s = 0
    ts = 0
    sig_len = len(sig)
    fftdt2_lin = np.zeros(fftpt, 'float')
    if w == 0:
        w = np.ones(fftpt, 'float')
    if slidpt == 0:
        slidpt = fftpt
    while (s + fftpt) <= sig_len:
        e = s + fftpt
        fft = np.fft.fft(sig[s:e]*w)/np.sqrt(fftpt)
        fft = np.fft.fftshift(fft)
        fftdt2_lin += np.abs(fft)**2
        s += slidpt
        ts += 1
    fftdt2_lin /= ts
    plt.plot(range(len(fftdt2_lin)), 10*np.log10(fftdt2_lin))
    plt.title(title)
    if y_min != y_max:
        plt.ylim([y_min, y_max])
    plt.show()
    plt.close('all')


def showSpectrum_maxhold(sig, fftpt=1024, slidpt=0, w=0, title=''):
    s = 0
    ts = 0
    sig_len = len(sig)
    if w == 0:
        w = np.ones(fftpt, 'float')
    if slidpt == 0:
        slidpt = fftpt
    fftdt2_lin_buf = np.zeros((0, fftpt), 'float' )
    while (s + fftpt) <= sig_len:
        e = s + fftpt
        fft = np.fft.fft(sig[s:e]*w)/np.sqrt(fftpt)
        fft = np.fft.fftshift(fft)
        fftdt2_lin = np.abs(fft)**2
        fftdt2_lin_buf.append(fftdt2_lin)
        s += slidpt
    #maxholdをとる
    fftdt2_lin_maxhold = np.max(fftdt2_lin_buf, axis=0)
    plt.plot(range(len(fftdt2_lin_maxhold)), 10*np.log10(fftdt2_lin_maxhold))
    plt.title(title)
    plt.show()
    plt.close('all')
    

def showPlotMulti(*args, title=""):
    plt.close('all')
    if len(args) == 0:
        return
    d_list = args
    x_len = len(d_list[0])
    for d in d_list:
        d_len = len(d)
        if d_len < x_len:
            x_len = d_len
    for idx, d in enumerate(d_list):
        plt.plot(d[:x_len], label=str(idx))
    plt.legend()            
    plt.title(title)
    plt.show()
    plt.close('all')


def show_eye_pattern_ext(sig, t_sa):
    """
    アイパターンの表示
    １シンボルあたりのサンプル数が整数でない場合も考慮
    """
    pat_num = int((len(sig)-1)/t_sa)
    scope_len = int(round(2*t_sa))
    for idx in range(1, pat_num):
        s_idx = int(round(idx*t_sa))
        e_idx = s_idx + scope_len
        sig_scope = sig[s_idx:e_idx]
        plt.plot(sig_scope, color='black')
    plt.show()


def showStem(data, title=""):
    plt.close('all')
    data_len = len(data)
    plt.stem(range(data_len), data)
    plt.title(title)
    plt.show()
    plt.close('all')


def showScatter(data, title=""):
    plt.close('all')
    data_len = len(data)
    plt.scatter(range(data_len), data, s=50, facecolor='None', edgecolors='red')
    plt.title(title)
    plt.show()
    plt.close('all')


def showBar(height, label='', title=""):
    plt.close('all')
    labels = []
    if label != "":
        labels = [str(s) for s in label]
    linwdt = 10
    left = np.arange(len(height))
    if label == []:
        label = [str(d) for d in left]
    plt.bar(left, height, linewidth = linwdt, tick_label=label, \
        align="center")
    plt.title(title)
    plt.show()
    plt.close('all')



def show_eye_pattern_ext(sig, t_sa):
    """
    アイパターンの表示
    １シンボルあたりのサンプル数が整数でない場合も考慮
    """
    pat_num = int((len(sig)-1)/t_sa)
    scope_len = int(round(2*t_sa))
    for idx in range(1, pat_num):
        s_idx = int(round(idx*t_sa))
        e_idx = s_idx + scope_len
        sig_scope = sig[s_idx:e_idx]
        plt.plot(sig_scope, color='black')
    plt.show()


def show_constellation(data, title='constaration'):
    plt.close('all')
    x=[d.real for d in data]
    y=[d.imag for d in data]
    plt.scatter(x,  y, facecolor='None', edgecolors='red')
    plt.title(title)
    plt.show()
    plt.close('all')


def save_constellation(data, file_path, title='constaration'):
    plt.close('all')
    x=[d.real for d in data]
    y=[d.imag for d in data]
    plt.scatter(x,  y, facecolor='None', edgecolors='red')
    plt.title(title)
    plt.savefig(file_path)
    plt.close('all')


def save_phase_hist(data, file_path, b_width=15, title=0):
    #ヒストグラム
    bar_width = b_width
    phase_shift_deg = data * 180/np.pi
    half_w = 0.5 * bar_width
    bin_num = int(360/bar_width)+1
    hist_data_list = np.histogram(phase_shift_deg, bins = bin_num, \
        range=(-180-half_w, 180+half_w))
    hist_data = hist_data_list[0]
    linwdt = 10
    left = np.arange(len(hist_data))
    label = [str(d) for d in left]
    plt.bar(left, hist_data, linewidth = linwdt, tick_label=label, \
        align="center")
    plt.title(title)
    plt.savefig(file_path)
    plt.close('all')


def show_phase_hist(data, b_width=15, title=0):
    #ヒストグラム
    bar_width = b_width
    phase_shift_deg = data * 180/np.pi
    half_w = 0.5 * bar_width
    bin_num = int(360/bar_width)+1
    hist_data_list = np.histogram(phase_shift_deg, bins = bin_num, \
        range=(-180-half_w, 180+half_w))
    hist_data = hist_data_list[0]
    linwdt = 10
    left = np.arange(len(hist_data))
    label = [str(d) for d in left]
    plt.bar(left, hist_data, linewidth = linwdt, tick_label=label, \
        align="center")
    plt.title(title)
    plt.show()
    plt.close('all')


def show_teibai(sig, fftpt=8192, title=''):
    """
    ２逓倍、４逓倍、８逓倍のスペクトラムを表示
    """
    fp = fftpt
    sig2 = sig[:fp]*sig[:fp]
    title_a = title
    showSpectrum(sig2[:fp], fftpt=fp, title='2teibai' + ' ' + title_a)
    sig4 = sig2*sig2
    showSpectrum(sig4[:fp], fftpt=fp, title='4teibai' + ' ' + title_a)
    sig8 = sig4*sig4
    showSpectrum(sig8[:fp], fftpt=fp, title='8teibai' + ' ' + title_a)


def save_teibai(sig, output_path, fftpt=8192, title=''):
    """
    ２逓倍、４逓倍、８逓倍のスペクトラムを表示
    """
    fp = fftpt
    sig2 = sig[:fp]*sig[:fp]
    title_a = title
    file_path = output_path + '2teibai.jpg'
    saveSpectrum(sig2[:fp], file_path, fftpt=fp, title='2teibai' + ' ' + title_a)
    sig4 = sig2*sig2
    file_path = output_path + '4teibai.jpg'
    saveSpectrum(sig4[:fp], file_path, fftpt=fp, title='4teibai' + ' ' + title_a)
    sig8 = sig4*sig4
    file_path = output_path + '8teibai.jpg'
    saveSpectrum(sig8[:fp], file_path, fftpt=fp, title='8teibai' + ' ' + title_a)


def show3DPlot(data, title=''):
    # Figureと3DAxeS
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(111, projection="3d")
    # 軸ラベルを設定
    ax.set_xlabel("freq[Hz]", size = 16)
    ax.set_ylabel("time[ms]", size = 16)
    # (x,y)データを作成
    x = np.linspace(-8000, 8000, data.shape[1])
    y = np.linspace(0, 20, data.shape[0])
    # 格子点を作成
    X, Y = np.meshgrid(x, y)
    # 曲面を描画
    ax.plot_surface(X, Y, data, cmap = "jet")
    plt.show()


def showContourMap(data, title=''):
    fftpt = data.shape[1]
    time_bin = data.shape[0]
    # (x,y)データを作成
    x = np.linspace(-fftpt/2, fftpt/2, fftpt)
    y = np.linspace(-time_bin/2, time_bin/2, time_bin)
    X, Y = np.meshgrid(x, y)
    plt.pcolormesh(X, Y, data, cmap='jet') 
    plt.colorbar()
    plt.show()
    #plt.pcolor(X, Y, Z, cmap='hsv') 


