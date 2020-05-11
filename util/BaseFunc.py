# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


def generate_psk(sample_rate, phase_level, baudrate, len_s=1, base_freq_hz=0):
    """PSK変調信号を生成する"""
    #１シンボルあたりのサンプル
    symbol_sa = sample_rate/baudrate
    #シンボル数
    symbol_num = baudrate * len_s
    #データ列生成
    sym_list = np.random.randint(0, phase_level, symbol_num)
    sym_list_sa = np.repeat(sym_list, int(symbol_sa))
    #位相情報生成
    phase_unit = np.pi/phase_level
    base_phase = phase_unit * sym_list_sa + phase_unit
    base_sig = np.exp(1j * base_phase)
    sig_len_sa = len(base_sig)
    carr_sig = np.exp(1j*2*np.pi*base_freq_hz*np.arange(sig_len_sa)/sample_rate)
    sig = carr_sig * base_sig
    return sig


def calc_pw(sig):
    pw = np.dot(sig, np.conjugate(sig))/len(sig)
    return 10*np.log10(pw.real)


def create_power_spectrum(sig, fftpt, slidpt, fft_shift = True, w=0):
    """
    電力スペクトラム(lin)を生成する
    """
    s = 0
    ts = 0
    sig_len = len(sig)
    fftdt2_lin = np.zeros(fftpt, 'float')
    #信号の長さがfftptより短いときは０うめする
    if len(sig) < fftpt:
        sig_s = np.zeros(fftpt, 'complex')
        sig_s[:sig_len] = sig
        sig = sig_s
        sig_len = fftpt
    if w == 0:
        w = np.ones(fftpt, 'float')
    while (s + fftpt) <= sig_len:
        e = s + fftpt
        fft = np.fft.fft(sig[s:e]*w)/np.sqrt(fftpt)
        if fft_shift:
            fft = np.fft.fftshift(fft)
        fftdt2_lin += np.abs(fft)**2
        s += slidpt
        ts += 1
    fftdt2_lin /= ts
    return fftdt2_lin


def sync_sig(sig, freq_hz, sample_rate):
    """
    同調する
    """
    sig_len = len(sig)
    sync_sig = np.exp(2*np.pi*1j*freq_hz*np.arange(sig_len)/sample_rate)
    sig_shift = sig * sync_sig
    return sig_shift


def create_noise(sig_len, noise_level):
    noise = np.random.randn(sig_len) + 1j*np.random.randn(sig_len)
    pw_noise = np.dot(noise, np.conjugate(noise))
    pw_noise = pw_noise.real
    coeff_2 = (10**(noise_level/10) * sig_len)/pw_noise
    coeff = np.sqrt(coeff_2)
    noise *= coeff
    #検算
    pw = 10*np.log10(np.dot(noise, np.conjugate(noise))/len(noise))
    pw = pw.real
    return noise


def change_sig_level(sig, sig_level):
    sig_len = len(sig)
    pw_sig = np.dot(sig, np.conjugate(sig))
    pw_sig = pw_sig.real
    coeff_2 = (10**(sig_level/10) * sig_len)/pw_sig
    coeff = np.sqrt(coeff_2)
    sig_ = sig * coeff
    #検算
    pw = 10*np.log10(np.dot(sig_,  np.conjugate(sig_))/len(sig_))
    pw = pw.real
    return sig_


def add_noise(sig, sn=20):
    """ノイズを追加"""
    sig_len = len(sig)
    noise = np.random.randn(sig_len) + 1j*np.random.randn(sig_len)
    pw_sig = np.dot(sig, sig.conjugate())
    ns_sig = np.dot(noise, noise.conjugate())
    #ns_sig * coeff / pw_sig = 10**(-1*sn/10)
    coeff = 10**(-1*sn/10) * (pw_sig/ns_sig)
    noise *= np.sqrt(coeff)
    sig = sig + noise
    return sig


def differential_detection(sig, step_sa):
    """遅延検波"""
    sig_len = len(sig)
    phs = np.zeros(int(sig_len-step_sa), 'complex')
    step_sa = int(round(step_sa))
    for idx in range(sig_len - step_sa-1):
        step_idx = idx + step_sa
        iq = sig[idx]
        iq_s = sig[step_idx]
        #位相偏移量
        di_numrtr=iq.real*iq_s.real + iq.imag*iq_s.imag 
        di_dnmntr=np.sqrt(iq.real**2 + iq.imag**2)*np.sqrt(iq_s.real**2 + iq_s.imag**2)
        di = di_numrtr/di_dnmntr
        dq_numrtr=iq_s.real*iq.imag - iq.real*iq_s.imag 
        dq_dnmntr=np.sqrt(iq.real**2 + iq.imag**2)*np.sqrt(iq_s.real**2 + iq_s.imag**2)
        dq = dq_numrtr/dq_dnmntr
        phs[idx] = di + 1j*dq
    return phs


def differential_detection_no_norm(sig, step_sa):
    """遅延検波"""
    sig_len = len(sig)
    phs = np.zeros(int(sig_len-step_sa), 'complex')
    step_sa = int(round(step_sa))
    for idx in range(sig_len - step_sa-1):
        step_idx = idx + step_sa
        iq = sig[idx]
        iq_s = sig[step_idx]
        #位相偏移量
        di_numrtr=iq.real*iq_s.real + iq.imag*iq_s.imag 
        di_dnmntr=np.sqrt(iq.real**2 + iq.imag**2)*np.sqrt(iq_s.real**2 + iq_s.imag**2)
        di = di_numrtr/di_dnmntr
        dq_numrtr=iq_s.real*iq.imag - iq.real*iq_s.imag 
        dq_dnmntr=np.sqrt(iq.real**2 + iq.imag**2)*np.sqrt(iq_s.real**2 + iq_s.imag**2)
        dq = dq_numrtr/dq_dnmntr
        phs[idx] = di + 1j*dq
    return phs


def differential_detection_(sig, step_sa):
    """遅延検波"""
    sig_len = len(sig)
    phs = np.zeros(sig_len, 'complex')
    for idx in range(sig_len - step_sa):
        iq = sig[idx]
        iq_s = sig[idx + step_sa]
        #位相偏移量
        di_numrtr=iq.real*iq_s.real + iq.imag*iq_s.imag 
        di_dnmntr=np.sqrt(iq.real**2 + iq.imag**2)*np.sqrt(iq_s.real**2 + iq_s.imag**2)
        di = di_numrtr/di_dnmntr
        dq_numrtr=iq_s.real*iq.imag - iq.real*iq_s.imag 
        dq_dnmntr=np.sqrt(iq.real**2 + iq.imag**2)*np.sqrt(iq_s.real**2 + iq_s.imag**2)
        dq = dq_numrtr/dq_dnmntr
        phs[idx] = di + 1j*dq
    return phs

def fir_hamming(sig, tap_num, bw_hz, sample_rate):
    """
    ハミングフィルタを作用させる
    """
    #タップ数判定
    m = int((tap_num - 1)/2)
    cutoff = np.pi * (bw_hz/(0.5*sample_rate))
    hd = np.zeros(tap_num, 'float')
    w = np.zeros(tap_num, 'float')
    for i, n in enumerate(np.arange(-m,(m+1))):
        if n == 0:
            hd[i] = cutoff/np.pi
            w[i] = 0.54 + 0.46*np.cos(n*np.pi/m)
        else:
            hd[i] = np.sin(n * cutoff)/(n*np.pi)
            w[i] = 0.54 + 0.46*np.cos(n*np.pi/m)
    h = w * hd
    #sig_lpf = np.convolve(h, sig)
    a=1
    b = h
    filtered = signal.lfilter(b, a, sig)
    delay_sa = int(tap_num-1/2)
    return filtered[delay_sa:]
    #return sig_lpf[delay_sa:-tap_num]


def fir_win(sig, tap_num, bw_hz, sample_rate, win_type="hann"):
    """
    FIRフィルタを作用させる
    """
    cutoff = bw_hz/(sample_rate/2)
    a=1
    b = signal.firwin(tap_num, cutoff, window=win_type)
    filtered = signal.lfilter(b, a, sig)
    delay_sa = int(tap_num-1/2)
    return filtered[delay_sa:]


def cosrof(alpha, t_sa, tap_num):
    """
    alpha:ロールオフ率
    t_sa:ナイキスト間隔[Sa]
    tap_num:タップ数
    特記事項:タップ数は奇数であることを想定
    """
    rt = np.zeros(tap_num, 'float')
    eta = 0.001
    s=int((tap_num-1)/2)
    for tap_idx, idx in enumerate(range(-s,s+1)):
        d = t_sa/(2*alpha)
        if idx == 0:
            rt[tap_idx] = 1
        elif np.abs(idx - d) < eta:
            phase = (np.pi * d)/t_sa 
            rt[tap_idx] = np.sin(phase)/(phase*(np.pi/4))
        elif np.abs(idx + d) < eta:
            phase = -1 * (np.pi * d)/t_sa 
            rt[tap_idx] = np.sin(phase)/(phase*(np.pi/4))
        else:
            phase = (np.pi *idx)/t_sa
            sintrm = np.sin(phase)/phase
            costrm = np.cos(alpha *phase)/(1-(2*alpha*idx/t_sa)**2)
            rt[tap_idx] = sintrm * costrm
    return rt


def generate_psk_with_rf(sample_rate, phase_level, baudrate, len_s=1, base_freq_hz=0):
    t_sa = sample_rate/baudrate  
    #ロールオフフィルタパラメータ
    alpha = 0.7
    tap_num = 151
    #信号生成
    nrz = np.random.randint(0,phase_level,baudrate*len_s)*2-(phase_level-1)
    #print('base data ', nrz[:20])
    #NRZをゼロ補間し位相に変換
    sig_len = int(t_sa * len(nrz))
    base_sig = np.zeros(sig_len, 'complex')
    ps = np.pi/phase_level
    for idx, d in enumerate(nrz):
        i = int(round(idx * t_sa))
        base_sig[i] = np.exp(1j*d*ps)
    #self.showStem(base_sig[:200].real, 'zero insert @ real')
    #self.showPlotMulti(base_sig[:200].real, base_sig[:200].imag, title='np.exp(1j * nrz_s * ps) before rf ')
    #ロールオフフィルタ作用
    rf = cosrof(alpha, int(t_sa), tap_num)
    y = np.convolve(rf, base_sig, 'same')
    #周波数シフト
    sig_len_sa = len(y)
    carr_sig = np.exp(1j*2*np.pi*base_freq_hz*np.arange(sig_len_sa)/sample_rate)
    y *= carr_sig
    #ノイズ付加
    #y = add_noise(y, 20)
    return y


def pickup_maxmal_only(data, order_ = 5):
    #極大ピーク
    peak_idx_list_ = signal.argrelmax(data, order=order_)
    peak_idx_list = peak_idx_list_[0]
    #極大ピークに関してのみ補正していく
    peak_list_diff = np.zeros(len(peak_idx_list), 'float')
    #
    pk_diff_idx = 0
    for idx in range(len(peak_idx_list[:-1])):
        #極大点のとき
        if idx == 0:
            p_1 = peak_idx_list[idx+1]
        else:
            p_1 = peak_idx_list[idx-1]
        p_2 = peak_idx_list[idx]
        p_3 = peak_idx_list[idx+1]
        a = data[p_1]
        b = data[p_2]
        c = data[p_3]
        v = ((b-a) + (b-c))/2
        peak_list_diff[pk_diff_idx]=v
        pk_diff_idx += 1
    return peak_idx_list[:len(peak_list_diff)], peak_list_diff


def find_symbol_start(sig, symbol_sa):
    pow_sa = 10
    for period in range(1, 11):
        pow_n_sa = period * symbol_sa
        if(np.abs(np.round(pow_n_sa)-pow_n_sa) < 0.1):
            pow_sa = period
            break
    scope_sa = int(round(pow_sa * symbol_sa))
    print('pow_sa %.2f scope_sa %.2f' % (pow_sa, pow_sa * symbol_sa))
    fold_sig = np.zeros(scope_sa, 'float')
    for n in range(10):
        s = n * scope_sa
        e = s + scope_sa
        ss = np.abs(sig[s:e])**2
        fold_sig += ss
    #self.showPlot(fold_sig)
    min_idx=np.argmin(fold_sig)
    min_idx_in = min_idx
    while symbol_sa < min_idx_in:
        min_idx_in -= symbol_sa
    print('min_idx %d, start_sa %f'% (min_idx, min_idx_in))
    start_idx = int(round(min_idx_in + 0.5*symbol_sa))
    print('symbol_sa %.1f, start index %d' % (symbol_sa, start_idx))
    return start_idx


def clock_recovery(sig, sample_rate, baudrate):
    data_len = len(sig)
    symbol_sa = sample_rate/baudrate
    symbol_num = int(data_len/symbol_sa)
    clk_rvy_data = np.zeros(symbol_num, 'complex')
    for idx in range(symbol_num):
        cross_idx = int(round(idx*symbol_sa))
        clk_rvy_data[idx] = sig[cross_idx]
    return clk_rvy_data


def upsample(sig, sample_rate, up_rate):
    """
    アップサンプリング
    """
    #ゼロ挿入
    sig_len = len(sig)
    up_sig = np.zeros(int(sig_len * up_rate), 'complex')
    for idx, val in enumerate(sig):
        up_sig[int(idx*up_rate)] = val
    #LPF
    tap_num = 511
    up_sample_rate = int(sample_rate * up_rate)
    lpf_sig = fir_win(up_sig, tap_num, 0.5*sample_rate, up_sample_rate)
    delay = int((tap_num - 1)/2)
    lpf_sig = lpf_sig[delay:]
    return lpf_sig


def pickup_delta_diff_pt(data, thld):
    x = []
    y = []
    d1 = data[0]
    for idx, d in enumerate(data[1:]):
        d2 = d
        if np.abs(d1-d2) < thld:
            x.append(idx)
            y.append(d1)
        d1=d2
    plt.scatter(x,y)
    plt.title('only micro diff point')
    plt.show()


def decode_qpsk(data):
    code = []

    for d in data:
        if d < 0:
            d += 2*np.pi
        c = 0
        if 0 < d <=np.pi/2:
            c = 0
        elif np.pi/2 < d <= np.pi:
            c = 1
        elif np.pi < d <=  3*np.pi/2:
            c = 2
        else:
            c = 3
        code.append(c)
    return code
            

def revise_center_freq(sig, teibai, sample_rate, fftpt, range_hz):
    """
    逓倍のFFTから中心周波数のずれを測定
    """
    hz_per_bin = sample_rate / fftpt
    half_rng_bin = int(0.5*teibai*range_hz/hz_per_bin)
    cnt_bin = int(fftpt/2)
    sig_teibai = sig
    teibai_base = int(np.log2(teibai))
    for i in range(teibai_base):
        sig_teibai = sig_teibai * sig_teibai
    pw_lin = create_power_spectrum(sig_teibai, fftpt, fftpt)
    pw_lin = pw_lin[cnt_bin-half_rng_bin:cnt_bin+half_rng_bin]
    peak_idx_list, peak_idx_val = pickup_maxmal_only(10*np.log10(pw_lin), 1)
    plt.stem(peak_idx_list, peak_idx_val)
    peak_idx_val_std = np.sqrt(np.var(peak_idx_val))
    peak_val_avg = np.average(peak_idx_val)
    plt.title('peak only at envelop std %.2f avd %.2f' % (peak_idx_val_std, peak_val_avg))
    plt.show()
    #ピークがあるか？
    max_peak_val = np.max(peak_idx_val)
    if max_peak_val < 2*peak_idx_val_std:
        print('ピークなし')
        #return [False, 0]
    #ピークがある場合
    max_peak_idx = np.argmax(peak_idx_val)
    max_idx = peak_idx_list[max_peak_idx]
    diff_from_cnt_bin = max_idx - half_rng_bin
    diff_hz = diff_from_cnt_bin * hz_per_bin / teibai
    print('中心周波数修正 %.2f' % diff_hz)
    return [True, diff_hz]



def do_equlizer(sig, tsa, tap_num = 21):
    """
    等化フィルタ
    """
    mu = 0.001
    c_num = tap_num
    c = np.zeros(c_num, 'complex')
    c[0]=1 + 1j
    symb = sig[::tsa]
    z_m = np.zeros(len(symb), 'complex')
    m = 0
    for idx in range(c_num, len(symb)):
        s = symb[idx-c_num:idx]
        y = s[::-1]
        z = np.dot(y,c)
        z_m[m] = z
        m += 1
        #フィルタ更新
        coeff = mu * z *(np.abs(z)**2 -1)
        delta = coeff * y.conjugate()
        c -= delta
    return z_m