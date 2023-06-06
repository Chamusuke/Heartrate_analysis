import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ecgdetectors import Detectors
#from gatspy.periodic import LombScargleFast
#from hrv import HRV
from scipy import signal
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import interp1d
from spectrum import arma2psd, aryule
from pylab import log10
#from biosppy import storage
#from biosppy.signals import bvp

"""
信号処理確認用プログラム
"""

Fs = 100  # sampling frequency [10ms]
M = 2  # score of moving average ( )
DAT_LEN = 180  # data range[s]

WINDOW = 'hamming'  # 窓関数

BVP_FILE_NAME = 'DPG_rawdata_wake.csv'  # 心電図データファイル

h = 0.1  #微分区間10ms
Fh = 1.0  # ハイパス・フィルタ遮断周波数
Fl = 30.0  # ローパス・フィルタ遮断周波数
Nf = 1  # ハイパス・フィルタ/ローパス・フィルタの次数

Fn = 50.0  # ノッチ・フィルタの中心周波数
Q = 4.0  # Q ノッチ・フィルタのQ値

# フィルタの設計
bh, ah = signal.butter(Nf, Fh, 'high', fs=Fs)
bl, al = signal.butter(Nf, Fl, 'low', fs=Fs)
bn, an = signal.iirnotch(Fn, Q, fs=Fs)


AMP_COEF = 2.5 / 2 ** 10 / 100 * 1000  # 2.5V/10vitADC/Gain100*[mV]

YLIM = 0.75  ##Y_range of frequency
LF_MIN = 0.04  # LF周波数範囲（下限）
LF_MAX = 0.15  # LF周波数範囲（上限）
HF_MIN = 0.15  # HF周波数範囲（下限）
HF_MAX = 0.40  # HF周波数範囲（上限）

def read_dat(filename):
    dat = np.loadtxt(filename, delimiter=',')
    dat = dat[0:int(Fs * DAT_LEN)] * AMP_COEF
    return dat

def plot_wave(dat, is_wide=True, peak=None, title=''):
#def plot_wave(dat, high, wide, num, peak=None, title=''):
    t = np.arange(len(dat)) / Fs
    if is_wide:
        plt.figure(figsize=[11, 3])
    else:
        plt.figure(figsize=[7, 3])

    #plt.subplot(high, wide, num)
    plt.plot(t, dat, zorder=1)
    #plt.xlim(28, 30)


    if peak is not None:
        plt.scatter(t[peak], dat[peak], marker='o', color='r', zorder=2)
        plt.ylim(-YLIM, YLIM)
        plt.xlabel('Time [s]')
        plt.ylabel('BVP [mV]')
        plt.title(title)
        plt.show()


def plot_spectrum(dat: object, window: object = WINDOW) -> object:
    YLIM_MIN = -180
    LEN = len(dat)
    win = signal.get_window(window, LEN)

    rfft_dat = rfft(dat * win)
    rfft_freq = rfftfreq(LEN, d=1.0 / Fs)
    sp_rdat = np.abs(rfft_dat) ** 2 / (LEN * LEN)
    sp_rdat[1:-1] *= 2

    #plt.figure(figsize=[7, 3])
    #plt.plot(rfft_freq, 10 * np.log10(sp_rdat))
    #plt.ylim(YLIM_MIN, 0)
    #plt.yticks(np.arange(YLIM_MIN, 0, 20))
    #plt.grid()
    #plt.xlabel('Freqency [Hz]')
    #plt.ylabel('Power [dBmVrms]')
    return rfft_freq, sp_rdat, YLIM_MIN




if __name__ == "__main__":

    """ノイズ処理"""
    # data read
    BVP = read_dat(BVP_FILE_NAME)
    BVP = BVP - np.mean(BVP)
    #plot_wave(BVP)
    #plt.title('RAW_BVP')
    #plt.ylim(-0.3, 0.3)
    #plot_spectrum(BVP)

    # 移動平均を適用した波形とスペクトルの描画
    BVP_mva = pd.Series(BVP).rolling(window=M).mean().dropna()
    BVP_mva = np.array(BVP_mva)
    #plot_wave(BVP_mva)
    #plt.ylim(-0.1, 0.1)
    #plt.title("Moving Average")
    #plot_spectrum(BVP_mva)

    # バンドパス・フィルタを適用した波形とスペクトルの描画
    BVP_filt = signal.lfilter(bh, ah, BVP_mva)
    BVP_filt = signal.lfilter(bl, al, BVP_filt)
    #plot_wave(BVP_filt)
    #plt.ylim(-0.08, 0.06)
    #plt.title("Band Pass Filter")
    #plot_spectrum(BVP_filt)

    """
    #逆フーリエ変換による平滑化
    N = len(BVP_filt)
    threshold = 0.0065  # 振幅の閾値
    x = np.fft.fft(BVP_filt)
    x_abs = np.abs(x)
    x_abs = x_abs / N * 2
    x[x_abs < threshold] = 0
    x = np.fft.ifft(x)
    x = x.real  # 複素数から実数部だけ取り出す
    plot_wave(x)
    """


    """QRSの検出 パッケージ内にある6つの手法を試して一番ピークを追随できるプログラムを使用する"""
    detectors = Detectors(Fs)
    #qrs_1 = detectors.engzee_detector(BVP_filt)
    #qrs_2 = detectors.swt_detector(BVP_filt)
    #qrs_3 = detectors.christov_detector(BVP_filt)
    #qrs_4 = detectors.hamilton_detector(BVP_filt)
    qrs_5 = detectors.two_average_detector(BVP_filt)
    #qrs_6 = detectors.pan_tompkins_detector(BVP_filt)



    #plot_wave(BVP_filt, peak=qrs_1, title='Engelse and Zeelenberg')
    #plot_wave(BVP_filt, peak=qrs_2, title='Stationary Wavelet Transform')
    #plot_wave(BVP_filt, peak=qrs_3, title='Christov')
    #plot_wave(BVP_filt, peak=qrs_4, title='Hamilton')
    #plot_wave(BVP_filt, peak=qrs_5, title='Two Moving Average')
    #plot_wave(x, peak=BVP_fft, title='Two Moving Average')
    #plot_wave(BVP_filt, peak=qrs_6, title='Pan and Tompkins')



    """スプライン補完"""
    Fsr = 10    #アップサンプリング
    r_peaks_sec = np.array(qrs_5) / Fs
    rr_int = np.diff(r_peaks_sec)
    rs_timing = np.arange(np.ceil(r_peaks_sec[1]), np.floor(r_peaks_sec[-1]), 1 / Fsr)

    f = interp1d(r_peaks_sec[1:], rr_int, kind='cubic')
    rr_int_rs = f(rs_timing)

    """
    plt.subplot(3, 1, 1)
    plt.scatter(r_peaks_sec[1:], rr_int, marker='o', color='tomato')
    plt.title('Cubic Spline')

    plt.subplot(3, 1, 2)
    plt.plot(rs_timing, rr_int_rs, color="darkorange")
    plt.scatter(r_peaks_sec[1:], rr_int, marker='o', color='tomato', linestyle='--')

    plt.subplot(3, 1, 3)
    plt.plot(rs_timing, 60 / rr_int_rs, color='red')
    plt.show()
    """

    """周波数解析"""
    N = len(rr_int_rs)
    N1 = len(rr_int)


    #最大エントロピー法
    win = signal.get_window('hamming', N)
    AR, P, k = aryule(rr_int_rs * win, N // 4)
    PSD = arma2psd(AR, NFFT=N)
    freq = np.linspace(0, Fsr, N, endpoint=False)

    # FFT(高速フーリエ変換)
    # 窓関数
    win_fft = signal.get_window(WINDOW, len(rr_int_rs))
    fft_res = rfft(rr_int_rs * win_fft)
    freq_fft = rfftfreq(N, d=1.0 / Fsr)
    PSD_fft = np.abs(fft_res) ** 2 / N / Fsr
    PSD_fft[1:-1] *= 2
    idx_1Hz = int(N // Fsr)

    """LF/HFの算出"""
    lf_axis = np.where((freq >= LF_MIN) & (freq <= LF_MAX))[0]
    hf_axis = np.where((freq >= HF_MIN) & (freq <= HF_MAX))[0]
    lf_axis_fft = np.where((freq_fft >= LF_MIN) & (freq_fft <= LF_MAX))[0]
    hf_axis_fft = np.where((freq_fft >= HF_MIN) & (freq_fft <= HF_MAX))[0]

    lf = np.sum(PSD[lf_axis])
    hf = np.sum(PSD[hf_axis])
    lf_hf = lf / hf

    lf_fft = np.sum(PSD_fft[lf_axis_fft])
    hf_fft = np.sum(PSD_fft[hf_axis_fft])
    lf_hf_fft = lf_fft / hf_fft

     #---グラフの描画--------------------------------------------------------------"
     #信号解析

    t1 = np.arange(len(BVP)) / Fs
    t = np.arange(len(BVP_mva)) / Fs

    fig = plt.figure(1, figsize=[7, 6])

    plt.subplot(4, 1, 1)
    plt.plot(t1, BVP, zorder=1)
    plt.title("RAW_BVP")
    plt.xlim(100, 110)

    plt.subplot(4, 1, 2)
    plt.plot(t, BVP_mva, zorder=1)
    plt.title("Moving Average")
    plt.xlim(100, 110)

    plt.subplot(4, 1, 3)
    plt.plot(t, BVP_filt, zorder=1)
    plt.title("Band Pass Filter")
    plt.xlim(100, 110)

    plt.subplot(4, 1, 4)
    plt.plot(t, BVP_filt, zorder=1)
    peak = qrs_5
    plt.scatter(t[peak], BVP_filt[peak], marker='o', color='r', zorder=2)
    plt.title("Peak on Signal")
    plt.xlim(100, 110)

    plt.subplots_adjust(hspace=0.6)
    fig.text(0.5, 0.05, 'Time[s]', ha='center', va='center', fontsize=10)
    fig.text(0.05, 0.5, 'Voltage[V]', ha='center', va='center', rotation='vertical', fontsize=10)
    plt.show()

    #FFT確認
    fig = plt.figure(2, figsize=[7, 6])
    rfft_freq, sp_rdat, YLIM_MIN = plot_spectrum(BVP)
    plt.subplot(3, 1, 1)
    plt.plot(rfft_freq, 10 * np.log10(sp_rdat))
    plt.ylim(YLIM_MIN, 0)
    plt.yticks(np.arange(YLIM_MIN, 0, 20))
    plt.grid()
    plt.title("RAW_BVP")

    rfft_freq, sp_rdat, YLIM_MIN = plot_spectrum(BVP_mva)
    plt.subplot(3, 1, 2)
    plt.plot(rfft_freq, 10 * np.log10(sp_rdat))
    plt.ylim(YLIM_MIN, 0)
    plt.yticks(np.arange(YLIM_MIN, 0, 20))
    plt.grid()
    plt.title("Moving Average")

    rfft_freq, sp_rdat, YLIM_MIN = plot_spectrum(BVP_filt)
    plt.subplot(3, 1, 3)
    plt.plot(rfft_freq, 10 * np.log10(sp_rdat))
    plt.ylim(YLIM_MIN, 0)
    plt.yticks(np.arange(YLIM_MIN, 0, 20))
    plt.grid()
    plt.title("Band Pass Filter")

    plt.subplots_adjust(hspace=0.6)
    fig.text(0.5, 0.05, 'Freqency [Hz]', ha='center', va='center', fontsize=10)
    fig.text(0.05, 0.5, 'Power [dBmVrms]', ha='center', va='center', rotation='vertical', fontsize=10)
    plt.show()


    #心拍変動解析
    fig2 = plt.figure(3, figsize=[7, 6])
    plt.subplot(3, 1, 1)
    plt.scatter(r_peaks_sec[1:], 60 / rr_int, marker='o', color='darkorange')
    plt.title('Scatter')

    plt.subplot(3, 1, 2)
    plt.plot(rs_timing, 60 / rr_int_rs, color="red")
    plt.scatter(r_peaks_sec[1:], 60 / rr_int, marker='o', color='darkorange', linestyle='--')
    plt.title('Cubic Spline and Scatter')

    plt.subplot(3, 1, 3)
    plt.plot(rs_timing, 60 / rr_int_rs, color='red')
    plt.title('Cubic Spline')

    plt.subplots_adjust(hspace=0.6)
    fig.text(0.5, 0.05, 'Time[s]', ha='center', va='center', fontsize=10)
    fig.text(0.05, 0.5, 'Heat_Rate', ha='center', va='center', rotation='vertical', fontsize=10)
    plt.show()

    #周波数解析
    e_idx = int(N // Fsr)
    lf_fill = np.append(lf_axis, np.max(lf_axis) + 1)
    hf_fill = np.append(hf_axis, np.max(hf_axis) + 1)
    lf_fill_fft = np.append(lf_axis_fft, np.max(lf_axis_fft) + 1)
    hf_fill_fft = np.append(hf_axis_fft, np.max(hf_axis_fft) + 1)

    plt.subplot(1, 2, 1)
    ax = plt.subplot(1, 2, 1)
    plt.plot(freq[2:e_idx], PSD[2:e_idx], lw=0.2, color='k')
    plt.fill_between(freq[lf_fill], PSD[lf_fill], facecolor='royalblue')
    plt.fill_between(freq[hf_fill], PSD[hf_fill], facecolor='orange')
    #plt.text(0.95, 0.85, 'LF/HF: {:.3f}', format(lf_hf), horizontalalignment='right',
    #transform=ax.transAxes, fontsize=22)

    plt.title('Yule AR 1/8')
    #plt.ylim(0, 100)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [dB]')

    plt.subplot(1, 2, 2)
    ax = plt.subplot(122)
    plt.plot(freq_fft[2:e_idx], PSD_fft[2:e_idx], lw=0.2, color='k')
    plt.fill_between(freq_fft[lf_fill_fft], PSD_fft[lf_fill_fft], facecolor='royalblue')
    plt.fill_between(freq_fft[hf_fill_fft], PSD_fft[hf_fill_fft], facecolor='orange')
    # plt.text(0.95, 0.85, 'LF/HF: {:.3f}', format(lf_hf), horizontalalignment='right',
    # transform=ax.transAxes, fontsize=22)

    plt.title('FFT')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [dB]')
    plt.subplots_adjust()

    plt.show()

    print('LF_HF_AR_8')
    print(lf_hf)
    print('LF_HF_FFT')
    print(lf_hf_fft)
    print(N)
    print(N1)

