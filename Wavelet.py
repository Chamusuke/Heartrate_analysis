import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy.interpolate import interp1d
from ecgdetectors import Detectors
from scipy import signal

width = 6
FILE_NAME = 'DPG_rawdata_wake.csv'  # wake データファイル
FILE_NAME_1 = 'DPG_rawdata_sleep.csv'  # sleep データファイル
Fs = 100  # sampling frequency [10ms]
M = 8  # score of moving average ( )
DAT_LEN = 180  # data range[s]

WINDOW = 'hamming'  # 窓関数

h = 0.1  # 微分区間 10ms
Fh = 1.0  # ハイパス・フィルタ遮断周波数
Fl = 30.0  # ローパス・フィルタ遮断周波数
Nf = 1  # ハイパス・フィルタ/ローパス・フィルタの次数

Fn = 50.0  # ノッチ・フィルタの中心周波数
Q = 4.0  # Q ノッチ・フィルタのQ値

# フィルタの設計
bh, ah = signal.butter(Nf, Fh, 'high', fs=Fs)
bl, al = signal.butter(Nf, Fl, 'low', fs=Fs)
bn, an = signal.iirnotch(Fn, Q, fs=Fs)

AMP_COEF = 2.5 / 2 ** 10 / 100 * 1000  # 2.5V/10vitADC/Gain100[mV]

YLIM = 0.75  ##Y_range of frequency
LF_MIN = 0.04  # LF周波数範囲（下限）
LF_MAX = 0.15  # LF周波数範囲（上限）
HF_MIN = 0.15  # HF周波数範囲（下限）
HF_MAX = 0.45  # HF周波数範囲（上限）

Fsr = 10  # スプライン補間のアップサンプリング


def read_dat(filename):
    dat = np.loadtxt(filename, delimiter=',')
    dat = dat[0:int(Fs * DAT_LEN)] * AMP_COEF
    return dat


def detect_qrs(filename):
    """信号処理とQRSの検出 FFT解析 最大エントロピー法
    LF/HFの算出
    """
    """ノイズ処理"""
    # data read
    ecg = read_dat(filename)
    ecg = ecg - np.mean(ecg)

    # 移動平均を適用した波形とスペクトルの描画
    ecg_mva = pd.Series(ecg).rolling(window=M).mean().dropna()
    ecg_mva = np.array(ecg_mva)

    # バンドパス・フィルタを適用した波形とスペクトルの描画
    ecg_filt = signal.lfilter(bh, ah, ecg_mva)
    ecg_filt = signal.lfilter(bl, al, ecg_filt)

    """QRSの検出"""
    detectors = Detectors(Fs)

    qrs = detectors.two_average_detector(ecg_filt)
    # plot_wave(ecg_filt, peak=qrs, title='Two Moving Average')

    """スプライン補完"""

    r_peaks_sec = np.array(qrs) / Fs
    rr_int = np.diff(r_peaks_sec)
    rr_timing = np.arange(np.ceil(r_peaks_sec[1]), np.floor(r_peaks_sec[-1]), 1 / Fsr)

    f = interp1d(r_peaks_sec[1:], rr_int, kind='cubic')
    rr_int_rs = f(rr_timing)

    return rr_timing, rr_int_rs


# マザーウェーブレット：モルレーウェーブレット
def morlet(x, f, width):
    sf = f / width
    st = 1 / (2 * math.pi * sf)
    A = 1 / (st * math.sqrt(2 * math.pi))
    h = -np.power(x, 2) / (2 * st ** 2)
    co1 = 1j * 2 * math.pi * f * x
    return A * np.exp(co1) * np.exp(h)


# 連続ウェーブレット変換
def mycwt(Fs, data, fmax, wavelet_R=2):
    # Fs:           サンプリング周波数
    # data:         信号
    # wavelet_R:    マザーウェーブレットの長さ(秒)
    # fmax:         解析する最大周波数

    Fs = 100
    Ts = 1 / Fs  # サンプリング時間幅
    data_length = len(data)  # 信号のサンプル数を取得

    # マザーウェーブレットの範囲
    wavelet_length = np.arange(-wavelet_R, wavelet_R, Ts)

    # 連続ウェーブレット変換後のデータを格納する配列の作成
    wn = np.zeros([fmax, data_length])

    # 連続ウェーブレット変換の実行
    for i in range(0, fmax):
        wn[i, :] = np.abs(np.convolve(data, morlet(wavelet_length, i + 1, width), mode='same'))
        wn[i, :] = (2 * wn[i, :] / Fs) ** 2

    return wn


# 連続ウェーブレット変換後のカラーマップ作成関数
def cwt_plot(CWT, sample_time, fmax, fig_title):
    plt.imshow(CWT, cmap='jet', aspect='auto', vmax=0.006, vmin=0)
    plt.title(fig_title)
    plt.xlabel("time[s/10]")
    plt.ylabel("frequency[Hz]")
    plt.axis([0, len(sample_time), 0, fmax - 1])
    plt.ylim(0, 0.5)
    plt.colorbar(aspect=40, pad=0.25, shrink=0.6, orientation='horizontal', extend='both')


if __name__ == "__main__":
    rr_timing, rr_int_rs = detect_qrs(FILE_NAME)

    rr_timing_1, rr_int_rs_1 = detect_qrs(FILE_NAME_1)

    # 連続ウェーブレット変換 ----------------------------------------
    fmax = 2  # 解析する最大周波数
    cwt_signal = mycwt(Fs=Fs, data=rr_int_rs, fmax=fmax)
    cwt_signal_1 = mycwt(Fs=Fs, data=rr_int_rs_1, fmax=fmax)

    # 以下、図用 ----------------------------------------
    '''
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 2, 1)
    plt.title("Cubic Spline HRT Awake")
    plt.plot(rr_timing, 60 / rr_int_rs)
    plt.xlim(0, len(rr_timing) / 10)
    plt.xlabel("time[s]")
    plt.ylim(25, 125)
    '''
    plt.subplot(2, 1, 1)  # 連続ウェーブレット変換した時のカラーマップの図
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    fig_title01 = "CWT_Awake"
    cwt_plot(cwt_signal, rr_timing, fmax, fig_title01)
    '''
    plt.subplot(2, 2, 2)
    plt.title("Cubic Spline HRT Sleep")
    plt.plot(rr_timing_1, 60 / rr_int_rs_1)
    plt.xlim(0, len(rr_timing_1) / 10)
    plt.xlabel("time[s]")
    plt.ylim(25, 125)
    '''
    plt.subplot(2, 1, 2)  # 連続ウェーブレット変換した時のカラーマップの図
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    fig_title01 = "CWT_Sleep"
    cwt_plot(cwt_signal_1, rr_timing_1, fmax, fig_title01)
    plt.show()
    print(cwt_signal)
    print(cwt_signal_1)
