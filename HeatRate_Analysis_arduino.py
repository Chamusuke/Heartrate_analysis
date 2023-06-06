import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import serial
from ecgdetectors import Detectors
from scipy import signal
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import interp1d
from spectrum import aryule, arma2psd

"""
センサー実行
信号処理用プログラム
"""



Fs = 100  # sampling frequency [10ms]
M = 8  # score of moving average ( )
DAT_LEN = 180  # data range[s]

WINDOW = 'hamming'  # 窓関数

FILE_NAME = 'test.csv'  # データファイル

h = 0.01  # 微分区間10ms
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



def read_dat(filename):
    dat = np.loadtxt(filename, delimiter='\n')
    dat = dat[0:int(Fs * DAT_LEN)] * AMP_COEF
    return dat

def plot_wave(dat, is_wide=True, peak=None, title=''):
    t = np.arange(len(dat)) / Fs
    if is_wide:
        plt.figure(figsize=[11, 3])
    else:
        plt.figure(figsize=[7, 3])

    plt.plot(t, dat, zorder=1)

    if peak is not None:
        plt.scatter(t[peak], dat[peak], marker='o', color='r', zorder=2)
        plt.ylim(-YLIM, YLIM)
        plt.xlabel('Time [s]')
        plt.ylabel('ECG [mV]')
        plt.title(title)
        plt.show()


def plot_spectrum(dat: object, window: object = WINDOW) -> object:
    YLIM_MIN = -140
    LEN = len(dat)
    win = signal.get_window(window, LEN)

    rfft_dat = rfft(dat * win)
    rfft_freq = rfftfreq(LEN, d=1.0 / Fs)
    sp_rdat = np.abs(rfft_dat) ** 2 / (LEN * LEN)
    sp_rdat[1:-1] *= 2

    plt.figure(figsize=[7, 3])
    plt.plot(rfft_freq, 10 * np.log10(sp_rdat))
    plt.ylim(YLIM_MIN, 0)
    plt.yticks(np.arange(YLIM_MIN, 0, 20))
    plt.grid()
    plt.xlabel('Freqency [Hz]')
    plt.ylabel('Power [dBmVrms]')

    plt.show()

def detect_qrs(filename):
    """信号処理とQRSの検出　FFT解析　最大エントロピー法
    LF/HFの算出
    """
    """ノイズ処理"""
    # data read
    ecg = read_dat(FILE_NAME)
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
    #plot_wave(ecg_filt, peak=qrs, title='Two Moving Average')

    """スプライン補完"""
    Fsr = 20
    r_peaks_sec = np.array(qrs) / Fs
    rr_int = np.diff(r_peaks_sec)
    rs_timing = np.arange(np.ceil(r_peaks_sec[1]), np.floor(r_peaks_sec[-1]), 1 / Fsr)

    f = interp1d(r_peaks_sec[1:], rr_int, kind='cubic')
    rr_int_rs = f(rs_timing)
    N = len(rr_int_rs)
    N1 =len(rr_int)

    """周波数解析"""

    #最大エントロピー法
    win = signal.get_window('hamming', len(rr_int_rs))
    AR, P, k =aryule(rr_int_rs * win, N // 8)
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
    lf_hf_AR_8 = lf / hf

    lf_fft = np.sum(PSD_fft[lf_axis_fft])
    hf_fft = np.sum(PSD_fft[hf_axis_fft])
    lf_hf_fft = lf_fft / hf_fft

    return lf_hf_AR_8, lf_hf_fft

if __name__ == "__main__":


    #　センサから読み取り
    T = DAT_LEN*100 + 1000
    ser = serial.Serial('COM3', 9600)  # ポートの情報を記入
    # 空ファイルまたはクリアされたファイル用意
    with open('test.csv', 'w') as ff:
        print('0.0')
    ff.close()

    # i回分のデータをCSVファイルに保存　[0.01s=100Hz]
    for i in range(T):
        value = int(ser.readline().decode('utf-8').rstrip('\n'))
        # シリアル通信ではfloatだとエラーを起こしやすので整数で扱う（信号品質に問題ない）

        if i < 1000:
            pass

        else:
            print(value)
            with open('test.csv', 'a') as f:
                print('{}'.format(value), file=f)

    f.close()

    print("csv wrote")

    LF_HF_1, LF_HF_2 = detect_qrs(FILE_NAME)
    print('LF_HF_AR_8')
    print(LF_HF_1)
    print('LF_HF_FFT')
    print(LF_HF_2)
