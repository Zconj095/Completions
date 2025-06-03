import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa import stattools as ts
from scipy import signal, optimize, stats
from scipy.fftpack import rfft, fftshift
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.linear_model import LogisticRegression, ElasticNetCV
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.cluster import AffinityPropagation, MeanShift
from sklearn import datasets
import warnings
warnings.filterwarnings('ignore')

class SunspotAnalyzer:
    """Comprehensive sunspot data analysis toolkit"""
    
    def __init__(self):
        self.data = None
        self.df = None
        self.years = None
        self.sunspots = None
        self.load_data()
    
    def load_data(self):
        """Load sunspot data"""
        data_loader = sm.datasets.sunspots.load_pandas()
        self.data = data_loader.data
        self.years = self.data["YEAR"].values
        self.sunspots = self.data["SUNACTIVITY"].values
        self.df = pd.DataFrame({'SUNACTIVITY': self.sunspots}, 
                              index=pd.to_datetime(self.years, format='%Y'))
    
    def plot_moving_averages(self, windows=[11, 22]):
        """Plot original data with moving averages"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.years, self.sunspots, label="Original", alpha=0.7)
        
        for window in windows:
            ma = self.df.rolling(window=window).mean()["SUNACTIVITY"].values
            plt.plot(self.years, ma, label=f"SMA {window}", linewidth=2)
        
        plt.title("Sunspot Activity with Moving Averages")
        plt.xlabel("Year")
        plt.ylabel("Sunspot Activity")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_window_functions(self, window=22, n_samples=150):
        """Plot different window function smoothing"""
        df_subset = self.df.tail(n_samples)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        df_subset.plot(ax=ax, label="Original", alpha=0.7)
        
        window_types = ['boxcar', 'triang', 'blackman', 'hann', 'bartlett']
        for wintype in window_types:
            smoothed = df_subset.rolling(window=window, win_type=wintype, 
                                       center=False).mean()
            smoothed.plot(ax=ax, label=wintype, linewidth=2)
        
        plt.title(f"Window Function Smoothing (window={window})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def autocorrelation_analysis(self):
        """Perform autocorrelation analysis"""
        y = self.sunspots - np.mean(self.sunspots)
        norm = np.sum(y ** 2)
        correlated = np.correlate(y, y, mode='full') / norm
        res = correlated[len(correlated)//2:]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Manual autocorrelation
        ax1.plot(res)
        ax1.set_title("Manual Autocorrelation")
        ax1.set_xlabel("Lag")
        ax1.set_ylabel("Autocorrelation")
        ax1.grid(True, alpha=0.3)
        
        # Pandas autocorrelation plot
        pd.plotting.autocorrelation_plot(self.sunspots, ax=ax2)
        ax2.set_title("Pandas Autocorrelation Plot")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Top 5 autocorrelation peaks at lags: {np.argsort(res)[-5:]}")
    
    def fft_analysis(self):
        """Perform FFT analysis"""
        # Create synthetic sine wave for comparison
        t = np.linspace(-2 * np.pi, 2 * np.pi, len(self.sunspots))
        mid = np.ptp(self.sunspots) / 2
        sine = mid + mid * np.sin(np.sin(t))
        
        # Compute FFTs
        sine_fft = np.abs(fftshift(rfft(sine)))
        sunspot_fft = np.abs(fftshift(rfft(self.sunspots)))
        
        # Create frequency array
        freqs = np.fft.fftfreq(len(self.sunspots), d=1)
        freqs = freqs[:len(sunspot_fft)]
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # Time domain plots
        axes[0, 0].plot(self.years, self.sunspots, label="Sunspots")
        axes[0, 0].set_title("Sunspot Data")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(sine, label="Synthetic Sine", color='orange')
        axes[0, 1].set_title("Synthetic Sine Wave")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Magnitude spectrum
        axes[1, 0].plot(sunspot_fft, label="Sunspot FFT")
        axes[1, 0].set_title("Sunspot Magnitude Spectrum")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(sine_fft, label="Sine FFT", color='orange')
        axes[1, 1].set_title("Sine Magnitude Spectrum")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Power and phase spectrum
        axes[2, 0].plot(sunspot_fft ** 2, label="Power Spectrum")
        axes[2, 0].set_title("Sunspot Power Spectrum")
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        phase = np.angle(fftshift(rfft(self.sunspots)))
        axes[2, 1].plot(phase, label="Phase Spectrum")
        axes[2, 1].set_title("Sunspot Phase Spectrum")
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Top 5 FFT peaks for sine: {np.argsort(sine_fft)[-5:]}")
        print(f"Top 5 FFT peaks for sunspots: {np.argsort(sunspot_fft)[-5:]}")

class TimeSeriesPredictor:
    """Time series prediction models"""
    
    def __init__(self, data, years):
        self.data = data
        self.years = years
        self.models = {}
    
    def autoregressive_model(self, p1_lag=1, p2_lag=10, test_ratio=0.1):
        """Simple autoregressive model"""
        def model(params, x1, x10):
            p1, p10 = params
            return p1 * x1 + p10 * x10
        
        def error(params, data, x1, x10):
            return data - model(params, x1, x10)
        
        def fit(data):
            p0 = [0.5, 0.5]
            params = optimize.leastsq(error, p0, 
                                    args=(data[p2_lag:], data[p2_lag-1:-1], data[:-p2_lag]))[0]
            return params
        
        cutoff = int((1 - test_ratio) * len(self.data))
        params = fit(self.data[:cutoff])
        
        pred = (params[0] * self.data[cutoff-1:-1] + 
                params[1] * self.data[cutoff-p2_lag:-p2_lag])
        actual = self.data[cutoff:]
        
        self._plot_prediction(actual, pred, cutoff, "Autoregressive Model")
        self._calculate_metrics(actual, pred, "Autoregressive")
        
        return params, pred, actual
    
    def harmonic_model(self, test_ratio=0.1):
        """Harmonic model with multiple sine components"""
        def model(params, t):
            C, p1, f1, phi1, p2, f2, phi2, p3, f3, phi3 = params
            return (C + p1 * np.sin(f1 * t + phi1) + 
                   p2 * np.sin(f2 * t + phi2) + 
                   p3 * np.sin(f3 * t + phi3))
        
        def error(params, y, t):
            return y - model(params, t)
        
        def fit(y, t):
            p0 = [y.mean(), 0, 2*np.pi/11, 0, 0, 2*np.pi/22, 0, 0, 2*np.pi/100, 0]
            params = optimize.leastsq(error, p0, args=(y, t))[0]
            return params
        
        cutoff = int((1 - test_ratio) * len(self.data))
        params = fit(self.data[:cutoff], self.years[:cutoff])
        
        pred = model(params, self.years[cutoff:])
        actual = self.data[cutoff:]
        
        self._plot_prediction(actual, pred, cutoff, "Harmonic Model")
        self._calculate_metrics(actual, pred, "Harmonic")
        
        return params, pred, actual
    
    def arma_model(self, order=(2, 1), start_year='1975'):
        """ARMA model using statsmodels"""
        df = pd.DataFrame({'SUNACTIVITY': self.data}, 
                         index=pd.to_datetime(self.years, format='%Y'))
        
        try:
            model = sm.tsa.ARIMA(df, order=(order[0], 0, order[1])).fit()
            start_idx = df.index.get_loc(pd.to_datetime(start_year, format='%Y'))
            prediction = model.predict(start=start_idx, end=len(df)-1, dynamic=True)
            
            plt.figure(figsize=(12, 6))
            df[start_year:].plot(label="Actual", alpha=0.7)
            prediction_df = pd.DataFrame({'ARMA Prediction': prediction}, 
                                       index=df.index[start_idx:])
            prediction_df.plot(style='--', label='ARMA Prediction', linewidth=2, ax=plt.gca())
            plt.title(f"ARMA({order[0]},{order[1]}) Model Prediction")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
            return model, prediction
        except Exception as e:
            print(f"ARMA model failed: {e}")
            return None, None
        
        return model, prediction
    
    def _plot_prediction(self, actual, pred, cutoff, model_name):
        """Plot prediction results"""
        plt.figure(figsize=(12, 6))
        year_range = self.years[cutoff:]
        plt.plot(year_range, actual, 'o', label="Actual", alpha=0.7)
        plt.plot(year_range, pred, 'x', label="Prediction", markersize=8)
        plt.title(f"{model_name} Prediction Results")
        plt.xlabel("Year")
        plt.ylabel("Sunspot Activity")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def _calculate_metrics(self, actual, pred, model_name):
        """Calculate and print prediction metrics"""
        rmse = np.sqrt(np.mean((actual - pred) ** 2))
        mae = np.mean(np.abs(actual - pred))
        mape = 100 * np.mean(np.abs(actual - pred) / actual)
        r2 = r2_score(actual, pred)
        
        print(f"\n{model_name} Model Metrics:")
        print(f"RMSE: {rmse:.3f}")
        print(f"MAE: {mae:.3f}")
        print(f"MAPE: {mape:.3f}%")
        print(f"R²: {r2:.3f}")

class SignalProcessor:
    """Signal processing utilities"""
    
    @staticmethod
    def filter_comparison(data, years):
        """Compare different filtering methods"""
        plt.figure(figsize=(12, 8))
        plt.plot(years, data, label="Original", alpha=0.7)
        plt.plot(years, signal.medfilt(data, 11), label="Median Filter", linewidth=2)
        plt.plot(years, signal.wiener(data, 11), '--', label="Wiener Filter", linewidth=2)
        plt.plot(years, signal.detrend(data), label="Detrended", linewidth=2)
        
        plt.title("Signal Filtering Comparison")
        plt.xlabel("Year")
        plt.ylabel("Sunspot Activity")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

def main():
    """Main execution function"""
    print("🌞 Enhanced Sunspot Analysis Toolkit")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = SunspotAnalyzer()
    
    # Perform analyses
    print("\n1. Plotting moving averages...")
    analyzer.plot_moving_averages()
    
    print("\n2. Window function analysis...")
    analyzer.plot_window_functions()
    
    print("\n3. Autocorrelation analysis...")
    analyzer.autocorrelation_analysis()
    
    print("\n4. FFT analysis...")
    analyzer.fft_analysis()
    
    print("\n5. Signal filtering comparison...")
    SignalProcessor.filter_comparison(analyzer.sunspots, analyzer.years)
    
    # Prediction models
    print("\n6. Time series prediction models...")
    predictor = TimeSeriesPredictor(analyzer.sunspots, analyzer.years)
    
    print("\n6a. Autoregressive model...")
    ar_params, ar_pred, ar_actual = predictor.autoregressive_model()
    
    print("\n6b. Harmonic model...")
    harm_params, harm_pred, harm_actual = predictor.harmonic_model()
    
    print("\n6c. ARMA model...")
    arma_model, arma_pred = predictor.arma_model()
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
