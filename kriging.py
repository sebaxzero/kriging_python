from pykrige.ok3d import OrdinaryKriging3D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

class Kriging():
    def __init__(self, x, y, z, val, round = 0.5, separation = 0.1):
        self.x = x
        self.y = y
        self.z = z
        self.val = val
        self.round = round
        self.separation = separation
        
    def __narray__(self):
        self.x_array = np.array(self.x)
        self.y_array = np.array(self.y)
        self.z_array = np.array(self.z)
        self.val_array = np.array(self.val)
    
    def __limit__(self):
        self.xmin = min(self.x)
        self.xmax = max(self.x)
        self.ymin = min(self.y)
        self.ymax = max(self.y)
        self.zmin = min(self.z)
        self.zmax = max(self.z)
    
    def __rounded_limit__(self):
        self.__limit__()
        self.rounded_xmin = round(self.xmin / self.round) * self.round - self.round
        self.rounded_xmax = round(self.xmax / self.round) * self.round + self.round
        self.rounded_ymin = round(self.ymin / self.round) * self.round - self.round
        self.rounded_ymax = round(self.ymax / self.round) * self.round + self.round
        self.rounded_zmin = round(self.zmin / self.round) * self.round - self.round
        self.rounded_zmax = round(self.zmax / self.round) * self.round + self.round
    
    def __points_number__(self):
        self.__rounded_limit__()
        self.x_points_number = int((self.rounded_xmax - self.rounded_xmin) / self.separation)
        self.y_points_number = int((self.rounded_ymax - self.rounded_ymin) / self.separation)
        self.z_points_number = int((self.rounded_zmax - self.rounded_zmin) / self.separation)

    def __grid__(self):
        self.__points_number__()
        self.gridx = np.linspace(self.rounded_xmin, self.rounded_xmax, self.x_points_number)
        self.gridy = np.linspace(self.rounded_ymin, self.rounded_ymax, self.y_points_number)
        self.gridz = np.linspace(self.rounded_zmin, self.rounded_zmax, self.z_points_number)
        self.xi, self.yi, self.zi = np.meshgrid(self.gridz, self.gridy, self.gridx, indexing='ij')
    
    def scatter(self):
        self.__narray__()
        self.__grid__()
        
    def variogram(self, variogram_model = "spherical", nlags = 10):
        self.scatter()
        self.ok3d = OrdinaryKriging3D(self.x_array, self.y_array, self.z_array, self.val_array, variogram_model=variogram_model, nlags=nlags, verbose=True, enable_plotting=True)

    def execute(self, variogram_model = "spherical", nlags = 10):
        self.variogram(variogram_model, nlags)
        self.k3d, self.sigma3d = self.ok3d.execute('grid', self.gridx, self.gridy, self.gridz)
        return self.k3d, self.sigma3d

def plot_2D(kriging: Kriging, df, y_value=40, tolerance=1):
    y_grid = kriging.gridy[y_value]
    filtered_data = df[(df['Y'] >= y_grid - tolerance) & (df['Y'] <= y_grid + tolerance)]
    grouped_data = filtered_data.groupby('HID').agg({'X': 'mean', 'Z': 'mean', 'Combinacion Lineal': 'mean'})

    x_mean = grouped_data['X'].values
    z_mean = grouped_data['Z'].values
    val_mean = grouped_data['Combinacion Lineal'].values

    fig_2d, ax = plt.subplots()
    im = ax.imshow(kriging.k3d[:, y_value, :], origin='lower',
              extent=(kriging.gridx[0], kriging.gridx[-1], kriging.gridz[0], kriging.gridz[-1]))
    ax.scatter(x_mean, z_mean, c=val_mean, edgecolors='black', linewidths=1, marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title(f'Y = {y_grid} ± {tolerance}')
    cbar = fig_2d.colorbar(im)  # Usamos 'im' en lugar de 'ax' para la barra de color
    return fig_2d
    
def plot_3D(kriging):
    x = kriging.xi
    y = kriging.yi
    z = kriging.zi
    k = kriging.k3d
    fig_plot_3d = px.scatter_3d(x=x.ravel(),
                    y=y.ravel(),
                    z=z.ravel(),
                    color=k.ravel(),
                    opacity=0.7,
                    title='Kriging 3D Interpolation',
                    color_continuous_scale='Jet',
                    color_continuous_midpoint=np.mean(k))
    st.plotly_chart(fig_plot_3d)

def scatter_3d(kriging):
    x = kriging.x_array
    y = kriging.y_array
    z = kriging.z_array
    k = kriging.val_array
    fig_scatter_3d = px.scatter_3d(x=x.ravel(),
                        y=y.ravel(),
                        z=z.ravel(),
                        color=k.ravel(),
                        opacity=0.7,
                        title='Data',
                        color_continuous_scale='Jet',
                        color_continuous_midpoint=np.mean(k))
    st.plotly_chart(fig_scatter_3d)
    
def plot_variogram(self):
    fig_variogram, ax = plt.subplots()
    ax.plot(self.lags, self.semivariance, "r*")
    ax.plot(self.lags, self.variogram_function(self.variogram_model_parameters, self.lags), "k-")
    ax.set_title("Variograma")
    ax.set_xlabel("Lag [m]")
    ax.set_ylabel("Semivarianza")
    return fig_variogram
    
if __name__ == '__main__':
    df = pd.read_csv('data.csv', sep=',')
    x, y, z, val = df['X'], df['Y'], df['Z'], df['Combinacion Lineal']
    k = Kriging(x, y, z, val, 0.5, 0.1)
    k3d, sigma3d = k.execute()
    plot_2D(k, df, 40, 0.2)
    plot_3D(k)
    
    



        
    