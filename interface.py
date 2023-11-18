from kriging import Kriging, plot_2D, plot_3D, scatter_3d, plot_variogram
import streamlit as st
import pandas as pd
import time

# streamlit configuration
st.set_page_config(layout="centered")
st.title("Kriging 3D")
st.sidebar.title("Parametros")
st.sidebar.markdown("Seleccione los parametros para el kriging 3D")
st.sidebar.markdown("")

# lock widget
def callback():
    st.session_state.lock_widget = True

# session state variables
if 'lock_widget' not in st.session_state:
    st.session_state.lock_widget = False

if 'kriging' not in st.session_state:
    st.session_state.kriging = None

# data upload
uploaded_file = st.sidebar.file_uploader("Subir archivo csv", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.markdown("selección de datos")
    x = st.sidebar.selectbox("Seleccione X", data.columns, index=0, disabled=st.session_state.lock_widget)
    y = st.sidebar.selectbox("Seleccione Y", data.columns, index=1, disabled=st.session_state.lock_widget)
    z = st.sidebar.selectbox("Seleccione Z", data.columns, index=2, disabled=st.session_state.lock_widget)
    v = st.sidebar.selectbox("Seleccione Valor", data.columns, index=3, disabled=st.session_state.lock_widget)
    st.sidebar.markdown("parametros de grilla")
    separation = st.sidebar.slider(min_value=0.1, max_value=1.0, value=0.1, step=0.1, label="Separacion entre puntos", key='separation', disabled=st.session_state.lock_widget)
    round = st.sidebar.slider(min_value=0.1, max_value=1.0, value=0.5, step=0.1, label="Redondeo", key='round', disabled=st.session_state.lock_widget)
    st.sidebar.markdown("")

def main():
    tab1, tab2, tab3= st.tabs(['Datos', 'Variograma', 'Kriging'])
    
    with tab1:
        if uploaded_file is None:
            st.button(label='actualizar data', key='disableddata', disabled=True) 
        else:
            if st.button(label='actualizar data', key='enabledata', disabled=st.session_state.lock_widget, on_click=callback):
                with st.spinner('cargando...'):
                    kriging = Kriging(data[x], data[y], data[z], data[v], separation=separation, round=round)
                    kriging.scatter()
                    st.markdown(f'X min: {kriging.xmin}, X max: {kriging.xmax}')
                    st.markdown(f'Y min: {kriging.ymin}, Y max: {kriging.ymax}')
                    st.markdown(f'Z min: {kriging.zmin}, Z max: {kriging.zmax}')
                    st.markdown("")
                    st.markdown(f'X min rounded: {kriging.rounded_xmin}, X max rounded: {kriging.rounded_xmax}')
                    st.markdown(f'Y min rounded: {kriging.rounded_ymin}, Y max rounded: {kriging.rounded_ymax}')
                    st.markdown(f'Z min rounded: {kriging.rounded_zmin}, Z max rounded: {kriging.rounded_zmax}')
                    st.markdown("")
                    st.markdown(f"numero de datos en X: {kriging.x_points_number}")
                    st.markdown(f"numero de datos en Y: {kriging.y_points_number}")
                    st.markdown(f"numero de datos en Z: {kriging.z_points_number}")
                    st.markdown("")
                    scatter_3d(kriging)
                st.session_state.lock_widget = False
                st.session_state.kriging = kriging
    with tab2:
        variogram_model=st.selectbox("Seleccione modelo de variograma", ['linear', 'power', 'spherical', 'gaussian', 'exponential', 'hole-effect'], index=0, key='variogrammodel', disabled=st.session_state.lock_widget)
        nlags=st.slider(min_value=1, max_value=100, value=10, step=1, label="Numero de lags", key='nlags', disabled=st.session_state.lock_widget)
        if st.session_state.kriging is None:
            st.button(label='actualizar variograma', key='disabledvariogram', disabled=True) 
        else:
            if st.button(label='actualizar variograma', key='enablevariogram', disabled=st.session_state.lock_widget, on_click=callback):
                with st.spinner('cargando...'):
                    kriging = st.session_state.kriging
                    kriging.variogram(variogram_model=variogram_model, nlags=nlags)
                    st.success('variograma cargado')
                    variogram = plot_variogram(kriging.ok3d)
                    st.session_state.lock_widget = False
                    st.session_state.kriging = kriging
                    st.pyplot(variogram)
                    st.markdown(f'Range: {kriging.ok3d.variogram_model_parameters[1]}')
                    st.markdown(f'Sill: {kriging.ok3d.variogram_model_parameters[0] + kriging.ok3d.variogram_model_parameters[2]}')
                    st.markdown(f'Nugget: {kriging.ok3d.variogram_model_parameters[2]}')
                    
                    
    with tab3:
        if st.session_state.kriging is None:
            st.button(label='actualizar kriging', key='disabledkriging', disabled=True) 
        else:
            kriging = st.session_state.kriging
            y_value = st.slider(min_value=0, max_value=kriging.y_points_number-1, value=0, step=1, label="Y", key='yvalue', disabled=st.session_state.lock_widget)  
            tolerance = st.slider(min_value=0.0, max_value=10.0, value=1.0, step=0.5, label="Tolerancia", key='tolerance', disabled=st.session_state.lock_widget)
            if st.button(label='actualizar kriging', key='enablekriging', disabled=st.session_state.lock_widget, on_click=callback):
                with st.spinner('cargando...'):
                    kriging.execute(variogram_model=variogram_model, nlags=nlags)
                    st.session_state.kriging = kriging
                    st.session_state.lock_widget = False
                    st.success('kriging cargado')
                    plot_3D(kriging)
                    fig_2D = plot_2D(kriging, data, y_value=y_value, tolerance=tolerance)
                    st.pyplot(fig_2D)
                    st.session_state.lock_widget = False
        

if __name__ == "__main__":
    main()