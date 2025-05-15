import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import plotly.express as px


# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Análisis Sísmico",
    page_icon="🌍",
    layout="wide"
)

# Título y descripción
st.title("🌍 Earthquakes Dashboard")
st.markdown("""
Este dashboard permite explorar datos sísmicos de un mes completo. 
Utiliza los filtros y selectores en la barra lateral para personalizar tu análisis.
""")

# Variables globales para inicialización segura
filtered_df = None
df = None

# Función para cargar datos
@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_csv("all_month.csv")
        
        # Convertir columnas de fecha a datetime Y ELIMINAR ZONA HORARIA
        df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
        df['updated'] = pd.to_datetime(df['updated']).dt.tz_localize(None)
        
        # Crear columnas adicionales útiles
        df['day'] = df['time'].dt.date
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.day_name()
        df['week'] = df['time'].dt.isocalendar().week
        
        # Categorizar magnitudes
        conditions = [
            (df['mag'] < 2.0),
            (df['mag'] >= 2.0) & (df['mag'] < 4.0),
            (df['mag'] >= 4.0) & (df['mag'] < 6.0),
            (df['mag'] >= 6.0)
        ]
        choices = ['Menor (<2)', 'Leve (2-4)', 'Moderado (4-6)', 'Fuerte (6+)']
        df['magnitud_categoria'] = np.select(conditions, choices, default='No clasificado')
        
        return df
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None

# Traducción de días de la semana
days_translation = {
    'Monday': 'Lunes',
    'Tuesday': 'Martes',
    'Wednesday': 'Miércoles',
    'Thursday': 'Jueves',
    'Friday': 'Viernes',
    'Saturday': 'Sábado',
    'Sunday': 'Domingo'
}

# Esquema de colores para magnitudes
magnitude_colors = {
    'Menor (<2)': 'blue',
    'Leve (2-4)': 'green',
    'Moderado (4-6)': 'orange',
    'Fuerte (6+)': 'red'
}

# Función para asegurar tamaños positivos para marcadores
def ensure_positive(values, min_size=3):
    if isinstance(values, (pd.Series, np.ndarray, list)):
        return np.maximum(np.abs(values), min_size)
    else:
        return max(abs(values), min_size)

# Cargar datos
try:
    with st.spinner('Cargando datos...'):
        df = load_data()
        
    if df is not None and not df.empty:
        # Sidebar para filtros
        st.sidebar.header("Filtros")
        
        # Filtro de fechas
        min_date = df['time'].min().date()
        max_date = df['time'].max().date()
        
        date_range = st.sidebar.date_input(
            "Rango de fechas",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        
        # Convertir las fechas seleccionadas a datetime para filtrar
        if len(date_range) == 2:
            start_date, end_date = date_range
            
            # Convertir a objetos datetime sin zona horaria
            start_datetime = pd.Timestamp(start_date)
            end_datetime = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            
            # Filtrar el dataframe (ahora ambos son del mismo tipo)
            filtered_df = df[(df['time'] >= start_datetime) & (df['time'] <= end_datetime)].copy()
        else:
            filtered_df = df.copy()
        
        # Filtro de magnitud
        min_mag, max_mag = st.sidebar.slider(
            "Rango de magnitud",
            min_value=float(df['mag'].min()),
            max_value=float(df['mag'].max()),
            value=(float(df['mag'].min()), float(df['mag'].max())),
            step=0.1
        )
        filtered_df = filtered_df[(filtered_df['mag'] >= min_mag) & (filtered_df['mag'] <= max_mag)]
        
        # Filtro de profundidad
        min_depth, max_depth = st.sidebar.slider(
            "Rango de profundidad (km)",
            min_value=float(df['depth'].min()),
            max_value=float(df['depth'].max()),
            value=(float(df['depth'].min()), float(df['depth'].max())),
            step=5.0
        )
        filtered_df = filtered_df[(filtered_df['depth'] >= min_depth) & (filtered_df['depth'] <= max_depth)]
        
        # Filtro de tipo de evento
        event_types = df['type'].unique().tolist()
        selected_types = st.sidebar.multiselect(
            "Tipos de eventos",
            options=event_types,
            default=event_types
        )
        if selected_types:
            filtered_df = filtered_df[filtered_df['type'].isin(selected_types)]
        
        # Filtro de región (opcional)
        all_regions = sorted(df['place'].str.split(', ').str[-1].unique().tolist())
        selected_regions = st.sidebar.multiselect(
            "Filtrar por región",
            options=all_regions,
            default=[]
        )
        if selected_regions:
            region_mask = filtered_df['place'].str.contains('|'.join(selected_regions), case=False)
            filtered_df = filtered_df[region_mask]
        
        # Mostrar conteo de eventos filtrados
        st.sidebar.metric("Eventos seleccionados", len(filtered_df))
        
        # Opciones avanzadas en sidebar
        st.sidebar.markdown("---")
        st.sidebar.header("Opciones Avanzadas")
        
        show_clusters = st.sidebar.checkbox("Mostrar Análisis de Clusters", value=False)
        show_advanced_charts = st.sidebar.checkbox("Mostrar Gráficos Avanzados", value=False)
        
        # Verificar si hay datos después de filtrar
        if len(filtered_df) == 0:
            st.warning("No hay datos disponibles con los filtros seleccionados. Por favor, ajuste los filtros.")
        else:
            # Tabs principales para organizar el dashboard
            main_tabs = st.tabs(["📊 Resumen General", "🌐 Análisis Geográfico", "⏱️ Análisis Temporal", "📈 Análisis Avanzado"])
            
            # Tab 1: Resumen General
            with main_tabs[0]:
                # Métricas principales
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Total Eventos", len(filtered_df))
                col2.metric("Magnitud Promedio", f"{filtered_df['mag'].mean():.2f}")
                col3.metric("Magnitud Máxima", f"{filtered_df['mag'].max():.2f}")
                col4.metric("Profundidad Promedio", f"{filtered_df['depth'].mean():.2f} km")
                
                # Distribución de magnitudes y profundidades
                col_dist1, col_dist2 = st.columns(2)
                
                with col_dist1:
                    st.subheader("Distribución de Magnitudes")
                    
                    fig_mag = px.histogram(
                        filtered_df,
                        x="mag",
                        nbins=30,
                        color="magnitud_categoria",
                        color_discrete_map=magnitude_colors,
                        labels={"mag": "Magnitud", "count": "Frecuencia"},
                        title="Distribución de Magnitudes por Categoría"
                    )
                    fig_mag.update_layout(bargap=0.1)
                    st.plotly_chart(fig_mag, use_container_width=True)
                
                with col_dist2:
                    st.subheader("Distribución de Profundidades")
                    
                    fig_depth = px.histogram(
                        filtered_df,
                        x="depth",
                        nbins=30,
                        color="magnitud_categoria",
                        color_discrete_map=magnitude_colors,
                        labels={"depth": "Profundidad (km)", "count": "Frecuencia"},
                        title="Distribución de Profundidades por Categoría de Magnitud"
                    )
                    fig_depth.update_layout(bargap=0.1)
                    st.plotly_chart(fig_depth, use_container_width=True)
                
                # Relación Magnitud vs Profundidad
                st.subheader("Relación entre Magnitud y Profundidad")
                
                # Asegurar valores positivos para el tamaño
                size_values = ensure_positive(filtered_df['mag'])
                
                fig_scatter = px.scatter(
                    filtered_df,
                    x="depth",
                    y="mag",
                    color="magnitud_categoria",
                    size=size_values,  # Usar valores garantizados positivos
                    size_max=15,
                    opacity=0.7,
                    hover_name="place",
                    color_discrete_map=magnitude_colors,
                    labels={"depth": "Profundidad (km)", "mag": "Magnitud"}
                )
                
                fig_scatter.update_layout(height=500)
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Top 10 Regiones
                st.subheader("Top 10 Regiones con Mayor Actividad Sísmica")
                
                top_places = filtered_df['place'].value_counts().head(10).reset_index()
                top_places.columns = ['Región', 'Número de Eventos']
                
                fig_top = px.bar(
                    top_places,
                    x='Número de Eventos',
                    y='Región',
                    orientation='h',
                    text='Número de Eventos',
                    color='Número de Eventos',
                    color_continuous_scale='Viridis'
                )
                
                fig_top.update_traces(textposition='outside')
                fig_top.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
                
                st.plotly_chart(fig_top, use_container_width=True)
            
            # Tab 2: Análisis Geográfico
            with main_tabs[1]:
                geo_tabs = st.tabs(["Mapa de Eventos", "Mapa de Calor", "Análisis de Clusters"])
                
                # Tab 1: Mapa de Eventos
                with geo_tabs[0]:
                    st.subheader("Distribución Geográfica de Sismos")
                    
                    # Crear un mapa básico con px.scatter_geo en lugar de scatter_map
                    fig_map = px.scatter_geo(
                        filtered_df,
                        lat="latitude",
                        lon="longitude",
                        color="magnitud_categoria",
                        size=ensure_positive(filtered_df['mag']),  # Asegurar valores positivos
                        size_max=15,
                        hover_name="place",
                        hover_data={
                            "latitude": False,
                            "longitude": False,
                            "magnitud_categoria": False,
                            "mag": ":.2f",
                            "depth": ":.2f km",
                            "time": True,
                            "type": True
                        },
                        color_discrete_map=magnitude_colors,
                        projection="natural earth"
                    )
                    
                    fig_map.update_layout(
                        margin={"r": 0, "t": 0, "l": 0, "b": 0},
                        height=600,
                        geo=dict(
                            showland=True,
                            landcolor="lightgray",
                            showocean=True,
                            oceancolor="lightblue",
                            showcountries=True,
                            countrycolor="white",
                            showcoastlines=True,
                            coastlinecolor="white"
                        )
                    )
                    
                    st.plotly_chart(fig_map, use_container_width=True)
                    
                    # Lista de eventos significativos
                    st.subheader("Eventos Significativos (Magnitud ≥ 4.0)")
                    significant_events = filtered_df[filtered_df['mag'] >= 4.0].sort_values(by='mag', ascending=False)
                    
                    if not significant_events.empty:
                        st.dataframe(
                            significant_events[['time', 'place', 'mag', 'depth', 'type']],
                            use_container_width=True
                        )
                    else:
                        st.info("No hay eventos de magnitud ≥ 4.0 en el rango seleccionado.")
                
                # Tab 2: Mapa de Calor
                with geo_tabs[1]:
                    st.subheader("Mapa de Calor de Actividad Sísmica")
                    st.markdown("""
                    Este mapa de calor muestra las áreas con mayor concentración de actividad sísmica.
                    Las zonas más brillantes indican mayor densidad de eventos.
                    """)
                    
                    # Utilizar un enfoque de heatmap con scatter_geo y markersize para el mapa de calor
                    fig_heat = px.density_mapbox(
                        filtered_df,
                        lat="latitude",
                        lon="longitude",
                        z=ensure_positive(filtered_df['mag']),  # Asegurar que z sea positivo
                        radius=10,
                        center=dict(lat=filtered_df['latitude'].mean(), lon=filtered_df['longitude'].mean()),
                        zoom=1,
                        mapbox_style="open-street-map",
                        opacity=0.8
                    )
                    
                    fig_heat.update_layout(
                        margin={"r": 0, "t": 0, "l": 0, "b": 0},
                        height=600
                    )
                    
                    st.plotly_chart(fig_heat, use_container_width=True)
                    
                    # Mostrar eventos significativos en forma de tabla en lugar de mapa adicional
                    st.subheader("Eventos Significativos (Magnitud ≥ 4.0)")
                    strong_events = filtered_df[filtered_df['mag'] >= 4.0].sort_values(by='mag', ascending=False)
                    
                    if not strong_events.empty:
                        st.dataframe(
                            strong_events[['time', 'place', 'mag', 'depth', 'type']],
                            use_container_width=True
                        )
                    else:
                        st.info("No hay eventos de magnitud ≥ 4.0 en el rango seleccionado.")
                
                # Tab 3: Análisis de Clusters
                with geo_tabs[2]:
                    st.subheader("Análisis de Clusters Geográficos")
                    st.markdown("""
                    Este análisis identifica grupos de sismos que podrían estar relacionados geográficamente.
                    Utiliza el algoritmo DBSCAN que agrupa eventos basados en su proximidad espacial.
                    """)
                    
                    # Preparar los datos para clustering
                    if len(filtered_df) > 10:  # Asegurar que hay suficientes datos
                        # Seleccionar columnas para clustering
                        cluster_df = filtered_df[['latitude', 'longitude']].copy()
                        
                        # Escalar los datos
                        scaler = StandardScaler()
                        cluster_data = scaler.fit_transform(cluster_df)
                        
                        # Slider para ajustar los parámetros de DBSCAN
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            eps_distance = st.slider(
                                "Distancia máxima entre eventos para considerarlos vecinos (eps)",
                                min_value=0.05,
                                max_value=1.0,
                                value=0.2,
                                step=0.05
                            )
                        
                        with col2:
                            min_samples = st.slider(
                                "Número mínimo de eventos para formar un cluster",
                                min_value=2,
                                max_value=20,
                                value=5,
                                step=1
                            )
                        
                        # Ejecutar DBSCAN
                        dbscan = DBSCAN(eps=eps_distance, min_samples=min_samples)
                        cluster_result = dbscan.fit_predict(cluster_data)
                        filtered_df['cluster'] = cluster_result
                        
                        # Contar el número de clusters (excluyendo ruido, que es -1)
                        n_clusters = len(set(filtered_df['cluster'])) - (1 if -1 in filtered_df['cluster'] else 0)
                        n_noise = list(filtered_df['cluster']).count(-1)
                        
                        # Mostrar métricas
                        col1, col2 = st.columns(2)
                        col1.metric("Número de clusters identificados", n_clusters)
                        col2.metric("Eventos no agrupados (ruido)", n_noise)
                        
                        # Visualizar los clusters en un mapa
                        st.markdown("### Mapa de Clusters")
                        
                        # Crear una columna para mapear el cluster a string para mejor visualización
                        filtered_df['cluster_str'] = filtered_df['cluster'].apply(
                            lambda x: f'Cluster {x}' if x >= 0 else 'Sin Cluster'
                        )
                        
                        # Usar scatter_geo para el mapa de clusters
                        fig_cluster = px.scatter_geo(
                            filtered_df,
                            lat="latitude",
                            lon="longitude",
                            color="cluster_str",
                            size=ensure_positive(filtered_df['mag']),  # Asegurar valores positivos
                            size_max=15,
                            hover_name="place",
                            hover_data={
                                "latitude": False,
                                "longitude": False,
                                "cluster_str": False,
                                "mag": ":.2f",
                                "depth": ":.2f km",
                                "time": True
                            },
                            projection="natural earth"
                        )
                        
                        fig_cluster.update_layout(
                            margin={"r": 0, "t": 0, "l": 0, "b": 0},
                            height=500,
                            geo=dict(
                                showland=True,
                                landcolor="lightgray",
                                showocean=True,
                                oceancolor="lightblue",
                                showcountries=True,
                                countrycolor="white",
                                showcoastlines=True,
                                coastlinecolor="white"
                            )
                        )
                        
                        st.plotly_chart(fig_cluster, use_container_width=True)
                        
                        # Análisis de los clusters
                        if n_clusters > 0:
                            st.markdown("### Análisis de Clusters")
                            
                            # Tabla de resumen de clusters
                            cluster_summary = filtered_df[filtered_df['cluster'] >= 0].groupby('cluster_str').agg({
                                'mag': ['count', 'mean', 'max'],
                                'depth': ['mean', 'min', 'max']
                            }).reset_index()
                            
                            # Aplanar la tabla para una mejor visualización
                            cluster_summary.columns = [
                                'Cluster', 'Cantidad de Eventos', 'Magnitud Promedio', 'Magnitud Máxima',
                                'Profundidad Promedio', 'Profundidad Mínima', 'Profundidad Máxima'
                            ]
                            
                            st.dataframe(cluster_summary, use_container_width=True)
                            
                            # Seleccionar un cluster para análisis detallado
                            if n_clusters > 0:
                                cluster_options = [f'Cluster {i}' for i in range(n_clusters)]
                                if cluster_options:
                                    selected_cluster = st.selectbox(
                                        "Selecciona un cluster para ver detalles",
                                        options=cluster_options
                                    )
                                    
                                    # Filtrar datos para el cluster seleccionado
                                    cluster_data = filtered_df[filtered_df['cluster_str'] == selected_cluster]
                                    
                                    if not cluster_data.empty:
                                        # Mostrar eventos en el cluster seleccionado
                                        st.markdown(f"### Eventos en {selected_cluster}")
                                        st.dataframe(
                                            cluster_data[['time', 'place', 'mag', 'depth']].sort_values(by='time'),
                                            use_container_width=True
                                        )
                                        
                                        # Evolución temporal del cluster
                                        st.markdown(f"### Evolución temporal de {selected_cluster}")
                                        
                                        fig_timeline = px.scatter(
                                            cluster_data.sort_values('time'),
                                            x='time',
                                            y='mag',
                                            size=ensure_positive(cluster_data['mag']),  # Asegurar valores positivos
                                            color='depth',
                                            hover_name='place',
                                            title=f"Evolución temporal de eventos en {selected_cluster}",
                                            labels={'time': 'Fecha y Hora', 'mag': 'Magnitud', 'depth': 'Profundidad (km)'}
                                        )
                                        
                                        st.plotly_chart(fig_timeline, use_container_width=True)
                                        
                                        # Sugerencia de interpretación
                                        st.info("""
                                        **Interpretación de Clusters:**
                                        Los clusters pueden representar réplicas de un sismo principal, actividad en una falla específica, 
                                        o patrones de actividad sísmica en una región determinada.
                                        
                                        Observe la evolución temporal para identificar si se trata de eventos simultáneos o secuenciales.
                                        """)
                    else:
                        st.warning("No hay suficientes datos para realizar análisis de clustering con los filtros actuales.")
            
            # Tab 3: Análisis Temporal
            with main_tabs[2]:
                st.subheader("Análisis de Patrones Temporales")
                
                # Crear pestañas para diferentes análisis temporales
                temp_tab1, temp_tab2, temp_tab3 = st.tabs([
                    "Evolución Diaria", 
                    "Patrones Semanales",
                    "Patrones Horarios"
                ])
                
                # Tab 1: Evolución Diaria
                with temp_tab1:
                    st.subheader("Evolución de la Actividad Sísmica por Día")
                    
                    # Agrupar por día
                    try:
                        daily_counts = filtered_df.groupby('day').agg({
                            'id': 'count',
                            'mag': ['mean', 'max']
                        }).reset_index()
                        
                        daily_counts.columns = ['Fecha', 'Cantidad', 'Magnitud Media', 'Magnitud Máxima']
                        daily_counts['Fecha'] = pd.to_datetime(daily_counts['Fecha'])
                        
                        # Crear gráfico
                        fig_daily = go.Figure()
                        
                        # Añadir barras para cantidad de eventos
                        fig_daily.add_trace(go.Bar(
                            x=daily_counts['Fecha'],
                            y=daily_counts['Cantidad'],
                            name='Cantidad de Eventos',
                            marker_color='lightblue',
                            opacity=0.7
                        ))
                        
                        # Añadir línea para magnitud máxima
                        fig_daily.add_trace(go.Scatter(
                            x=daily_counts['Fecha'],
                            y=daily_counts['Magnitud Máxima'],
                            name='Magnitud Máxima',
                            mode='lines+markers',
                            marker=dict(color='red', size=6),
                            line=dict(width=2, dash='solid'),
                            yaxis='y2'
                        ))
                        
                        # Añadir línea para magnitud media
                        fig_daily.add_trace(go.Scatter(
                            x=daily_counts['Fecha'],
                            y=daily_counts['Magnitud Media'],
                            name='Magnitud Media',
                            mode='lines',
                            marker=dict(color='orange'),
                            line=dict(width=2, dash='dot'),
                            yaxis='y2'
                        ))
                        
                        # Configurar ejes y layout
                        fig_daily.update_layout(
                            title='Evolución Diaria de Eventos Sísmicos',
                            xaxis=dict(title='Fecha', tickformat='%d-%b'),
                            yaxis=dict(title='Cantidad de Eventos', side='left'),
                            yaxis2=dict(
                                title='Magnitud',
                                side='right',
                                overlaying='y',
                                range=[0, max(daily_counts['Magnitud Máxima']) + 0.5]
                            ),
                            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
                            hovermode='x unified',
                            height=500
                        )
                        
                        st.plotly_chart(fig_daily, use_container_width=True)
                        
                        # Agregar un análisis de tendencia
                        if len(daily_counts) > 5:
                            st.subheader("Análisis de Tendencia")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Calcular la tendencia de eventos por día
                                x = np.arange(len(daily_counts))
                                y = daily_counts['Cantidad']
                                z = np.polyfit(x, y, 1)
                                p = np.poly1d(z)
                                
                                trend_direction = "aumentando" if z[0] > 0 else "disminuyendo"
                                trend_value = abs(z[0])
                                
                                st.metric(
                                    "Tendencia de Eventos", 
                                    f"{trend_direction} ({trend_value:.2f} eventos/día)",
                                    delta=f"{trend_value:.2f}" if z[0] > 0 else f"-{trend_value:.2f}"
                                )
                            
                            with col2:
                                # Calcular la tendencia de magnitud por día
                                x = np.arange(len(daily_counts))
                                y = daily_counts['Magnitud Media']
                                z_mag = np.polyfit(x, y, 1)
                                p_mag = np.poly1d(z_mag)
                                
                                trend_direction_mag = "aumentando" if z_mag[0] > 0 else "disminuyendo"
                                trend_value_mag = abs(z_mag[0])
                                
                                st.metric(
                                    "Tendencia de Magnitud", 
                                    f"{trend_direction_mag} ({trend_value_mag:.3f} mag/día)",
                                    delta=f"{trend_value_mag:.3f}" if z_mag[0] > 0 else f"-{trend_value_mag:.3f}"
                                )
                    except Exception as e:
                        st.error(f"Error en el análisis de evolución diaria: {e}")
                
                # Tab 2: Patrones Semanales
                with temp_tab2:
                    try:
                        # Traducir días de semana
                        filtered_df['day_name'] = filtered_df['day_of_week'].map(days_translation)
                        
                        # Ordenar días de la semana correctamente
                        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        ordered_days = [days_translation[day] for day in days_order]
                        
                        # Agrupar por día de la semana
                        dow_data = filtered_df.groupby('day_name').agg({
                            'id': 'count',
                            'mag': ['mean', 'max']
                        }).reset_index()
                        
                        # Renombrar columnas
                        dow_data.columns = ['Día', 'Cantidad', 'Magnitud Media', 'Magnitud Máxima']
                        
                        # Ordenar días
                        dow_data['Día_ordenado'] = pd.Categorical(dow_data['Día'], categories=ordered_days, ordered=True)
                        dow_data = dow_data.sort_values('Día_ordenado')
                        
                        # Crear gráfico
                        fig_dow = px.bar(
                            dow_data,
                            x='Día',
                            y='Cantidad',
                            color='Magnitud Media',
                            text='Cantidad',
                            title='Distribución de Eventos por Día de la Semana',
                            color_continuous_scale='Viridis',
                            labels={'Cantidad': 'Número de Eventos', 'Magnitud Media': 'Magnitud Promedio'}
                        )
                        
                        fig_dow.update_traces(textposition='outside')
                        fig_dow.update_layout(height=400)
                        
                        st.plotly_chart(fig_dow, use_container_width=True)
                        
                        # Añadir una interpretación
                        st.markdown("""
                        ### Análisis de patrones semanales
                        
                        Este gráfico muestra cómo se distribuyen los eventos sísmicos a lo largo de la semana.
                        Patrones significativos podrían indicar:
                        
                        - Posible influencia de actividades humanas (ej: explosiones controladas en días laborables)
                        - Tendencias que merecen investigación adicional
                        - Note que en fenómenos naturales generalmente no se esperan patrones semanales
                        """)
                        
                        # Crear un heatmap de actividad por día de la semana y semana del mes
                        st.subheader("Mapa de Calor: Actividad por Semana y Día")
                        
                        # Añadir columna de número de semana relativa dentro del período
                        filtered_df['week_num'] = filtered_df['time'].dt.isocalendar().week
                        min_week = filtered_df['week_num'].min()
                        filtered_df['rel_week'] = filtered_df['week_num'] - min_week + 1
                        
                        # Agrupar por semana relativa y día de la semana
                        heatmap_weekly = filtered_df.groupby(['rel_week', 'day_name']).size().reset_index(name='count')
                        
                        # Pivotar para crear el formato para el heatmap
                        pivot_weekly = pd.pivot_table(
                            heatmap_weekly, 
                            values='count', 
                            index='day_name', 
                            columns='rel_week',
                            fill_value=0
                        )
                        
                        # Reordenar los días
                        if set(ordered_days).issubset(set(pivot_weekly.index)):
                            pivot_weekly = pivot_weekly.reindex(ordered_days)
                        
                        # Crear heatmap
                        fig_weekly_heat = px.imshow(
                            pivot_weekly,
                            labels=dict(x="Semana", y="Día de la Semana", color="Número de Eventos"),
                            x=[f"Semana {i}" for i in pivot_weekly.columns],
                            y=pivot_weekly.index,
                            color_continuous_scale="YlOrRd",
                            title="Mapa de Calor: Actividad Sísmica por Semana y Día"
                        )
                        
                        st.plotly_chart(fig_weekly_heat, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error en el análisis de patrones semanales: {e}")
                
                # Tab 3: Patrones Horarios
                with temp_tab3:
                    try:
                        st.subheader("Distribución de Eventos por Hora del Día")
                        
                        # Agrupar por hora
                        hourly_counts = filtered_df.groupby('hour').agg({
                            'id': 'count',
                            'mag': ['mean', 'max']
                        }).reset_index()
                        
                        # Renombrar columnas
                        hourly_counts.columns = ['Hora', 'Cantidad', 'Magnitud Media', 'Magnitud Máxima']
                        
                        # Crear gráfico de barras para distribución por hora
                        fig_hourly = px.bar(
                            hourly_counts,
                            x='Hora',
                            y='Cantidad',
                            color='Magnitud Media',
                            title="Distribución de eventos sísmicos por hora del día",
                            labels={"Hora": "Hora del día (UTC)", "Cantidad": "Número de eventos"},
                            color_continuous_scale='Viridis',
                            text='Cantidad'
                        )
                        
                        fig_hourly.update_traces(textposition='outside')
                        fig_hourly.update_layout(height=400)
                        
                        st.plotly_chart(fig_hourly, use_container_width=True)
                        
                        # Mapa de calor por hora y día de la semana
                        st.subheader("Mapa de Calor: Actividad por Hora y Día de la Semana")
                        
                        # Asegurarnos de que 'day_name' existe
                        if 'day_name' not in filtered_df.columns:
                            filtered_df['day_name'] = filtered_df['day_of_week'].map(days_translation)
                        
                        # Agrupar por hora y día de la semana
                        heatmap_data = filtered_df.groupby(['day_name', 'hour']).size().reset_index(name='count')
                        
                        # Pivotar para crear el formato para el heatmap
                        pivot_data = pd.pivot_table(
                            heatmap_data, 
                            values='count', 
                            index='day_name', 
                            columns='hour',
                            fill_value=0
                        )
                        
                        # Reordenar los días
                        ordered_days = [days_translation[day] for day in days_order]
                        if set(ordered_days).issubset(set(pivot_data.index)):
                            pivot_data = pivot_data.reindex(ordered_days)
                        
                        # Crear heatmap
                        fig_heatmap = px.imshow(
                            pivot_data,
                            labels=dict(x="Hora del Día (UTC)", y="Día de la Semana", color="Número de Eventos"),
                            x=[f"{h}:00" for h in range(24)],
                            y=pivot_data.index,
                            color_continuous_scale="YlOrRd",
                            title="Mapa de Calor: Actividad Sísmica por Hora y Día de la Semana"
                        )
                        
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                        
                        st.markdown("""
                        ### Interpretación del mapa de calor
                        
                        Este mapa de calor muestra la distribución de eventos sísmicos por hora y día de la semana.
                        
                        - Las celdas más oscuras indican momentos con mayor actividad sísmica
                        - Patrones horizontales sugieren horas del día con mayor actividad
                        - Patrones verticales indican días de la semana con más eventos
                        - Celdas aisladas de color intenso pueden indicar eventos especiales o clusters temporales
                        """)
                    except Exception as e:
                        st.error(f"Error en el análisis de patrones horarios: {e}")
            
            # Tab 4: Análisis Avanzado
            with main_tabs[3]:
                adv_tab1, adv_tab2, adv_tab3 = st.tabs([
                    "Correlaciones", 
                    "Magnitud por Región", 
                    "Comparativas"
                ])
                
                # Tab 1: Correlaciones
                with adv_tab1:
                    try:
                        st.subheader("Matriz de Correlación")
                        
                        # Seleccionar variables para correlación
                        corr_cols = ['mag', 'depth', 'rms', 'gap', 'horizontalError', 'depthError']
                        
                        # Filtrar columnas que existen en el DataFrame
                        valid_cols = [col for col in corr_cols if col in filtered_df.columns]
                        
                        if len(valid_cols) > 1:
                            corr_df = filtered_df[valid_cols].dropna()
                            
                            if len(corr_df) > 1:  # Asegurar que hay suficientes datos para correlación
                                corr_matrix = corr_df.corr()
                                
                                fig_corr = px.imshow(
                                    corr_matrix,
                                    text_auto=True,
                                    color_continuous_scale="RdBu_r",
                                    title="Matriz de Correlación",
                                    aspect="auto"
                                )
                                
                                st.plotly_chart(fig_corr, use_container_width=True)
                                
                                st.markdown("""
                                **Interpretación de la matriz de correlación:**
                                - Valores cercanos a 1 indican correlación positiva fuerte
                                - Valores cercanos a -1 indican correlación negativa fuerte
                                - Valores cercanos a 0 indican poca o ninguna correlación
                                """)
                                
                                # Añadir análisis de correlación detallado
                                st.subheader("Análisis Detallado de Correlaciones")
                                
                                # Encontrar correlaciones significativas
                                significant_corr = []
                                for i in range(len(valid_cols)):
                                    for j in range(i+1, len(valid_cols)):
                                        corr_val = corr_matrix.iloc[i, j]
                                        if abs(corr_val) > 0.3:  # Umbral para correlación significativa
                                            significant_corr.append({
                                                'Variables': f"{valid_cols[i]} vs {valid_cols[j]}",
                                                'Correlación': corr_val,
                                                'Fuerza': 'Fuerte' if abs(corr_val) > 0.7 else 'Moderada' if abs(corr_val) > 0.5 else 'Débil',
                                                'Tipo': 'Positiva' if corr_val > 0 else 'Negativa'
                                            })
                                
                                if significant_corr:
                                    significant_df = pd.DataFrame(significant_corr)
                                    significant_df = significant_df.sort_values('Correlación', key=abs, ascending=False)
                                    
                                    st.dataframe(significant_df, use_container_width=True)
                                    
                                    # Visualizar la correlación más fuerte
                                    if len(significant_df) > 0:
                                        top_corr = significant_df.iloc[0]
                                        var1, var2 = top_corr['Variables'].split(' vs ')
                                        
                                        st.subheader(f"Visualización de Correlación: {top_corr['Variables']}")
                                        
                                        fig_scatter_corr = px.scatter(
                                            filtered_df,
                                            x=var1,
                                            y=var2,
                                            color='magnitud_categoria',
                                            size=ensure_positive(filtered_df['mag']),  # Usar valores positivos garantizados
                                            hover_name='place',
                                            title=f"Correlación {top_corr['Tipo']} {top_corr['Fuerza']} (r={top_corr['Correlación']:.2f})",
                                            color_discrete_map=magnitude_colors
                                        )
                                        
                                        fig_scatter_corr.update_layout(height=500)
                                        
                                        st.plotly_chart(fig_scatter_corr, use_container_width=True)
                                else:
                                    st.info("No se encontraron correlaciones significativas entre las variables analizadas.")
                            else:
                                st.warning("No hay suficientes datos para calcular las correlaciones.")
                        else:
                            st.warning("No hay suficientes columnas numéricas para calcular correlaciones.")
                    except Exception as e:
                        st.error(f"Error en el análisis de correlaciones: {e}")
                
                # Tab 2: Magnitud por Región
                with adv_tab2:
                    try:
                        st.subheader("Análisis de Magnitud por Región")
                        
                        # Extraer regiones principales
                        filtered_df['region'] = filtered_df['place'].str.split(', ').str[-1]
                        region_stats = filtered_df.groupby('region').agg({
                            'id': 'count',
                            'mag': ['mean', 'max', 'min'],
                            'depth': 'mean'
                        }).reset_index()
                        
                        # Aplanar columnas multinivel
                        region_stats.columns = ['Región', 'Cantidad', 'Magnitud Media', 'Magnitud Máxima', 'Magnitud Mínima', 'Profundidad Media']
                        
                        # Filtrar regiones con suficientes eventos
                        min_events = st.slider("Mínimo de eventos por región", 1, 50, 5)
                        filtered_regions = region_stats[region_stats['Cantidad'] >= min_events].sort_values('Magnitud Media', ascending=False)
                        
                        # Visualizar
                        if not filtered_regions.empty:
                            fig_regions = px.bar(
                                filtered_regions.head(15),  # Top 15 regiones
                                x='Región',
                                y='Magnitud Media',
                                error_y=filtered_regions.head(15)['Magnitud Máxima'] - filtered_regions.head(15)['Magnitud Media'],
                                color='Cantidad',
                                hover_data=['Cantidad', 'Magnitud Máxima', 'Profundidad Media'],
                                title='Magnitud Media por Región (Top 15)',
                                color_continuous_scale='Viridis'
                            )
                            
                            fig_regions.update_layout(height=500, xaxis_tickangle=-45)
                            st.plotly_chart(fig_regions, use_container_width=True)
                            
                            # Mostrar tabla detallada
                            st.dataframe(
                                filtered_regions.sort_values('Cantidad', ascending=False),
                                use_container_width=True
                            )
                        else:
                            st.warning(f"No hay regiones con al menos {min_events} eventos. Intenta reducir el mínimo.")
                    except Exception as e:
                        st.error(f"Error en el análisis de magnitud por región: {e}")
                
                # Tab 3: Comparativas
                with adv_tab3:
                    try:
                        st.subheader("Análisis Comparativo")
                        
                        # Columnas numéricas disponibles
                        numeric_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()
                        numeric_cols = [col for col in numeric_cols if col not in ['cluster', 'rel_week', 'week_num']]
                        
                        # Seleccionar variables para comparar
                        if len(numeric_cols) >= 2:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                x_variable = st.selectbox(
                                    "Variable X",
                                    options=numeric_cols,
                                    index=numeric_cols.index('mag') if 'mag' in numeric_cols else 0
                                )
                            
                            with col2:
                                y_variable = st.selectbox(
                                    "Variable Y",
                                    options=numeric_cols,
                                    index=numeric_cols.index('depth') if 'depth' in numeric_cols else min(1, len(numeric_cols)-1)
                                )
                            
                            # Crear gráfico de dispersión personalizado
                            fig_custom = px.scatter(
                                filtered_df,
                                x=x_variable,
                                y=y_variable,
                                color='magnitud_categoria',
                                size=ensure_positive(filtered_df['mag']),  # Valores positivos
                                hover_name='place',
                                title=f"Relación entre {x_variable} y {y_variable}",
                                color_discrete_map=magnitude_colors,
                                trendline='ols'  # Añadir línea de tendencia
                            )
                            
                            fig_custom.update_layout(height=500)
                            st.plotly_chart(fig_custom, use_container_width=True)
                            
                            # Análisis por categoría
                            st.subheader("Estadísticas por Categoría de Magnitud")
                            
                            # Agrupar por categoría de magnitud
                            cat_stats = filtered_df.groupby('magnitud_categoria').agg({
                                'id': 'count',
                                'mag': ['mean', 'std'],
                                'depth': ['mean', 'std'],
                                'rms': 'mean'
                            }).reset_index()
                            
                            # Aplanar columnas
                            cat_stats.columns = [
                                'Categoría', 'Cantidad', 'Magnitud Media', 'Desviación Mag', 
                                'Profundidad Media', 'Desviación Prof', 'RMS Medio'
                            ]
                            
                            # Ordenar categorías
                            cat_order = ['Menor (<2)', 'Leve (2-4)', 'Moderado (4-6)', 'Fuerte (6+)']
                            cat_stats['Orden'] = cat_stats['Categoría'].map({cat: i for i, cat in enumerate(cat_order)})
                            cat_stats = cat_stats.sort_values('Orden').drop('Orden', axis=1)
                            
                            # Visualizar estadísticas
                            st.dataframe(cat_stats, use_container_width=True)
                            
                            # Gráfico de barras comparativo
                            fig_cats = go.Figure()
                            
                            # Añadir barras para cantidad
                            fig_cats.add_trace(go.Bar(
                                x=cat_stats['Categoría'],
                                y=cat_stats['Cantidad'],
                                name='Cantidad',
                                marker_color='lightskyblue',
                                opacity=0.7
                            ))
                            
                            # Añadir línea para profundidad media
                            fig_cats.add_trace(go.Scatter(
                                x=cat_stats['Categoría'],
                                y=cat_stats['Profundidad Media'],
                                name='Profundidad Media (km)',
                                mode='lines+markers',
                                marker=dict(color='darkred', size=8),
                                line=dict(width=2),
                                yaxis='y2'
                            ))
                            
                            # Configurar ejes y layout
                            fig_cats.update_layout(
                                title='Comparación de Cantidad y Profundidad por Categoría',
                                xaxis=dict(title='Categoría de Magnitud'),
                                yaxis=dict(title='Cantidad de Eventos', side='left'),
                                yaxis2=dict(
                                    title='Profundidad Media (km)',
                                    side='right',
                                    overlaying='y'
                                ),
                                legend=dict(x=0.01, y=0.99),
                                barmode='group',
                                height=400
                            )
                            
                            st.plotly_chart(fig_cats, use_container_width=True)
                        else:
                            st.warning("No hay suficientes columnas numéricas para realizar el análisis comparativo.")
                    except Exception as e:
                        st.error(f"Error en el análisis comparativo: {e}")
            
            # Tabla de datos (expandible)
            with st.expander("Ver datos en formato tabular"):
                try:
                    # Columnas disponibles para mostrar
                    display_cols = [col for col in ['time', 'place', 'mag', 'depth', 'type', 'magType', 'rms'] if col in filtered_df.columns]
                    
                    # Opciones de ordenamiento
                    sort_col = st.selectbox(
                        "Ordenar por",
                        options=display_cols,
                        index=0
                    )
                    
                    sort_order = st.radio(
                        "Orden",
                        options=['Descendente', 'Ascendente'],
                        index=0,
                        horizontal=True
                    )
                    
                    # Ordenar datos
                    sorted_df = filtered_df.sort_values(
                        by=sort_col,
                        ascending=(sort_order == 'Ascendente')
                    )
                    
                    # Mostrar tabla
                    st.dataframe(
                        sorted_df[display_cols],
                        use_container_width=True
                    )
                    
                    # Opción para descargar datos filtrados
                    csv = sorted_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Descargar datos filtrados (CSV)",
                        data=csv,
                        file_name="datos_sismicos_filtrados.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"Error al mostrar la tabla de datos: {e}")
    else:
        st.error("No se pudieron cargar los datos sísmicos. Verifique que el archivo 'all_month.csv' exista y tenga el formato correcto.")

except Exception as e:
    st.error(f"Error al cargar o procesar los datos: {e}")
    st.info("Verifique que el archivo 'all_month.csv' esté disponible y tenga el formato correcto.")

# Información del dashboard
st.sidebar.markdown("---")
st.sidebar.info("""
**Acerca de este Dashboard**

Este dashboard muestra datos sísmicos de aproximadamente un mes de actividad.
Desarrollado con Streamlit y Plotly Express.
""")