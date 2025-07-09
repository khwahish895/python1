import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🤖 ML Playground",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .stSelectbox > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header"> Interactive ML Playground</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title(" Control Panel")
st.sidebar.markdown("---")

# Project selection
project_type = st.sidebar.selectbox(
    " Choose ML Task",
    ["Classification", "Regression", "Dataset Explorer", "Model Comparison"]
)

def generate_synthetic_data(task_type, n_samples=1000):
    """Generate synthetic data for demonstration"""
    if task_type == "Classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=3,
            random_state=42
        )
        feature_names = ['Feature_A', 'Feature_B', 'Feature_C', 'Feature_D']
        return pd.DataFrame(X, columns=feature_names), y
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=4,
            noise=0.1,
            random_state=42
        )
        feature_names = ['Feature_A', 'Feature_B', 'Feature_C', 'Feature_D']
        return pd.DataFrame(X, columns=feature_names), y

def load_real_dataset(dataset_name):
    """Load real datasets"""
    if dataset_name == "Iris":
        data = load_iris()
        return pd.DataFrame(data.data, columns=data.feature_names), data.target
    elif dataset_name == "Wine":
        data = load_wine()
        return pd.DataFrame(data.data, columns=data.feature_names), data.target

def create_animated_scatter(df, x_col, y_col, color_col, title):
    """Create animated scatter plot"""
    fig = px.scatter(
        df, x=x_col, y=y_col, color=color_col,
        title=title,
        template="plotly_dark",
        color_continuous_scale="viridis"
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    return fig

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    corr_matrix = df.corr()
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Matrix",
        template="plotly_dark",
        color_continuous_scale="RdBu"
    )
    return fig

def create_feature_importance_plot(model, feature_names):
    """Create feature importance plot"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        fig = px.bar(
            x=feature_names,
            y=importances,
            title=" Feature Importance",
            template="plotly_dark",
            color=importances,
            color_continuous_scale="viridis"
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        return fig
    return None

# Main content based on selection
if project_type == "Classification":
    st.header(" Classification Challenge")
    
    # Dataset selection
    dataset_choice = st.sidebar.selectbox(
        " Dataset",
        ["Synthetic Data", "Iris", "Wine"]
    )
    
    # Load data
    if dataset_choice == "Synthetic Data":
        X_df, y = generate_synthetic_data("Classification")
    else:
        X_df, y = load_real_dataset(dataset_choice)
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "🤖 Choose Model",
        ["Random Forest", "Logistic Regression", "SVM", "K-Nearest Neighbors"]
    )
    
    # Display data info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Samples", len(X_df))
    with col2:
        st.metric("🎯 Features", len(X_df.columns))
    with col3:
        st.metric("🏷️ Classes", len(np.unique(y)))
    
    # Data visualization
    st.subheader("📈 Data Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Scatter plot
        if len(X_df.columns) >= 2:
            fig_scatter = create_animated_scatter(
                pd.concat([X_df, pd.Series(y, name='Target')], axis=1),
                X_df.columns[0], X_df.columns[1], 'Target',
                f"🎨 {X_df.columns[0]} vs {X_df.columns[1]}"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Correlation heatmap
        fig_corr = create_correlation_heatmap(X_df)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Model training
    if st.sidebar.button("🚀 Train Model"):
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=0.2, random_state=42
        )
        
        # Model selection
        if model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_choice == "Logistic Regression":
            model = LogisticRegression(random_state=42)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        elif model_choice == "SVM":
            model = SVC(random_state=42)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        else:  # KNN
            model = KNeighborsClassifier(n_neighbors=5)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Results
        accuracy = accuracy_score(y_test, y_pred)
        
        st.subheader("🎉 Model Results")
        st.success(f"🎯 Accuracy: {accuracy:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            title="🔍 Confusion Matrix",
            template="plotly_dark",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            fig_importance = create_feature_importance_plot(model, X_df.columns)
            if fig_importance:
                st.plotly_chart(fig_importance, use_container_width=True)

elif project_type == "Regression":
    st.header("📈 Regression Analysis")
    
    # Generate regression data
    X_df, y = generate_synthetic_data("Regression")
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "🤖 Choose Model",
        ["Random Forest", "Linear Regression", "Gradient Boosting"]
    )
    
    # Display data info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Samples", len(X_df))
    with col2:
        st.metric("🎯 Features", len(X_df.columns))
    with col3:
        st.metric("📍 Target Range", f"{y.min():.2f} - {y.max():.2f}")
    
    # Data visualization
    st.subheader("📊 Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Target distribution
        fig_hist = px.histogram(
            y, nbins=30,
            title="🎯 Target Distribution",
            template="plotly_dark",
            color_discrete_sequence=['#667eea']
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Correlation heatmap
        df_with_target = pd.concat([X_df, pd.Series(y, name='Target')], axis=1)
        fig_corr = create_correlation_heatmap(df_with_target)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Model training
    if st.sidebar.button("🚀 Train Model"):
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=0.2, random_state=42
        )
        
        # Model selection
        if model_choice == "Random Forest":
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_choice == "Linear Regression":
            model = LinearRegression()
        else:  # Gradient Boosting
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Results
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.subheader("🎉 Model Results")
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"📊 R² Score: {r2:.3f}")
        with col2:
            st.info(f"📉 MSE: {mse:.3f}")
        
        # Prediction vs Actual plot
        fig_pred = px.scatter(
            x=y_test, y=y_pred,
            title="🎯 Predictions vs Actual",
            template="plotly_dark",
            color_discrete_sequence=['#667eea']
        )
        fig_pred.add_shape(
            type="line",
            x0=y_test.min(), y0=y_test.min(),
            x1=y_test.max(), y1=y_test.max(),
            line=dict(color="red", dash="dash")
        )
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Feature importance
        fig_importance = create_feature_importance_plot(model, X_df.columns)
        if fig_importance:
            st.plotly_chart(fig_importance, use_container_width=True)

elif project_type == "Dataset Explorer":
    st.header("🔍 Dataset Explorer")
    
    # Dataset selection
    dataset_choice = st.sidebar.selectbox(
        "📊 Choose Dataset",
        ["Iris", "Wine", "Synthetic Classification"]
    )
    
    # Load data
    if dataset_choice == "Synthetic Classification":
        X_df, y = generate_synthetic_data("Classification")
    else:
        X_df, y = load_real_dataset(dataset_choice)
    
    # Combine features and target
    full_df = pd.concat([X_df, pd.Series(y, name='Target')], axis=1)
    
    # Dataset overview
    st.subheader("📋 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔢 Rows", len(full_df))
    with col2:
        st.metric("📊 Columns", len(full_df.columns))
    with col3:
        st.metric("🎯 Target Classes", len(np.unique(y)))
    with col4:
        st.metric("❌ Missing Values", full_df.isnull().sum().sum())
    
    # Data preview
    st.subheader("👀 Data Preview")
    st.dataframe(full_df.head(10), use_container_width=True)
    
    # Statistical summary
    st.subheader("📊 Statistical Summary")
    st.dataframe(full_df.describe(), use_container_width=True)
    
    # Interactive visualizations
    st.subheader("🎨 Interactive Visualizations")
    
    viz_type = st.selectbox(
        "Choose Visualization",
        ["Pairplot", "Distribution Plot", "Box Plot", "Violin Plot"]
    )
    
    if viz_type == "Pairplot":
        # Create pairplot using plotly
        fig = px.scatter_matrix(
            full_df,
            dimensions=X_df.columns,
            color='Target',
            title="🔄 Feature Pairplot",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Distribution Plot":
        feature = st.selectbox("Select Feature", X_df.columns)
        fig = px.histogram(
            full_df, x=feature, color='Target',
            title=f"📊 Distribution of {feature}",
            template="plotly_dark",
            marginal="box"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Box Plot":
        feature = st.selectbox("Select Feature", X_df.columns)
        fig = px.box(
            full_df, x='Target', y=feature,
            title=f"📦 Box Plot of {feature} by Target",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Violin Plot":
        feature = st.selectbox("Select Feature", X_df.columns)
        fig = px.violin(
            full_df, x='Target', y=feature,
            title=f"🎻 Violin Plot of {feature} by Target",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

elif project_type == "Model Comparison":
    st.header("⚔️ Model Battle Arena")
    
    # Dataset selection
    dataset_choice = st.sidebar.selectbox(
        "📊 Dataset",
        ["Iris", "Wine", "Synthetic Data"]
    )
    
    # Load data
    if dataset_choice == "Synthetic Data":
        X_df, y = generate_synthetic_data("Classification")
    else:
        X_df, y = load_real_dataset(dataset_choice)
    
    # Model selection
    selected_models = st.sidebar.multiselect(
        "🤖 Select Models to Compare",
        ["Random Forest", "Logistic Regression", "SVM", "K-Nearest Neighbors"],
        default=["Random Forest", "Logistic Regression"]
    )
    
    if st.sidebar.button("🥊 Start Battle!"):
        if len(selected_models) < 2:
            st.warning("Please select at least 2 models to compare!")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_df, y, test_size=0.2, random_state=42
            )
            
            results = {}
            
            for model_name in selected_models:
                # Model initialization
                if model_name == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    X_train_scaled, X_test_scaled = X_train, X_test
                elif model_name == "Logistic Regression":
                    model = LogisticRegression(random_state=42)
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                elif model_name == "SVM":
                    model = SVC(random_state=42)
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                else:  # KNN
                    model = KNeighborsClassifier(n_neighbors=5)
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                
                # Train and evaluate
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'model': model,
                    'predictions': y_pred
                }
            
            # Display results
            st.subheader("🏆 Battle Results")
            
            # Create comparison bar chart
            model_names = list(results.keys())
            accuracies = [results[name]['accuracy'] for name in model_names]
            
            fig_comparison = px.bar(
                x=model_names,
                y=accuracies,
                title="🏅 Model Accuracy Comparison",
                template="plotly_dark",
                color=accuracies,
                color_continuous_scale="viridis"
            )
            fig_comparison.update_layout(
                yaxis_title="Accuracy",
                xaxis_title="Models"
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Winner announcement
            best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
            st.success(f"🎉 Winner: {best_model} with {results[best_model]['accuracy']:.3f} accuracy!")
            
            # Detailed results
            st.subheader("📊 Detailed Results")
            for model_name, result in results.items():
                with st.expander(f"📋 {model_name} Details"):
                    st.write(f"**Accuracy:** {result['accuracy']:.3f}")
                    
                    # Confusion matrix
                    cm = confusion_matrix(y_test, result['predictions'])
                    fig_cm = px.imshow(
                        cm,
                        text_auto=True,
                        aspect="auto",
                        title=f"Confusion Matrix - {model_name}",
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        🤖 Built with Streamlit & Scikit-learn | 
        🎨 Interactive ML Dashboard | 
        🚀 Explore, Learn, Experiment!
    </div>
    """,
    unsafe_allow_html=True
)