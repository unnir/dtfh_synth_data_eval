import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import NearestNeighbors
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Preformatted, Image
from reportlab.lib import colors
from sklearn.preprocessing import OrdinalEncoder


@st.cache_data
def load_data(original_file, synthetic_file, n_samples, n_features):
    def read_file(file, file_type):
        try:
            content_preview = file.read(1024).decode()
            file.seek(0)
            st.write(f"{file_type} file preview:")
            st.code(content_preview)

            df = pd.read_csv(file)
            if df.empty:
                st.error(f"The {file_type} file is empty. Please check the file and try again.")
                return None
            st.success(f"Successfully read {file_type} file. Shape: {df.shape}")
            return df
        except pd.errors.EmptyDataError:
            st.error(f"No columns to parse from the {file_type} file. Please check if the file is not empty and is a valid CSV.")
            return None
        except Exception as e:
            st.error(f"Error reading the {file_type} file: {str(e)}")
            return None

    df_orig = read_file(original_file, "original")
    df_syn = read_file(synthetic_file, "synthetic")


    # Select features (prioritize numeric columns)
    numeric_features = df_orig.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df_orig.select_dtypes(exclude=[np.number]).columns.tolist()
    selected_features = numeric_features[:n_features] + categorical_features[:max(0, n_features - len(numeric_features))]
    
    
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        
    # Fit the encoder on the original dataframe
    encoder.fit(df_orig.select_dtypes(include=['object']))
        
    # Transform the original dataframe
    df_orig_encoded = df_orig.copy()
    df_orig_encoded[df_orig.select_dtypes(include=['object']).columns] = encoder.transform(df_orig.select_dtypes(include=['object']))
        
    # Transform the synthetic dataframe
    df_syn_encoded = df_syn.copy()
    df_syn_encoded[df_syn.select_dtypes(include=['object']).columns] = encoder.transform(df_syn.select_dtypes(include=['object']))
        
    # Convert encoded data back to dataframes
    df_orig = df_orig_encoded.copy()
    df_syn = df_syn_encoded.copy()


    if df_orig is None or df_syn is None:
        return None, None

    #st.write("Original data shape (before processing):", df_orig.shape)
    #st.write("Synthetic data shape (before processing):", df_syn.shape)
    
    # Ensure both datasets have the same columns
    common_columns = list(set(df_orig.columns) & set(df_syn.columns))
    df_orig = df_orig[common_columns]
    df_syn = df_syn[common_columns]
    
    # Handle case where n_samples is larger than available data
    n_samples = min(n_samples, len(df_orig), len(df_syn))
    df_orig = df_orig.sample(n=n_samples, random_state=42)
    df_syn = df_syn.sample(n=n_samples, random_state=42)
    
    # Handle case where n_features is larger than available features
    n_features = min(n_features, len(common_columns))
    

    
    df_orig = df_orig[selected_features]
    df_syn = df_syn[selected_features]
    
    #st.write("Data shape after processing:", df_orig.shape)
    #st.write(f"Using {n_samples} samples and {n_features} features")
    
    return df_orig, df_syn

def plot_histogram(df_orig, df_syn, column):
    fig, ax = plt.subplots()
    ax.hist(df_orig[column], bins=30, alpha=0.5, label="Original")
    ax.hist(df_syn[column], bins=30, alpha=0.5, label="Synthetic")
    ax.set_title(f"Histogram of {column}")
    ax.legend()
    return fig

def plot_scatter(df_orig, df_syn, x_col, y_col):
    fig, ax = plt.subplots()
    ax.scatter(df_orig[x_col], df_orig[y_col], alpha=0.5, label="Original")
    ax.scatter(df_syn[x_col], df_syn[y_col], alpha=0.5, label="Synthetic")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
    ax.legend()
    return fig

def plot_correlation_heatmap(df, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax, vmin=-1, vmax=1)
    ax.set_title(title)
    return fig

def plot_pca(df_orig, df_syn):
    scaler = StandardScaler()
    pca = PCA(n_components=2)
    
    combined_data = pd.concat([df_orig, df_syn], axis=0)
    scaled_data = scaler.fit_transform(combined_data)
    pca_result = pca.fit_transform(scaled_data)
    
    fig, ax = plt.subplots()
    ax.scatter(pca_result[:len(df_orig), 0], pca_result[:len(df_orig), 1], alpha=0.5, label="Original")
    ax.scatter(pca_result[len(df_orig):, 0], pca_result[len(df_orig):, 1], alpha=0.5, label="Synthetic")
    ax.set_xlabel("First Principal Component")
    ax.set_ylabel("Second Principal Component")
    ax.set_title("PCA: Original vs Synthetic Data")
    ax.legend()
    return fig

def compute_ks_statistic(df_orig, df_syn, column):
    statistic, p_value = ks_2samp(df_orig[column], df_syn[column])
    return statistic, p_value

def discriminative_measure(df_orig, df_syn):
    X = pd.concat([df_orig, df_syn])
    y = np.concatenate([np.ones(len(df_orig)), np.zeros(len(df_syn))])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'accuracy': accuracy,
            'report': classification_report(y_test, y_pred)
        }
    
    return results

def plot_discriminative_results(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    models = list(results.keys())
    accuracies = [result['accuracy'] for result in results.values()]
    colors = ['green' if acc <= 0.6 else 'orange' if acc <= 0.8 else 'red' for acc in accuracies]
    ax.bar(models, accuracies, color=colors)
    ax.set_ylabel('Accuracy')
    ax.set_title('Discriminative Measure Results')
    plt.xticks(rotation=45)
    for i, v in enumerate(accuracies):
        ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    return fig

def plot_decision_tree(df_orig, df_syn):
    X = pd.concat([df_orig, df_syn])
    y = np.concatenate([np.ones(len(df_orig)), np.zeros(len(df_syn))])
    
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(clf, filled=True, feature_names=X.columns, class_names=['Synthetic', 'Original'], ax=ax)
    plt.title("Decision Tree Visualization")
    return fig

def highlight_extremes(s):
    if pd.api.types.is_numeric_dtype(s):
        return ['background-color: red' if v <= s.quantile(0.1) else 'background-color: green' if v >= s.quantile(0.9) else '' for v in s]
    else:
        return [''] * len(s)

def compute_dcr(df_orig, df_syn):
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(df_orig)
    distances, _ = nn.kneighbors(df_syn)
    return distances.flatten()

def compute_duplicates(df_orig, df_syn):
    duplicates = df_syn[df_syn.isin(df_orig.to_dict('list')).all(axis=1)]
    return len(duplicates)

def compute_k_anonymity(df, sensitive_columns):
    return df.groupby(sensitive_columns).size().min()


def generate_comprehensive_pdf_report(report_data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Comprehensive Data Comparison Report", styles['Title']))
    story.append(Spacer(1, 12))

    # Manual Table of Contents
    story.append(Paragraph("Table of Contents", styles['Heading1']))
    toc_items = [
        "1. Basic Statistics",
        "2. Discriminative Measure",
        "3. Statistical Tests",
        "4. Privacy Measures",
        "5. Correlation Analysis",
        "6. Dimensionality Reduction",
        "7. Feature Distributions",
        "8. Feature Relationships",
        "9. Decision Tree Visualization"
    ]
    for item in toc_items:
        story.append(Paragraph(item, styles['Normal']))
    story.append(PageBreak())

    # 1. Basic Statistics
    story.append(Paragraph("1. Basic Statistics", styles['Heading1']))
    story.append(Paragraph(f"Original data shape: {report_data['basic_stats']['orig_shape']}", styles['Normal']))
    story.append(Paragraph(f"Synthetic data shape: {report_data['basic_stats']['syn_shape']}", styles['Normal']))
    story.append(Paragraph("Original Data Summary:", styles['Heading2']))
    story.append(create_table(report_data['basic_stats']['orig_describe']))
    story.append(Paragraph("Synthetic Data Summary:", styles['Heading2']))
    story.append(create_table(report_data['basic_stats']['syn_describe']))
    story.append(PageBreak())

    # 2. Discriminative Measure
    story.append(Paragraph("2. Discriminative Measure", styles['Heading1']))
    for model, result in report_data['discriminative_measure'].items():
        story.append(Paragraph(f"{model} Results:", styles['Heading2']))
        story.append(Paragraph(f"Accuracy: {result['accuracy']:.4f}", styles['Normal']))
        story.append(Paragraph("Classification Report:", styles['Normal']))
        story.append(Preformatted(result['report'], styles['Code']))
    story.append(create_image(report_data['plots']['discriminative'], width=400, height=300))
    story.append(PageBreak())

    # 3. Statistical Tests
    story.append(Paragraph("3. Statistical Tests", styles['Heading1']))
    story.append(Paragraph("Kolmogorov-Smirnov Test Results:", styles['Heading2']))
    ks_data = [['Feature', 'KS Statistic', 'p-value']] + [[feature, f"{statistic:.4f}", f"{p_value:.4f}"] for feature, (statistic, p_value) in report_data['ks_test'].items()]
    story.append(create_table(ks_data))
    story.append(PageBreak())


    # 4. Privacy Measures
    story.append(Paragraph("4. Privacy Measures", styles['Heading1']))
    story.append(Paragraph(f"Mean DCR: {report_data['privacy_measures']['mean_dcr']:.4f}", styles['Normal']))
    story.append(Paragraph(f"Number of duplicates: {report_data['privacy_measures']['n_duplicates']}", styles['Normal']))
    story.append(Paragraph(f"Percentage of duplicates: {report_data['privacy_measures']['duplicate_percentage']:.2f}%", styles['Normal']))
    if 'k_anonymity' in report_data['privacy_measures']:
        story.append(Paragraph(f"k-anonymity value: {report_data['privacy_measures']['k_anonymity']}", styles['Normal']))
    story.append(create_image(report_data['plots']['dcr'], width=400, height=300))
    story.append(PageBreak())

    # 5. Correlation Analysis
    story.append(Paragraph("5. Correlation Analysis", styles['Heading1']))
    for i, (title, plot) in enumerate(zip(['Original', 'Synthetic', 'Difference'], report_data['plots']['correlation'])):
        story.append(Paragraph(f"{title} Data Correlation:", styles['Heading2']))
        story.append(create_image(plot, width=400, height=300))
        if i < 2:  # Add page break after first two plots
            story.append(PageBreak())
    story.append(PageBreak())

    # 6. Dimensionality Reduction
    story.append(Paragraph("6. Dimensionality Reduction", styles['Heading1']))
    story.append(Paragraph("PCA Results:", styles['Heading2']))
    story.append(Paragraph(f"Explained variance ratio: {report_data['pca']['explained_variance_ratio']}", styles['Normal']))
    story.append(Paragraph(f"Cumulative explained variance ratio: {report_data['pca']['cumulative_explained_variance_ratio']}", styles['Normal']))
    story.append(create_image(report_data['plots']['pca'], width=400, height=300))
    story.append(PageBreak())

    # 7. Feature Distributions
    story.append(Paragraph("7. Feature Distributions", styles['Heading1']))
    for i, plot in enumerate(report_data['plots']['histograms']):
        story.append(Paragraph(f"Histogram for Feature {i+1}:", styles['Heading2']))
        story.append(create_image(plot, width=400, height=300))
        if (i+1) % 2 == 0:  # Add page break after every two plots
            story.append(PageBreak())

    # 8. Feature Relationships
    story.append(Paragraph("8. Feature Relationships", styles['Heading1']))
    for i, plot in enumerate(report_data['plots']['scatter']):
        story.append(Paragraph(f"Scatter Plot {i+1}:", styles['Heading2']))
        story.append(create_image(plot, width=400, height=300))
        if i < len(report_data['plots']['scatter']) - 1:  # Add page break between plots
            story.append(PageBreak())

    # 9. Decision Tree Visualization
    story.append(Paragraph("9. Decision Tree Visualization", styles['Heading1']))
    story.append(create_image(report_data['plots']['decision_tree'], width=500, height=300))

    # Build the PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def create_table(data):
    if isinstance(data, pd.DataFrame):
        # If it's a DataFrame, convert it to a list of lists
        data = [data.columns.tolist()] + data.values.tolist()
    elif not isinstance(data, list):
        # If it's neither a DataFrame nor a list, raise an error
        raise ValueError("Input must be a DataFrame or a list of lists")
    
    t = Table(data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    return t


def create_image(plt_figure, width, height):
    img_data = io.BytesIO()
    plt_figure.savefig(img_data, format='png')
    img_data.seek(0)
    img = Image(img_data, width=width, height=height)
    return img

def display_main_page(df_orig, df_syn):
    st.header("Data Overview")
    st.write("Original Data Shape:", df_orig.shape)
    st.write("Synthetic Data Shape:", df_syn.shape)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Data Sample")
        st.dataframe(df_orig.head())
    with col2:
        st.subheader("Synthetic Data Sample")
        st.dataframe(df_syn.head())

def display_analysis_page(page, df_orig, df_syn, features, sensitive_columns):
    if page == "Overview":
        st.header("Dataset Overview")
        st.subheader("Summary Statistics")
        tab1, tab2 = st.tabs(["Original Data", "Synthetic Data"])
        with tab1:
            st.dataframe(df_orig.describe())
        with tab2:
            st.dataframe(df_syn.describe())

    elif page == "Univariate Analysis":
        st.header("Univariate Analysis")
        feature = st.selectbox("Select a feature", features)
        
        st.subheader(f"Histogram: {feature}")
        fig = plot_histogram(df_orig, df_syn, feature)
        st.pyplot(fig)

        st.subheader(f"Summary Statistics: {feature}")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Original Data:")
            st.dataframe(df_orig[feature].describe().to_frame().T)
        with col2:
            st.write("Synthetic Data:")
            st.dataframe(df_syn[feature].describe().to_frame().T)

    elif page == "Bivariate Analysis":
        st.header("Bivariate Analysis")
        x_col = st.selectbox("Select X-axis feature", features)
        y_col = st.selectbox("Select Y-axis feature", features, index=1 if len(features) > 1 else 0)
        
        st.subheader(f"Scatter Plot: {x_col} vs {y_col}")
        fig = plot_scatter(df_orig, df_syn, x_col, y_col)
        st.pyplot(fig)

    elif page == "Correlation Analysis":
        st.header("Correlation Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Original Data", "Synthetic Data", "Difference"])
        with tab1:
            st.subheader("Original Data Correlation Heatmap")
            fig_orig = plot_correlation_heatmap(df_orig, "Original Data Correlation")
            st.pyplot(fig_orig)
        
        with tab2:
            st.subheader("Synthetic Data Correlation Heatmap")
            fig_syn = plot_correlation_heatmap(df_syn, "Synthetic Data Correlation")
            st.pyplot(fig_syn)

        with tab3:
            st.subheader("Correlation Difference (Original - Synthetic)")
            diff_corr = df_orig.corr() - df_syn.corr()
            fig_diff = plot_correlation_heatmap(diff_corr, "Correlation Difference")
            st.pyplot(fig_diff)

    elif page == "Dimensionality Reduction":
        st.header("Dimensionality Reduction")
        
        st.subheader("PCA: Original vs Synthetic Data")
        fig = plot_pca(df_orig, df_syn)
        st.pyplot(fig)

    elif page == "Statistical Tests":
        st.header("Statistical Tests")
        
        st.subheader("Kolmogorov-Smirnov Test")
        for feature in features:
            statistic, p_value = compute_ks_statistic(df_orig, df_syn, feature)
            
            st.write(f"Feature: {feature}")
            st.write(f"KS Statistic: {statistic:.4f}")
            st.write(f"p-value: {p_value:.4f}")
            
            if p_value < 0.05:
                st.markdown(f"<font color='red'>The distributions are significantly different (reject null hypothesis)</font>", unsafe_allow_html=True)
            else:
                st.markdown(f"<font color='green'>The distributions are not significantly different (fail to reject null hypothesis)</font>", unsafe_allow_html=True)
            
            st.markdown("---")

    elif page == "Discriminative Measure":
        st.header("Discriminative Measure")
        results = discriminative_measure(df_orig, df_syn)
        
        st.subheader("Classification Accuracies")
        for model, result in results.items():
            accuracy = result['accuracy']
            color = 'green' if accuracy <= 0.6 else 'orange' if accuracy <= 0.8 else 'red'
            st.markdown(f"{model} Accuracy: <font color='{color}'>{accuracy:.4f}</font>", unsafe_allow_html=True)
            st.text("Classification Report:")
            st.text(result['report'])
            st.markdown("---")
        
        st.subheader("Visualization of Results")
        fig = plot_discriminative_results(results)
        st.pyplot(fig)
        
        st.subheader("Decision Tree Visualization")
        fig_tree = plot_decision_tree(df_orig, df_syn)
        st.pyplot(fig_tree)
        
        st.write("An accuracy close to 0.5 indicates that the synthetic data is similar to the original data, "
                 "while an accuracy close to 1.0 suggests that the synthetic data is easily distinguishable from the original data.")
        
        overall_accuracy = np.mean([result['accuracy'] for result in results.values()])
        if overall_accuracy <= 0.6:
            st.markdown("<font color='green'>Overall, the synthetic data appears to be of good quality and privacy-preserving.</font>", unsafe_allow_html=True)
        elif overall_accuracy <= 0.8:
            st.markdown("<font color='orange'>The synthetic data shows moderate similarity to the original data. Some refinement may be needed.</font>", unsafe_allow_html=True)
        else:
            st.markdown("<font color='red'>The synthetic data is easily distinguishable from the original data. Significant improvements are needed.</font>", unsafe_allow_html=True)

    elif page == "Privacy Measures":
        st.header("Privacy Measures")
        
        st.subheader("Distance to Closest Record (DCR)")
        dcr_values = compute_dcr(df_orig, df_syn)
        fig, ax = plt.subplots()
        ax.hist(dcr_values, bins=30)
        ax.set_title("Distribution of DCR Values")
        ax.set_xlabel("DCR")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        st.write("DCR measures the distance between each synthetic record and its closest original record. "
                 "Higher values indicate better privacy as synthetic records are more distinct from original ones.")
        
        mean_dcr = np.mean(dcr_values)
        if mean_dcr > 1.0:
            st.markdown(f"<font color='green'>Mean DCR: {mean_dcr:.4f} - Good privacy preservation</font>", unsafe_allow_html=True)
        elif mean_dcr > 0.5:
            st.markdown(f"<font color='orange'>Mean DCR: {mean_dcr:.4f} - Moderate privacy preservation</font>", unsafe_allow_html=True)
        else:
            st.markdown(f"<font color='red'>Mean DCR: {mean_dcr:.4f} - Poor privacy preservation</font>", unsafe_allow_html=True)
        
        st.subheader("Number of Duplicates")
        n_duplicates = compute_duplicates(df_orig, df_syn)
        duplicate_percentage = (n_duplicates / len(df_syn)) * 100
        st.write(f"Number of exact duplicates in synthetic data: {n_duplicates}")
        st.write(f"Percentage of duplicates: {duplicate_percentage:.2f}%")
        st.write("This shows how many records in the synthetic data are exact copies of records in the original data. "
                 "Fewer duplicates generally indicate better privacy.")
        
        if duplicate_percentage <= 1:
            st.markdown("<font color='green'>Very low percentage of duplicates - Good privacy preservation</font>", unsafe_allow_html=True)
        elif duplicate_percentage <= 5:
            st.markdown("<font color='orange'>Moderate percentage of duplicates - Some privacy concerns</font>", unsafe_allow_html=True)
        else:
            st.markdown("<font color='red'>High percentage of duplicates - Significant privacy risk</font>", unsafe_allow_html=True)
        
        st.subheader("k-Anonymity")
        if sensitive_columns:
            k_anonymity = compute_k_anonymity(df_syn, sensitive_columns)
            st.write(f"k-anonymity value: {k_anonymity}")
            st.write("k-anonymity measures how many records in the dataset are indistinguishable from each other "
                     "based on the selected sensitive attributes. A higher k value indicates better privacy.")
            
            if k_anonymity >= 5:
                st.markdown("<font color='green'>Good k-anonymity - Strong privacy protection</font>", unsafe_allow_html=True)
            elif k_anonymity >= 2:
                st.markdown("<font color='orange'>Moderate k-anonymity - Some privacy protection</font>", unsafe_allow_html=True)
            else:
                st.markdown("<font color='red'>Poor k-anonymity - Weak privacy protection</font>", unsafe_allow_html=True)
        else:
            st.write("Please select sensitive columns in the sidebar to compute k-anonymity.")

    # Add the report generation button and functionality
    if st.sidebar.button("Generate Comprehensive PDF Report"):
        with st.spinner("Generating comprehensive PDF report... This may take a few minutes."):
            # Perform all analyses
            disc_results = discriminative_measure(df_orig, df_syn)
            ks_results = {feature: compute_ks_statistic(df_orig, df_syn, feature) for feature in features}
            dcr_values = compute_dcr(df_orig, df_syn)
            n_duplicates = compute_duplicates(df_orig, df_syn)
            duplicate_percentage = (n_duplicates / len(df_syn)) * 100
            
            privacy_results = {
                'mean_dcr': np.mean(dcr_values),
                'n_duplicates': n_duplicates,
                'duplicate_percentage': duplicate_percentage
            }
            
            if sensitive_columns:
                privacy_results['k_anonymity'] = compute_k_anonymity(df_syn, sensitive_columns)
            
            # Correlation analysis
            corr_orig = df_orig.corr()
            corr_syn = df_syn.corr()
            corr_diff = corr_orig - corr_syn
            
            # PCA
            scaler = StandardScaler()
            pca = PCA(n_components=2)
            combined_data = pd.concat([df_orig, df_syn], axis=0)
            scaled_data = scaler.fit_transform(combined_data)
            pca_result = pca.fit_transform(scaled_data)
            
            # Generate plots
            hist_plots = [plot_histogram(df_orig, df_syn, feature) for feature in features[:5]]  # Limit to first 5 features
            scatter_plots = [plot_scatter(df_orig, df_syn, features[0], feature) for feature in features[1:3]]  # 2 scatter plots
            corr_plots = [
                plot_correlation_heatmap(corr_orig, "Original Data Correlation"),
                plot_correlation_heatmap(corr_syn, "Synthetic Data Correlation"),
                plot_correlation_heatmap(corr_diff, "Correlation Difference")
            ]
            pca_plot = plot_pca(df_orig, df_syn)
            disc_plot = plot_discriminative_results(disc_results)
            decision_tree_plot = plot_decision_tree(df_orig, df_syn)
            dcr_plot = plt.figure()
            plt.hist(dcr_values, bins=30)
            plt.title("Distribution of DCR Values")
            plt.xlabel("DCR")
            plt.ylabel("Frequency")
            
            # Compile all results
            report_data = {
                'basic_stats': {
                    'orig_shape': df_orig.shape,
                    'syn_shape': df_syn.shape,
                    'orig_describe': df_orig.describe(),
                    'syn_describe': df_syn.describe()
                },
                'discriminative_measure': disc_results,
                'ks_test': ks_results,
                'privacy_measures': privacy_results,
                'correlation': {
                    'original': corr_orig,
                    'synthetic': corr_syn,
                    'difference': corr_diff
                },
                'pca': {
                    'explained_variance_ratio': pca.explained_variance_ratio_,
                    'cumulative_explained_variance_ratio': np.cumsum(pca.explained_variance_ratio_)
                },
                'plots': {
                    'histograms': hist_plots,
                    'scatter': scatter_plots,
                    'correlation': corr_plots,
                    'pca': pca_plot,
                    'discriminative': disc_plot,
                    'decision_tree': decision_tree_plot,
                    'dcr': dcr_plot
                },
                'sensitive_columns': sensitive_columns
            }
            
            # Generate PDF report
            pdf_buffer = generate_comprehensive_pdf_report(report_data)
            
            # Offer download
            st.sidebar.download_button(
                label="Download Comprehensive PDF Report",
                data=pdf_buffer,
                file_name="comprehensive_data_comparison_report.pdf",
                mime="application/pdf"
            )
            
            st.success("Comprehensive PDF report generated successfully!")

def main():
    st.set_page_config(page_title="Data Comparison Dashboard", layout="wide")
    st.title("Synthetic Data Evaluational Dashboard")

    # Custom CSS to adjust image positioning
    st.markdown("""
        <style>
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 0rem;
        }
        [data-testid="stSidebar"] [data-testid="stImage"] {
            margin-top: -5rem;
            margin-bottom: -5rem;
        }
        </style>
        """, unsafe_allow_html=True)

    st.sidebar.image("dtfh_logo.png", use_column_width=True)
    st.sidebar.header("Data Upload and Settings")

    original_file = st.sidebar.file_uploader("Upload Original Data CSV", type="csv")
    synthetic_file = st.sidebar.file_uploader("Upload Synthetic Data CSV", type="csv")
    
    if original_file is not None and synthetic_file is not None:
        try:
            df_orig_full = pd.read_csv(original_file)
            df_syn_full = pd.read_csv(synthetic_file)
        except Exception as e:
            st.error(f"Error reading the uploaded files: {str(e)}")
            return

        max_samples = min(len(df_orig_full), len(df_syn_full))
        max_features = min(df_orig_full.shape[1], df_syn_full.shape[1])
        
        n_samples = st.sidebar.number_input("Number of samples to use", min_value=100, max_value=max_samples, value=min(10000, max_samples), step=100)
        n_features = st.sidebar.number_input("Number of features to analyze", min_value=1, max_value=max_features, value=min(20, max_features), step=1)

        df_orig, df_syn = load_data(original_file, synthetic_file, n_samples, n_features)
        
        if df_orig is None or df_syn is None:
            return

        features = df_orig.columns.tolist()

        # Add this section to define sensitive_columns
        st.sidebar.header("Privacy Settings")
        sensitive_columns = st.sidebar.multiselect("Select sensitive columns for k-anonymity", features)

        st.sidebar.header("Navigation")
        page = st.sidebar.radio("Go to", ["Main", "Overview", "Univariate Analysis", "Bivariate Analysis", "Correlation Analysis", "Dimensionality Reduction", "Statistical Tests", "Discriminative Measure", "Privacy Measures"])

        # Display file preview and initial information only on the main page
        if page == "Main":
            display_main_page(df_orig, df_syn)
        else:
            display_analysis_page(page, df_orig, df_syn, features, sensitive_columns)

    else:
        st.write("Please upload both the original and synthetic data files to begin the analysis.")


if __name__ == "__main__":
    main()