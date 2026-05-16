# AutoDL: Intelligent Deep Learning Model Recommender

![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Overview

**AutoDL** is a production-ready, intelligent deep learning model recommender system that automates the entire machine learning pipeline. It leverages advanced meta-learning, Bayesian optimization, and explainable AI techniques to intelligently recommend and train optimal deep learning architectures for your datasets.

This system is designed for data scientists and ML engineers who want to accelerate model development by automatically discovering, training, and evaluating the best neural network configurations without extensive manual tuning.

## 🎯 Key Features

- **Automated Task Detection**: Intelligently identifies whether your dataset is suited for classification, regression, or other ML tasks
- **Intelligent Model Recommendation**: Suggests optimal deep learning architectures based on dataset characteristics and meta-features
- **Hyperparameter Optimization**: Uses Optuna with Bayesian optimization for efficient hyperparameter tuning
- **Multi-Model Training**: Simultaneously trains multiple model configurations with early stopping to prevent overfitting
- **Explainable AI (XAI)**: Generates SHAP and LIME visualizations to interpret model predictions
- **Natural Language Insights**: Leverages Gemini or Claude APIs to generate human-readable analysis reports
- **User-Friendly Interface**: Interactive Streamlit web application for seamless workflow
- **Multi-Modal Support**: Handles tabular, image, and text datasets with adaptive preprocessing

## 🏗️ Architecture

The system follows a modular architecture with clear separation of concerns:

```
AutoDL/
├── data/
│   ├── uploads/          # User-uploaded datasets
│   └── models/           # Trained model checkpoints and artifacts
├── src/
│   ├── data_handler.py      # Data loading, validation, and preprocessing
│   ├── task_detector.py     # Data type and task classification
│   ├── meta_extractor.py    # Meta-feature extraction (PyMFE)
│   ├── model_selector.py    # Architecture recommendation engine
│   ├── trainer.py           # Training orchestration with Optuna
│   ├── explainer.py         # SHAP and LIME explanation generation
│   ├── llm_generator.py     # Natural language report generation
│   ├── flaml_trainer.py     # FLAML integration for AutoML
│   ├── surrogate_selector.py # Surrogate model selection
│   └── utils.py             # Utility functions and helpers
├── scripts/
│   ├── train_surrogate.py    # Surrogate model training
│   ├── test_surrogate.py     # Surrogate model validation
│   ├── benchmark_paper.py    # Benchmark evaluation scripts
│   ├── statistical_tests.py  # Statistical analysis
│   └── xai_analysis.py       # XAI workflow analysis
├── app.py                  # Streamlit application entry point
├── config.yaml             # Configuration and API settings
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 📊 Component Details

| Component | Purpose |
|-----------|---------|
| **data_handler.py** | Loads, validates, and preprocesses data; handles train-test splitting |
| **task_detector.py** | Automatically detects task type and data characteristics |
| **meta_extractor.py** | Extracts meta-features using PyMFE for meta-learning |
| **model_selector.py** | Generates architecture recommendations and training configurations |
| **trainer.py** | Orchestrates model training with Optuna optimization and evaluation |
| **explainer.py** | Generates SHAP and LIME explanations for model interpretability |
| **llm_generator.py** | Creates natural language summaries using Gemini or Claude APIs |
| **utils.py** | Provides logging, plotting, and utility functions |

## 🚀 Installation & Setup

### Prerequisites

- **Python**: 3.10 or higher
- **pip**: Package installer for Python
- **Virtual Environment**: Recommended for dependency isolation

### Step 1: Clone the Repository

```bash
git clone https://github.com/AnushkaSomawanshi/AutoDL-Intelligent-Model-Recommneder.git
cd AutoDL-Intelligent-Model-Recommneder
```

### Step 2: Create Virtual Environment

```bash
# Using Python venv
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure API Keys

Edit `config.yaml` and add your API credentials:

```yaml
api:
  claude_api_key: "your-claude-api-key"
  gemini_api_key: "your-gemini-api-key"
  use_llm: "gemini"  # or "claude"
```

**To obtain API keys:**
- **Claude API**: Visit [Anthropic Console](https://console.anthropic.com/)
- **Gemini API**: Visit [Google AI Studio](https://aistudio.google.com/)

## 🎮 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Complete Workflow (7-Step Pipeline)

#### **Step 1: Data Upload & Preview**
- Upload your dataset (CSV, Excel, or image files)
- Preview data statistics and distributions
- Validate data quality

#### **Step 2: Automatic Detection**
- System automatically detects task type (classification, regression, etc.)
- Extracts relevant meta-features using PyMFE
- Analyzes dataset characteristics

#### **Step 3: Model Recommendation**
- Reviews recommended deep learning architectures
- Displays multiple configuration options
- Shows rationale for recommendations based on meta-features

#### **Step 4: Model Training**
- Select configurations to train
- Monitor real-time training progress
- View live metrics (loss, accuracy, etc.)
- Automatic early stopping to prevent overfitting

#### **Step 5: Performance Evaluation**
- View best model metrics (accuracy, precision, recall, F1-score)
- Analyze confusion matrix
- Compare performance across trained configurations

#### **Step 6: Explainability Analysis**
- Generate SHAP force plots and summary plots
- Create LIME local explanations
- Visualize feature importance

#### **Step 7: Report Generation**
- Generate AI-powered insights using Gemini or Claude
- Compile comprehensive PDF report
- Download results and artifacts

## ⚙️ Configuration

### config.yaml Parameters

```yaml
# API Configuration
api:
  claude_api_key: "your-key"
  gemini_api_key: "your-key"
  use_llm: "gemini"              # LLM provider

# Training Configuration
training:
  epochs: 100                    # Maximum epochs per model
  batch_size: 32                 # Training batch size
  learning_rate: 0.001           # Initial learning rate
  early_stopping_patience: 10    # Epochs before early stopping
  optuna_trials: 50              # Number of Optuna optimization trials

# Paths
paths:
  data_dir: "data"
  models_dir: "data/models"
  uploads_dir: "data/uploads"
  plots_dir: "data/models/plots"
```

## 📁 Output Files

All generated artifacts are automatically saved:

| Output | Location |
|--------|----------|
| **Model Checkpoints** | `data/models/*.pth` |
| **Preprocessors** | `data/models/*.joblib` |
| **Training Curves** | `data/models/plots/training_curve.png` |
| **Confusion Matrix** | `data/models/plots/confusion_matrix.png` |
| **SHAP Explanations** | `data/models/plots/shap_*.png` |
| **LIME Explanation** | `data/models/plots/lime_explanation.html` |
| **PDF Report** | Downloaded from web interface |
| **Metrics JSON** | `metrics.json` |

## 🔬 Supported Dataset Types

### Tabular Data
- ✅ Fully implemented
- Supports classification and regression tasks
- Automatic feature engineering and preprocessing

### Image Data
- ✅ Detection support in UI
- Convolutional architectures recommended
- Image classification pipeline ready

### Text Data
- ✅ Detection support in UI
- NLP architectures recommended
- Text classification pipeline ready

## 📦 Dependencies

Key libraries and versions:

```
streamlit>=1.28.0          # Web UI framework
torch>=2.0.0               # Deep learning framework
optuna>=3.0.0              # Hyperparameter optimization
shap>=0.42.0               # SHAP explanations
lime>=0.2.0                # LIME explanations
PyMFE>=0.4.0               # Meta-feature extraction
scikit-learn>=1.0.0        # ML utilities
pandas>=1.5.0              # Data manipulation
numpy>=1.23.0              # Numerical computing
matplotlib>=3.5.0          # Plotting library
anthropic>=0.7.0           # Claude API client
google-generativeai>=0.3.0 # Gemini API client
```

See `requirements.txt` for complete dependencies list.

## 🛠️ Development

### Running Tests

```bash
# Run surrogate model tests
python scripts/test_surrogate.py

# Run statistical analysis
python scripts/statistical_tests.py

# Generate benchmarks
python scripts/benchmark_paper.py
```

### Code Structure

- **Modular design**: Each component is independent and testable
- **Logging**: Comprehensive logging for debugging
- **Error handling**: User-friendly error messages and graceful fallbacks
- **Type hints**: Full type annotations for code clarity

## 📈 Performance & Benchmarks

The system has been evaluated on multiple datasets:

- **Tabular Classification**: Achieves >95% accuracy on standard benchmarks
- **Training Speed**: Trains and evaluates 50+ configurations in minutes
- **Explanation Quality**: Generates human-interpretable SHAP and LIME visualizations
- **Report Generation**: Natural language insights generated in <30 seconds

See `benchmark_paper_results.csv` and `statistical_analysis_report.md` for detailed results.

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **API Keys Invalid** | Verify keys in `config.yaml` and API service status |
| **Memory Issues** | Reduce batch size in `config.yaml` |
| **Slow Training** | Decrease `optuna_trials` or use GPU acceleration |
| **Data Format Error** | Ensure CSV/Excel files have proper headers |

## 📝 Notes

- **Stability**: Tabular training, evaluation, and XAI are fully production-ready
- **UI Feedback**: All long-running operations include progress indicators and spinners
- **Robustness**: Comprehensive error handling with user-friendly messages
- **Extensibility**: Modular architecture allows easy addition of new architectures

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Anushka Somawanshi**
- GitHub: [@AnushkaSomawanshi](https://github.com/AnushkaSomawanshi)

## 🙏 Acknowledgments

- PyMFE for meta-feature extraction
- Optuna for hyperparameter optimization
- SHAP and LIME for explainability
- Streamlit for the web framework
- PyTorch for deep learning capabilities

## 📞 Support

For questions, issues, or suggestions:
- Open an [Issue](https://github.com/AnushkaSomawanshi/AutoDL-Intelligent-Model-Recommneder/issues)
- Check existing documentation in `ARCHITECTURE.md`
- Review analysis reports in `statistical_analysis_report.md`

---

**Last Updated**: May 2026  
**Version**: 1.0.0
