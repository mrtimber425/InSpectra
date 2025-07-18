# ai_analysis_engine.py
# AI-powered analysis engine with Hugging Face model support

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import json
from datetime import datetime
from pathlib import Path
import warnings
import tempfile
import os

# Hugging Face imports
try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        AutoModelForCausalLM, pipeline, BertTokenizer, BertModel,
        GPT2LMHeadModel, GPT2Tokenizer, DistilBertTokenizer, DistilBertModel
    )
    from huggingface_hub import hf_hub_download, list_repo_files

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers library not available. Install with: pip install transformers torch")

# Additional AI libraries
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch")

warnings.filterwarnings('ignore')


class AIAnalysisEngine:
    """AI-powered data analysis engine with Hugging Face integration"""

    def __init__(self):
        self.loaded_models = {}
        self.tokenizers = {}
        self.pipelines = {}
        self.model_configs = {}
        self.analysis_history = []

        # Check if required libraries are available
        if not TRANSFORMERS_AVAILABLE:
            print("Warning: Transformers not available. AI analysis features will be limited.")
        if not TORCH_AVAILABLE:
            print("Warning: PyTorch not available. Some models may not work.")

        # Pre-defined analysis templates
        self.analysis_templates = {
            "Data Insights": {
                "prompt": "Analyze this dataset and provide key insights about patterns, trends, and notable characteristics in the data.",
                "requires_text": True
            },
            "Pattern Detection": {
                "prompt": "Identify significant patterns and correlations in this dataset. Focus on relationships between variables and unexpected findings.",
                "requires_text": True
            },
            "Anomaly Detection": {
                "prompt": "Examine this data for anomalies, outliers, and unusual patterns that might indicate data quality issues or interesting phenomena.",
                "requires_text": True
            },
            "Trend Analysis": {
                "prompt": "Analyze temporal trends in this dataset. Identify growth patterns, cyclical behavior, and significant changes over time.",
                "requires_text": True
            },
            "Predictive Analysis": {
                "prompt": "Based on the patterns in this data, provide insights about potential future trends and make data-driven predictions.",
                "requires_text": True
            }
        }

    def is_available(self) -> bool:
        """Check if AI analysis is available"""
        return TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE

    def load_huggingface_model(self, model_name: str, model_type: str = "auto") -> bool:
        """Load model from Hugging Face Hub"""
        if not self.is_available():
            raise Exception("Required libraries (transformers, torch) not available")

        try:
            print(f"Loading model: {model_name}")

            # Determine model type if auto
            if model_type == "auto":
                model_type = self._detect_model_type(model_name)

            # Load tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            except Exception as e:
                print(f"Warning: Could not load tokenizer for {model_name}: {e}")
                tokenizer = None

            # Load model based on type
            if model_type == "text-generation":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                # Create pipeline
                pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

            elif model_type == "text-classification":
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

            elif model_type == "feature-extraction":
                model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                pipe = pipeline("feature-extraction", model=model, tokenizer=tokenizer)

            else:
                # Generic approach
                model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                pipe = None

            # Store the loaded components
            self.loaded_models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            if pipe:
                self.pipelines[model_name] = pipe

            self.model_configs[model_name] = {
                "type": model_type,
                "loaded_at": datetime.now(),
                "parameters": model.num_parameters() if hasattr(model, 'num_parameters') else "Unknown"
            }

            print(f"Successfully loaded model: {model_name}")
            return True

        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            return False

    def load_local_model(self, model_path: str) -> bool:
        """Load model from local directory"""
        if not self.is_available():
            raise Exception("Required libraries not available")

        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model path does not exist: {model_path}")

            model_name = model_path.name

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model
            model = AutoModel.from_pretrained(str(model_path))

            # Store components
            self.loaded_models[model_name] = model
            self.tokenizers[model_name] = tokenizer

            self.model_configs[model_name] = {
                "type": "local",
                "path": str(model_path),
                "loaded_at": datetime.now(),
                "parameters": model.num_parameters() if hasattr(model, 'num_parameters') else "Unknown"
            }

            print(f"Successfully loaded local model: {model_name}")
            return True

        except Exception as e:
            print(f"Error loading local model: {str(e)}")
            return False

    def _detect_model_type(self, model_name: str) -> str:
        """Detect model type based on model name"""
        model_name_lower = model_name.lower()

        if any(keyword in model_name_lower for keyword in ["gpt", "t5", "bart", "generator"]):
            return "text-generation"
        elif any(keyword in model_name_lower for keyword in ["classifier", "sentiment", "emotion", "roberta"]):
            return "text-classification"
        elif any(keyword in model_name_lower for keyword in ["bert", "distilbert", "feature"]):
            return "feature-extraction"
        else:
            return "text-generation"  # Default

    def analyze_data(self, df: pd.DataFrame, model_name: str,
                     analysis_type: str, custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Perform AI analysis on data"""
        if not self.is_available():
            raise Exception("AI analysis not available. Please install required libraries.")

        if model_name not in self.loaded_models:
            raise ValueError(f"Model {model_name} not loaded")

        try:
            # Prepare data summary for analysis
            data_summary = self._prepare_data_summary(df)

            # Get analysis prompt
            if custom_prompt:
                prompt = custom_prompt
            elif analysis_type in self.analysis_templates:
                prompt = self.analysis_templates[analysis_type]["prompt"]
            else:
                prompt = f"Analyze this dataset for {analysis_type}"

            # Combine prompt with data summary
            full_prompt = f"{prompt}\n\nDataset Summary:\n{data_summary}"

            # Perform analysis
            result = self._run_model_analysis(model_name, full_prompt)

            # Store analysis in history
            analysis_record = {
                "timestamp": datetime.now(),
                "model": model_name,
                "analysis_type": analysis_type,
                "prompt": full_prompt,
                "result": result,
                "data_shape": df.shape
            }
            self.analysis_history.append(analysis_record)

            return result

        except Exception as e:
            raise Exception(f"AI analysis failed: {str(e)}")

    def _prepare_data_summary(self, df: pd.DataFrame) -> str:
        """Prepare a comprehensive data summary for AI analysis"""
        try:
            summary = []

            # Basic info
            summary.append(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            summary.append(f"Memory Usage: {df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB")

            # Column information
            summary.append("\nColumn Information:")
            for col in df.columns:
                dtype = df[col].dtype
                null_count = df[col].isnull().sum()
                null_pct = (null_count / len(df)) * 100
                unique_count = df[col].nunique()

                summary.append(f"- {col}: {dtype}, {null_count} nulls ({null_pct:.1f}%), {unique_count} unique values")

            # Statistical summary for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary.append("\nNumeric Columns Statistics:")
                desc = df[numeric_cols].describe()
                summary.append(desc.to_string())

            # Sample data
            summary.append("\nSample Data (first 5 rows):")
            summary.append(df.head().to_string())

            # Data quality issues
            summary.append("\nData Quality Assessment:")
            total_nulls = df.isnull().sum().sum()
            duplicates = df.duplicated().sum()
            summary.append(f"- Total missing values: {total_nulls}")
            summary.append(f"- Duplicate rows: {duplicates}")

            # Correlations for numeric data
            if len(numeric_cols) > 1:
                summary.append("\nTop Correlations:")
                corr_matrix = df[numeric_cols].corr()
                # Get top correlations
                correlations = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.5:  # Only strong correlations
                            correlations.append((col1, col2, corr_val))

                correlations.sort(key=lambda x: abs(x[2]), reverse=True)
                for col1, col2, corr in correlations[:5]:  # Top 5
                    summary.append(f"- {col1} vs {col2}: {corr:.3f}")

            return "\n".join(summary)

        except Exception as e:
            return f"Error preparing data summary: {str(e)}"

    def _run_model_analysis(self, model_name: str, prompt: str) -> Dict[str, Any]:
        """Run analysis using the specified model"""
        try:
            model_config = self.model_configs[model_name]
            model_type = model_config["type"]

            # Use pipeline if available
            if model_name in self.pipelines:
                pipe = self.pipelines[model_name]

                if model_type == "text-generation":
                    # Text generation analysis
                    result = pipe(
                        prompt,
                        max_length=512,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=pipe.tokenizer.eos_token_id
                    )

                    generated_text = result[0]["generated_text"]
                    # Extract only the new generated part
                    if generated_text.startswith(prompt):
                        analysis_text = generated_text[len(prompt):].strip()
                    else:
                        analysis_text = generated_text

                    return {
                        "analysis_type": "text_generation",
                        "model": model_name,
                        "prompt": prompt,
                        "analysis": analysis_text,
                        "confidence": "N/A",
                        "timestamp": datetime.now().isoformat()
                    }

                elif model_type == "text-classification":
                    # Text classification analysis
                    result = pipe(prompt)

                    return {
                        "analysis_type": "text_classification",
                        "model": model_name,
                        "prompt": prompt,
                        "classification": result,
                        "confidence": result[0]["score"] if result else 0,
                        "timestamp": datetime.now().isoformat()
                    }

            else:
                # Manual analysis using model and tokenizer
                model = self.loaded_models[model_name]
                tokenizer = self.tokenizers[model_name]

                # Tokenize input
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)

                # Generate analysis
                with torch.no_grad():
                    outputs = model(**inputs)

                # Extract meaningful information from outputs
                if hasattr(outputs, 'last_hidden_state'):
                    # For models that output hidden states
                    hidden_states = outputs.last_hidden_state
                    # Use mean pooling
                    sentence_embedding = torch.mean(hidden_states, dim=1)

                    return {
                        "analysis_type": "feature_extraction",
                        "model": model_name,
                        "prompt": prompt,
                        "embeddings": sentence_embedding.tolist(),
                        "embedding_dimension": sentence_embedding.shape[-1],
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "analysis_type": "custom",
                        "model": model_name,
                        "prompt": prompt,
                        "analysis": "Model analysis completed but specific interpretation not available for this model type.",
                        "timestamp": datetime.now().isoformat()
                    }

        except Exception as e:
            return {
                "analysis_type": "error",
                "model": model_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def get_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about loaded models"""
        return self.model_configs.copy()

    def unload_model(self, model_name: str) -> bool:
        """Unload a specific model to free memory"""
        try:
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
            if model_name in self.tokenizers:
                del self.tokenizers[model_name]
            if model_name in self.pipelines:
                del self.pipelines[model_name]
            if model_name in self.model_configs:
                del self.model_configs[model_name]

            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return True
        except Exception as e:
            print(f"Error unloading model {model_name}: {e}")
            return False

    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get analysis history"""
        return self.analysis_history.copy()

    def clear_analysis_history(self):
        """Clear analysis history"""
        self.analysis_history.clear()

    def export_analysis_results(self, filepath: str, format_type: str = "json") -> bool:
        """Export analysis results to file"""
        try:
            if format_type.lower() == "json":
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(self.analysis_history, f, indent=2, default=str)
            elif format_type.lower() == "csv":
                # Convert to DataFrame for CSV export
                df_data = []
                for analysis in self.analysis_history:
                    row = {
                        "timestamp": analysis["timestamp"],
                        "model": analysis["model"],
                        "analysis_type": analysis["analysis_type"],
                        "data_shape": str(analysis.get("data_shape", "")),
                        "result": str(analysis["result"])
                    }
                    df_data.append(row)

                df = pd.DataFrame(df_data)
                df.to_csv(filepath, index=False)
            else:
                raise ValueError(f"Unsupported format: {format_type}")

            return True
        except Exception as e:
            print(f"Error exporting analysis results: {e}")
            return False

    def get_model_recommendations(self, analysis_type: str) -> List[str]:
        """Get recommended models for specific analysis type"""
        recommendations = {
            "Data Insights": [
                "microsoft/DialoGPT-large",
                "gpt2",
                "facebook/bart-large"
            ],
            "Pattern Detection": [
                "microsoft/DialoGPT-large",
                "facebook/bart-large",
                "google/flan-t5-base"
            ],
            "Sentiment Analysis": [
                "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "nlptown/bert-base-multilingual-uncased-sentiment",
                "distilbert-base-uncased-finetuned-sst-2-english"
            ],
            "Financial Analysis": [
                "ProsusAI/finbert",
                "yiyanghkust/finbert-tone",
                "ElKulako/cryptobert"
            ],
            "Text Classification": [
                "distilbert-base-uncased",
                "roberta-base",
                "microsoft/DialoGPT-medium"
            ]
        }

        return recommendations.get(analysis_type, ["gpt2", "distilbert-base-uncased"])

    def cleanup(self):
        """Cleanup all loaded models and free memory"""
        try:
            model_names = list(self.loaded_models.keys())
            for model_name in model_names:
                self.unload_model(model_name)

            print("AI Analysis Engine cleaned up successfully")
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information"""
        try:
            memory_info = {
                "loaded_models": len(self.loaded_models),
                "total_models": len(self.model_configs)
            }

            if torch.cuda.is_available():
                memory_info["gpu_memory_allocated"] = torch.cuda.memory_allocated()
                memory_info["gpu_memory_cached"] = torch.cuda.memory_reserved()

            return memory_info
        except Exception as e:
            return {"error": str(e)}

    def benchmark_model(self, model_name: str, test_prompts: List[str]) -> Dict[str, Any]:
        """Benchmark model performance"""
        if model_name not in self.loaded_models:
            raise ValueError(f"Model {model_name} not loaded")

        try:
            import time

            results = []
            total_time = 0

            for prompt in test_prompts:
                start_time = time.time()
                result = self._run_model_analysis(model_name, prompt)
                end_time = time.time()

                processing_time = end_time - start_time
                total_time += processing_time

                results.append({
                    "prompt": prompt,
                    "processing_time": processing_time,
                    "result": result
                })

            return {
                "model": model_name,
                "total_prompts": len(test_prompts),
                "total_time": total_time,
                "average_time": total_time / len(test_prompts),
                "results": results,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {"error": str(e)}