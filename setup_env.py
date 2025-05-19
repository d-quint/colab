#!/usr/bin/env python3
"""
Setup Script for YOLO Training Environment
This script creates the necessary directory structure and base files for running the YOLO training interface.
"""

import os
import shutil
import yaml
import argparse
from rich.console import Console
from rich.panel import Panel

console = Console()

def setup_environment(base_dir=None):
    """Setup the required directory structure and base files."""
    if base_dir is None:
        # If in Google Colab, use its base directory, otherwise use current directory
        try:
            from google.colab import drive
            # Mount Google Drive if in Colab
            console.print("[bold yellow]Google Colab detected! Mounting Google Drive...[/bold yellow]")
            drive.mount('/content/drive')
            base_dir = '/content/drive/MyDrive/YOLO_Training'
            console.print(f"[green]Setting up environment in {base_dir}[/green]")
        except ImportError:
            base_dir = os.getcwd()
            console.print(f"[green]Setting up environment in current directory: {base_dir}[/green]")
    
    # Create main directory structure
    dirs_to_create = [
        os.path.join(base_dir, "runs", "detect"),
        os.path.join(base_dir, "datasets"),
    ]

    for directory in dirs_to_create:
        if not os.path.exists(directory):
            os.makedirs(directory)
            console.print(f"Created directory: [cyan]{directory}[/cyan]")
        else:
            console.print(f"Directory already exists: [cyan]{directory}[/cyan]")
    
    # Create a sample dataset.yaml file if it doesn't exist
    sample_dataset_yaml = os.path.join(base_dir, "dataset.yaml")
    if not os.path.exists(sample_dataset_yaml):
        dataset_config = {
            'path': './datasets/sample',  # dataset root dir
            'train': 'images/train',      # train images (relative to 'path')
            'val': 'images/val',          # val images (relative to 'path')
            'test': 'images/test',        # test images (optional)
            'names': {
                0: 'class0',
                1: 'class1',
                2: 'class2',
            }
        }
        
        with open(sample_dataset_yaml, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
            console.print(f"Created sample dataset config: [cyan]{sample_dataset_yaml}[/cyan]")
    
    # Create directories for the sample dataset
    sample_dataset_dirs = [
        os.path.join(base_dir, "datasets", "sample", "images", "train"),
        os.path.join(base_dir, "datasets", "sample", "images", "val"),
        os.path.join(base_dir, "datasets", "sample", "images", "test"),
        os.path.join(base_dir, "datasets", "sample", "labels", "train"),
        os.path.join(base_dir, "datasets", "sample", "labels", "val"),
        os.path.join(base_dir, "datasets", "sample", "labels", "test"),
    ]
    
    for directory in sample_dataset_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            console.print(f"Created sample dataset directory: [cyan]{directory}[/cyan]")
    
    # Create a sample run structure for demonstration
    sample_run_dir = os.path.join(base_dir, "runs", "detect", "yolov8n_sample")
    sample_weights_dir = os.path.join(sample_run_dir, "weights")
    
    if not os.path.exists(sample_weights_dir):
        os.makedirs(sample_weights_dir)
        console.print(f"Created sample weights directory: [cyan]{sample_weights_dir}[/cyan]")
    
    # Create sample args.yaml
    sample_args_yaml = os.path.join(sample_run_dir, "args.yaml")
    if not os.path.exists(sample_args_yaml):
        args_config = {
            'task': 'detect',
            'model': 'yolov8n.pt',
            'data': 'dataset.yaml',
            'epochs': 100,
            'patience': 50,
            'batch': 16,
            'imgsz': 640,
            'save': True,
            'exist_ok': False,
            'name': 'yolov8n_sample'
        }
        
        with open(sample_args_yaml, 'w') as f:
            yaml.dump(args_config, f, default_flow_style=False)
            console.print(f"Created sample args config: [cyan]{sample_args_yaml}[/cyan]")
    
    # Create sample results.csv
    sample_results_csv = os.path.join(sample_run_dir, "results.csv")
    if not os.path.exists(sample_results_csv):
        with open(sample_results_csv, 'w') as f:
            f.write("epoch,train/box_loss,train/cls_loss,train/dfl_loss,train/loss,metrics/precision,metrics/recall,metrics/mAP50,metrics/mAP50-95,val/box_loss,val/cls_loss,val/dfl_loss,val/loss\n")
            f.write("1,0.5123,0.3456,0.2345,1.0924,0.6543,0.7654,0.6523,0.5432,0.4321,0.3210,0.2123,0.9654\n")
            console.print(f"Created sample results file: [cyan]{sample_results_csv}[/cyan]")
    
    # Create sample weight file (empty file for demonstration)
    sample_weight_file = os.path.join(sample_weights_dir, "last.pt")
    if not os.path.exists(sample_weight_file):
        with open(sample_weight_file, 'w') as f:
            f.write("# This is a sample weight file for demonstration purposes\n")
            console.print(f"Created sample weight file: [cyan]{sample_weight_file}[/cyan]")
    
    # Copy the program.py and requirements.txt to the base directory if they exist
    # This is mainly useful for setting up in Google Colab
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    for file in ["program.py", "requirements.txt"]:
        src_file = os.path.join(script_dir, file)
        dst_file = os.path.join(base_dir, file)
        
        if os.path.exists(src_file) and not os.path.exists(dst_file):
            shutil.copy2(src_file, dst_file)
            console.print(f"Copied {file} to base directory: [cyan]{dst_file}[/cyan]")
    
    console.print(Panel.fit("[bold green]Environment setup complete![/bold green] The directory structure for YOLO training is now ready.", 
                           title="Setup Complete", 
                           border_style="green"))
    
    # Return the path to help with navigation in the notebook
    return base_dir

def download_pretrained_models(base_dir=None):
    """Download pretrained YOLOv8 models for use in training."""
    try:
        from ultralytics import YOLO
        
        if base_dir is None:
            try:
                from google.colab import drive
                base_dir = '/content/drive/MyDrive/YOLO_Training'
            except ImportError:
                base_dir = os.getcwd()
        
        models_dir = os.path.join(base_dir, "pretrained_models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        model_sizes = ['n', 's', 'm', 'l', 'x']
        
        console.print("[bold yellow]Downloading pretrained YOLOv8 models...[/bold yellow]")
        
        for size in model_sizes:
            model_name = f"yolov8{size}.pt"
            console.print(f"Downloading [cyan]{model_name}[/cyan]...")
            
            try:
                # Using a simplified approach to download models
                model = YOLO(model_name)
                
                # Try to save to the models directory
                model_path = os.path.join(models_dir, model_name)
                if not os.path.exists(model_path):
                    shutil.copy2(model.ckpt_path, model_path)
                
                console.print(f"[green]Successfully downloaded {model_name}[/green]")
            except Exception as e:
                console.print(f"[red]Failed to download {model_name}: {str(e)}[/red]")
        
        console.print(Panel.fit("[bold green]Model download complete![/bold green] YOLOv8 pretrained models are now available.", 
                               title="Download Complete", 
                               border_style="green"))
    except ImportError:
        console.print("[red]Error: Ultralytics package not installed. Run 'pip install ultralytics' first.[/red]")

def google_colab_example():
    """Create a sample notebook cell to run the program in Google Colab."""
    colab_example = """
# Example code for running in Google Colab
!pip install ultralytics rich pyyaml pandas matplotlib numpy

# Run the environment setup
%run setup_env.py

# Run the YOLO Training Interface
%run program.py
"""
    example_file = "colab_example.txt"
    with open(example_file, 'w') as f:
        f.write(colab_example)
    
    console.print(f"Created Google Colab example: [cyan]{example_file}[/cyan]")
    console.print("Copy the contents of this file into a Colab notebook cell to start using the program.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup the YOLO training environment")
    parser.add_argument("--base_dir", help="Base directory for setup", default=None)
    parser.add_argument("--download_models", action="store_true", help="Download pretrained YOLOv8 models")
    parser.add_argument("--colab_example", action="store_true", help="Create a sample Google Colab notebook cell")
    
    args = parser.parse_args()
    
    base_dir = setup_environment(args.base_dir)
    
    if args.download_models:
        download_pretrained_models(base_dir)
    
    if args.colab_example:
        google_colab_example()
