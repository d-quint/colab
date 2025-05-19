from ultralytics import YOLO
import os
from rich.console import Console
from rich.panel import Panel
import sys
import yaml

console = Console()


def get_available_directories():
    """Get list of available training directories."""
    runs_dir = 'runs/detect'
    if not os.path.exists(runs_dir):
        return []
    return [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]


def get_checkpoints(directory):
    """Get available checkpoint files for a directory."""
    weights_dir = os.path.join('runs/detect', directory, 'weights')
    if not os.path.exists(weights_dir):
        return []
    return [f for f in os.listdir(weights_dir) if f.endswith('.pt')]

def get_training_info(directory):
    """Get training information from args.yaml if available."""
    args_file = os.path.join('runs/detect', directory, 'args.yaml')
    if os.path.exists(args_file):
        with open(args_file, 'r') as f:
            args = yaml.safe_load(f)
            return {
                'total_epochs': args.get('epochs', 0),
                'imgsz': args.get('imgsz', 640),
                'batch': args.get('batch', 8)
            }
    return None


def fine_tune_menu():
    """Interactive menu for fine-tuning configuration."""
    print("\n=== Fine-tuning Configuration ===")

    # First, select the base model
    print("\nSelect base model source:")
    print("1. Use pre-trained YOLO model")
    print("2. Use previously trained custom model")

    source_choice = get_numeric_input("Input Choice: ", min_val=1, max_val=2)

    if source_choice == 1:
        # Select YOLO model type
        print("\nSelect model type:")
        model_types = ['n', 's', 'm', 'l', 'x']
        for i, type_ in enumerate(model_types, 1):
            print(f"{i}. YOLOv8{type_}")

        choice = get_numeric_input("Input Choice: ", min_val=1, max_val=len(model_types))
        model_type = model_types[choice - 1]
        base_model = f"yolov8{model_type}.pt"
    else:
        # Select from existing trained models
        directories = get_available_directories()
        if not directories:
            console.print("[red]No existing training directories found![/red]")
            return None

        print("\nAvailable trained models:")
        for i, dir_ in enumerate(directories, 1):
            print(f"{i}. {dir_}")

        dir_choice = get_numeric_input("Select directory number: ", min_val=1, max_val=len(directories))
        selected_dir = directories[dir_choice - 1]

        checkpoints = get_checkpoints(selected_dir)
        if not checkpoints:
            console.print("[red]No checkpoint files found in selected directory![/red]")
            return None

        print("\nAvailable checkpoints:")
        for i, checkpoint in enumerate(checkpoints, 1):
            print(f"{i}. {checkpoint}")

        checkpoint_choice = get_numeric_input("Select checkpoint number: ", min_val=1, max_val=len(checkpoints))
        selected_checkpoint = checkpoints[checkpoint_choice - 1]
        base_model = os.path.join('runs/detect', selected_dir, 'weights', selected_checkpoint)

    # Get fine-tuning parameters
    model_name = input("\nEnter new model name: ")
    data_yaml = input("Enter path to fine-tuning dataset.yaml [dataset.yaml]: ") or "dataset.yaml"

    # Fine-tuning specific parameters
    batch = get_numeric_input("Enter batch size", 8, min_val=1)
    epochs = get_numeric_input("Enter number of epochs", 100, min_val=1)
    imgsz = get_numeric_input("Enter image size", 640, min_val=32)
    patience = get_numeric_input("Enter patience (0 to disable early stopping): ", 0, min_val=0)

    # Fine-tuning specific learning parameters
    learning_rate = float(input("Enter initial learning rate [0.001]: ") or "0.001")

    return {
        'base_model': base_model,
        'model_name': model_name,
        'data_yaml': data_yaml,
        'batch': batch,
        'epochs': epochs,
        'imgsz': imgsz,
        'patience': patience,
        'lr0': learning_rate
    }


def fine_tune_model(config):
    """Execute model fine-tuning with given configuration."""
    try:
        # Load the base model
        model = YOLO(config['base_model'])

        # Start fine-tuning
        results = model.train(
            data=config['data_yaml'],
            epochs=config['epochs'],
            imgsz=config['imgsz'],
            batch=config['batch'],
            name=config['model_name'],
            patience=config['patience'],
            lr0=config['lr0'],
            exist_ok=False
        )
        return results

    except RuntimeError as e:
        if "out of memory" in str(e):
            console.print("\n[red]CUDA out of memory error occurred. Try:[/red]")
            console.print("1. Reducing batch size (currently: {})".format(config.get('batch')))
            console.print("2. Reducing image size (currently: {})".format(config.get('imgsz', 'N/A')))
            console.print("3. Using a smaller model type")
        raise

def hyperparameter_tuning_menu():
    """Interactive menu for hyperparameter tuning configuration."""
    print("\n=== Hyperparameter Tuning Configuration ===")

    # Get list of available training directories
    directories = get_available_directories()
    if not directories:
        console.print("[red]No existing training directories found in runs/detect![/red]")
        return None

    print("\nAvailable training directories:")
    for i, dir_ in enumerate(directories, 1):
        last_epoch, total_epochs, is_completed = get_training_status(dir_)
        status = "Completed" if is_completed else f"In progress ({last_epoch}/{total_epochs} epochs)"
        print(f"{i}. {dir_} - {status}")

    dir_choice = get_numeric_input("Select directory number: ", min_val=1, max_val=len(directories))
    selected_dir = directories[dir_choice - 1]

    # Get available checkpoints
    checkpoints = get_checkpoints(selected_dir)
    if not checkpoints:
        console.print("[red]No checkpoint files found in selected directory![/red]")
        return None

    print("\nAvailable checkpoints:")
    for i, checkpoint in enumerate(checkpoints, 1):
        print(f"{i}. {checkpoint}")

    checkpoint_choice = get_numeric_input("Select checkpoint number: ", min_val=1, max_val=len(checkpoints))
    selected_checkpoint = checkpoints[checkpoint_choice - 1]
    base_model = os.path.join('runs/detect', selected_dir, 'weights', selected_checkpoint)

    # Basic configuration
    model_name = input("\nEnter new model name for tuning results: ")
    
    # Get data YAML path from original training if available
    args_file = os.path.join('runs/detect', selected_dir, 'args.yaml')
    data_yaml = "dataset.yaml"  # Default
    if os.path.exists(args_file):
        try:
            with open(args_file, 'r') as f:
                args = yaml.safe_load(f)
                if 'data' in args:
                    data_yaml = args['data']
                    print(f"Using data path from original training: {data_yaml}")
        except Exception as e:
            console.print(f"[yellow]Error reading args.yaml: {str(e)}[/yellow]")
    
    data_yaml = input(f"Enter path to dataset.yaml [{data_yaml}]: ") or data_yaml
    
    # Tuning specific parameters
    print("\nTuning Configuration:")
    epochs = get_numeric_input("Enter epochs for each iteration", 30, min_val=1)
    iterations = get_numeric_input("Enter number of tuning iterations", 300, min_val=10)
    
    # Optimizer selection
    print("\nSelect optimizer:")
    optimizers = ['SGD', 'Adam', 'AdamW']
    for i, opt in enumerate(optimizers, 1):
        print(f"{i}. {opt}")
    
    opt_choice = get_numeric_input("Input Choice: ", min_val=1, max_val=len(optimizers))
    optimizer = optimizers[opt_choice - 1]
    
    # Advanced options
    use_ray = input("\nUse Ray Tune for distributed tuning? (y/n) [n]: ").lower() == 'y'
    
    # Custom search space
    use_custom_space = input("\nUse custom hyperparameter search space? (y/n) [n]: ").lower() == 'y'
    
    custom_space = None
    if use_custom_space:
        custom_space = {}
        
        # Check if original hyperparameters are available
        orig_hp = {}
        args_file = os.path.join('runs/detect', selected_dir, 'args.yaml')
        if os.path.exists(args_file):
            try:
                with open(args_file, 'r') as f:
                    args = yaml.safe_load(f)
                    # Extract relevant hyperparameters
                    for param in ['lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs', 
                                'warmup_momentum', 'box', 'cls', 'dfl']:
                        if param in args:
                            orig_hp[param] = args[param]
                    
                    if orig_hp:
                        console.print("\n[bold blue]Original Training Hyperparameters:[/bold blue]")
                        for param, value in orig_hp.items():
                            console.print(f"[blue]{param}: {value}[/blue]")
                        
                        use_orig_as_base = input("\nUse original hyperparameters as starting point? (y/n) [y]: ").lower() != 'n'
                        if use_orig_as_base:
                            # Set ranges based on original values
                            if 'lr0' in orig_hp:
                                lr_val = orig_hp['lr0']
                                lr_min = lr_val * 0.5
                                lr_max = lr_val * 2.0
                                custom_space['lr0'] = (lr_min, lr_max)
                                print(f"Setting lr0 range: {lr_min} to {lr_max} (based on original: {lr_val})")
                            
                            if 'lrf' in orig_hp:
                                lrf_val = orig_hp['lrf']
                                lrf_min = max(0.01, lrf_val * 0.5)
                                lrf_max = min(1.0, lrf_val * 2.0)
                                custom_space['lrf'] = (lrf_min, lrf_max)
                                print(f"Setting lrf range: {lrf_min} to {lrf_max} (based on original: {lrf_val})")
                            
                            if 'momentum' in orig_hp:
                                mom_val = orig_hp['momentum']
                                mom_min = max(0.6, mom_val * 0.9)
                                mom_max = min(0.98, mom_val * 1.1)
                                custom_space['momentum'] = (mom_min, mom_max)
                                print(f"Setting momentum range: {mom_min} to {mom_max} (based on original: {mom_val})")
                            
                            if 'weight_decay' in orig_hp:
                                wd_val = orig_hp['weight_decay']
                                wd_min = wd_val * 0.5
                                wd_max = wd_val * 2.0
                                custom_space['weight_decay'] = (wd_min, wd_max)
                                print(f"Setting weight_decay range: {wd_min} to {wd_max} (based on original: {wd_val})")
                            
                            # Loss weights
                            if 'box' in orig_hp:
                                box_val = orig_hp['box']
                                box_min = box_val * 0.7
                                box_max = box_val * 1.3
                                custom_space['box'] = (box_min, box_max)
                                print(f"Setting box loss range: {box_min} to {box_max} (based on original: {box_val})")
                            
                            if 'cls' in orig_hp:
                                cls_val = orig_hp['cls']
                                cls_min = cls_val * 0.7
                                cls_max = cls_val * 1.3
                                custom_space['cls'] = (cls_min, cls_max)
                                print(f"Setting cls loss range: {cls_min} to {cls_max} (based on original: {cls_val})")
                            
                            if 'dfl' in orig_hp:
                                dfl_val = orig_hp['dfl']
                                dfl_min = dfl_val * 0.7
                                dfl_max = dfl_val * 1.3
                                custom_space['dfl'] = (dfl_min, dfl_max)
                                print(f"Setting dfl loss range: {dfl_min} to {dfl_max} (based on original: {dfl_val})")
            except Exception as e:
                console.print(f"[yellow]Error reading original hyperparameters: {str(e)}[/yellow]")
        
        print("\nEnter custom search space (leave empty to use default or previously set ranges):")
        
        print("\n1. Training Hyperparameters:")
        # Learning rate
        if 'lr0' not in custom_space:
            lr_min = input("Minimum learning rate (lr0) [0.001]: ") or "0.001"
            lr_max = input("Maximum learning rate (lr0) [0.01]: ") or "0.01"
            custom_space['lr0'] = (float(lr_min), float(lr_max))
        
        # Learning rate factor
        if 'lrf' not in custom_space:
            lrf_min = input("Minimum learning rate factor (lrf) [0.01]: ") or "0.01"
            lrf_max = input("Maximum learning rate factor (lrf) [0.1]: ") or "0.1"
            custom_space['lrf'] = (float(lrf_min), float(lrf_max))
        
        # Momentum
        if 'momentum' not in custom_space:
            momentum_min = input("Minimum momentum [0.6]: ") or "0.6"
            momentum_max = input("Maximum momentum [0.98]: ") or "0.98"
            custom_space['momentum'] = (float(momentum_min), float(momentum_max))
        
        # Weight decay
        if 'weight_decay' not in custom_space:
            weight_decay_min = input("Minimum weight decay [0.0001]: ") or "0.0001"
            weight_decay_max = input("Maximum weight decay [0.001]: ") or "0.001"
            custom_space['weight_decay'] = (float(weight_decay_min), float(weight_decay_max))
        
        # Batch size
        if 'batch' not in custom_space:
            batch_min = get_numeric_input("Minimum batch size", 8, min_val=1)
            batch_max = get_numeric_input("Maximum batch size", 64, min_val=batch_min)
            custom_space['batch'] = (batch_min, batch_max)
        
        print("\n2. Loss Function Hyperparameters:")
        # Box loss weight
        if 'box' not in custom_space and input("Tune box loss weight? (y/n) [y]: ").lower() != 'n':
            box_min = input("Minimum box loss weight [7.0]: ") or "7.0"
            box_max = input("Maximum box loss weight [20.0]: ") or "20.0"
            custom_space['box'] = (float(box_min), float(box_max))
        
        # Classification loss weight
        if 'cls' not in custom_space and input("Tune classification loss weight? (y/n) [y]: ").lower() != 'n':
            cls_min = input("Minimum cls loss weight [0.3]: ") or "0.3"
            cls_max = input("Maximum cls loss weight [4.0]: ") or "4.0"
            custom_space['cls'] = (float(cls_min), float(cls_max))
        
        # DFL loss weight
        if 'dfl' not in custom_space and input("Tune dfl loss weight? (y/n) [y]: ").lower() != 'n':
            dfl_min = input("Minimum dfl loss weight [0.5]: ") or "0.5"
            dfl_max = input("Maximum dfl loss weight [2.0]: ") or "2.0"
            custom_space['dfl'] = (float(dfl_min), float(dfl_max))
        
        print("\n3. Augmentation Hyperparameters:")
        tune_augmentation = input("Tune data augmentation parameters? (y/n) [y]: ").lower() != 'n'
        if tune_augmentation:
            # HSV augmentation
            if 'hsv_h' not in custom_space:
                hsv_h_min = input("Minimum HSV-Hue augmentation [0.0]: ") or "0.0"
                hsv_h_max = input("Maximum HSV-Hue augmentation [0.1]: ") or "0.1"
                custom_space['hsv_h'] = (float(hsv_h_min), float(hsv_h_max))
            
            if 'hsv_s' not in custom_space:
                hsv_s_min = input("Minimum HSV-Saturation augmentation [0.0]: ") or "0.0"
                hsv_s_max = input("Maximum HSV-Saturation augmentation [0.9]: ") or "0.9"
                custom_space['hsv_s'] = (float(hsv_s_min), float(hsv_s_max))
            
            if 'hsv_v' not in custom_space:
                hsv_v_min = input("Minimum HSV-Value augmentation [0.0]: ") or "0.0"
                hsv_v_max = input("Maximum HSV-Value augmentation [0.9]: ") or "0.9"
                custom_space['hsv_v'] = (float(hsv_v_min), float(hsv_v_max))
            
            # Geometric augmentations
            if 'degrees' not in custom_space:
                degrees_min = input("Minimum rotation degrees [0.0]: ") or "0.0"
                degrees_max = input("Maximum rotation degrees [10.0]: ") or "10.0"
                custom_space['degrees'] = (float(degrees_min), float(degrees_max))
            
            if 'translate' not in custom_space:
                translate_min = input("Minimum translate factor [0.0]: ") or "0.0"
                translate_max = input("Maximum translate factor [0.2]: ") or "0.2"
                custom_space['translate'] = (float(translate_min), float(translate_max))
            
            if 'scale' not in custom_space:
                scale_min = input("Minimum scale factor [0.0]: ") or "0.0"
                scale_max = input("Maximum scale factor [0.9]: ") or "0.9"
                custom_space['scale'] = (float(scale_min), float(scale_max))
            
            # Mosaic and mixup
            if 'mosaic' not in custom_space:
                mosaic_min = input("Minimum mosaic probability [0.0]: ") or "0.0"
                mosaic_max = input("Maximum mosaic probability [1.0]: ") or "1.0"
                custom_space['mosaic'] = (float(mosaic_min), float(mosaic_max))
            
            if 'mixup' not in custom_space:
                mixup_min = input("Minimum mixup probability [0.0]: ") or "0.0"
                mixup_max = input("Maximum mixup probability [1.0]: ") or "1.0"
                custom_space['mixup'] = (float(mixup_min), float(mixup_max))
    
    # Visualization options
    print("\nVisualization Options:")
    plot_results = input("Generate plots of tuning results? (y/n) [y]: ").lower() != 'n'
    save_results = input("Save best model weights? (y/n) [y]: ").lower() != 'n'
    val_during_tune = input("Run validation during tuning? (y/n) [n]: ").lower() == 'y'
    
    return {
        'model': base_model,
        'model_name': model_name,
        'data_yaml': data_yaml,
        'epochs': epochs,
        'iterations': iterations,
        'optimizer': optimizer,
        'use_ray': use_ray,
        'custom_space': custom_space,
        'plot_results': plot_results,
        'save_results': save_results,
        'val_during_tune': val_during_tune
    }

def display_tuning_results(results_path):
    """Display the results of hyperparameter tuning."""
    if not os.path.exists(results_path):
        console.print(f"[red]Results directory {results_path} not found.[/red]")
        return
    
    best_hp_file = os.path.join(results_path, 'best_hyperparameters.yaml')
    if os.path.exists(best_hp_file):
        with open(best_hp_file, 'r') as f:
            console.print(Panel.fit("\n[bold green]Best Hyperparameters[/bold green]", style="green"))
            console.print(f.read())
    
    results_csv = os.path.join(results_path, 'tune_results.csv')
    if os.path.exists(results_csv):
        import pandas as pd
        try:
            df = pd.read_csv(results_csv)
            console.print(Panel.fit("\n[bold green]Tuning Summary[/bold green]", style="green"))
            console.print(f"Total iterations: {len(df)}")
            console.print(f"Best fitness: {df['fitness'].max():.4f}")
            console.print(f"Average fitness: {df['fitness'].mean():.4f}")
        except Exception as e:
            console.print(f"[yellow]Error reading results CSV: {str(e)}[/yellow]")
    
    console.print(Panel.fit("\n[bold blue]Results Files[/bold blue]", style="blue"))
    console.print(f"[blue]- best_hyperparameters.yaml: Best performing hyperparameters[/blue]")
    console.print(f"[blue]- best_fitness.png: Plot of fitness vs. iterations[/blue]")
    console.print(f"[blue]- tune_results.csv: Detailed results of each iteration[/blue]")
    console.print(f"[blue]- tune_scatter_plots.png: Scatter plots of hyperparameters[/blue]")
    console.print(f"[blue]- weights/: Best model weights[/blue]")

def tune_model(config):
    """Execute hyperparameter tuning with given configuration."""
    try:
        # Extract trained model directory if applicable
        model_path = config['model']
        model_dir = None
        if os.path.exists(model_path) and 'runs/detect' in model_path:
            # This is a custom trained model
            parts = model_path.split(os.sep)
            try:
                # Find the directory name in runs/detect/DIR/weights/model.pt
                detect_idx = parts.index('detect')
                if detect_idx + 1 < len(parts):
                    model_dir = parts[detect_idx + 1]
                    console.print(f"[blue]Original training directory: {model_dir}[/blue]")
            except ValueError:
                pass
                
        # Check for original hyperparameters if available
        orig_hp = {}
        if model_dir:
            args_file = os.path.join('runs/detect', model_dir, 'args.yaml')
            if os.path.exists(args_file):
                try:
                    with open(args_file, 'r') as f:
                        args = yaml.safe_load(f)
                        console.print("[blue]Loaded original training parameters as reference[/blue]")
                        # Extract relevant hyperparameters
                        for param in ['lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs', 
                                    'warmup_momentum', 'box', 'cls', 'dfl']:
                            if param in args:
                                orig_hp[param] = args[param]
                except Exception as e:
                    console.print(f"[yellow]Error reading original hyperparameters: {str(e)}[/yellow]")

        # Load the model
        console.print(f"[blue]Loading trained model from: {model_path}[/blue]")
        model = YOLO(model_path)
        
        # Prepare tuning kwargs
        tune_kwargs = {
            'data': config['data_yaml'],
            'epochs': config['epochs'],
            'iterations': config['iterations'],
            'optimizer': config['optimizer'],
            'plots': config['plot_results'],
            'save': config.get('save_results', True),
            'val': config.get('val_during_tune', False),
            'use_ray': config['use_ray'],
            'name': config['model_name'] if config['model_name'] else None
        }
        
        # Add custom search space if provided
        if config['custom_space']:
            # If we have original hyperparameters, print them for reference
            if orig_hp:
                console.print("\n[bold blue]Original Hyperparameters:[/bold blue]")
                for param, value in orig_hp.items():
                    if param in config['custom_space']:
                        min_val, max_val = config['custom_space'][param]
                        console.print(f"[blue]{param}: {value} (Search range: {min_val} to {max_val})[/blue]")
                    else:
                        console.print(f"[blue]{param}: {value} (Not being tuned)[/blue]")
            
            tune_kwargs['space'] = config['custom_space']
        
        # Add resume path if provided
        if 'resume' in config and config['resume']:
            tune_kwargs['resume'] = config['resume']
            console.print(f"[bold blue]Resuming tuning from {config['resume']}[/bold blue]")
        
        # Start hyperparameter tuning
        console.print("\n[bold green]Starting hyperparameter tuning on trained model...[/bold green]")
        console.print("[yellow]This may take a long time depending on your configuration.[/yellow]")
        console.print(f"[blue]Running {config['iterations']} iterations with {config['epochs']} epochs each.[/blue]")
        
        results = model.tune(**tune_kwargs)
        
        # Print information about best results
        console.print("\n[bold green]Hyperparameter tuning complete![/bold green]")
        results_path = os.path.join('runs/detect', config['model_name'] if config['model_name'] else 'tune')
        console.print(f"[green]Results saved to {results_path}[/green]")
        
        # Display detailed results
        display_tuning_results(results_path)
        
        return results
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            console.print("\n[red]CUDA out of memory error occurred. Try:[/red]")
            console.print("1. Reducing epochs (currently: {})".format(config.get('epochs')))
            console.print("2. Using a smaller model type")
        raise

def get_tuning_resume_options():
    """Get available tune results for resuming tuning."""
    runs_dir = 'runs/detect'
    if not os.path.exists(runs_dir):
        return []
    
    # Find directories that contain hyperparameter tuning results
    tune_dirs = []
    for d in os.listdir(runs_dir):
        tune_result_path = os.path.join(runs_dir, d, 'tune_results.csv')
        if os.path.exists(tune_result_path):
            tune_dirs.append(d)
    
    return tune_dirs

def new_training_menu():
    """Interactive menu for new training configuration."""
    print("\n=== New Training Configuration ===")

    # Model type selection
    print("\nSelect model type:")
    model_types = ['n', 's', 'm', 'l', 'x']
    for i, type_ in enumerate(model_types, 1):
        print(f"{i}. YOLOv8{type_}")

    while True:
        try:
            choice = get_numeric_input("Input Choice: ", min_val=1, max_val=len(model_types))
            model_type = model_types[choice - 1]
            break
        except (ValueError, IndexError):
            print("Invalid choice. Please try again.")

    model_name = input("\nEnter model name (e.g., yolov8_s_trash) [yolov8_s_trash]: ") or "yolov8_s_trash"
    data_yaml = input("Enter path to dataset.yaml [dataset.yaml]: ") or "dataset.yaml"

    # Get training parameters
    batch = get_numeric_input("Enter batch size", 8, min_val=1)
    epochs = get_numeric_input("Enter number of epochs", 100, min_val=1)
    imgsz = get_numeric_input("Enter image size", 640, min_val=32)
    patience = get_numeric_input("Enter patience (0 to disable early stopping): ", 0, min_val=0)

    return {
        'model_name': model_name,
        'model_type': model_type,
        'data_yaml': data_yaml,
        'batch': batch,
        'epochs': epochs,
        'imgsz': imgsz,
        'patience': patience
    }


def get_training_status(directory):
    """
    Get comprehensive training status including whether training was completed.
    Returns tuple of (last_epoch, total_epochs, is_completed)
    """
    args_file = os.path.join('runs/detect', directory, 'args.yaml')
    results_file = os.path.join('runs/detect', directory, 'results.csv')

    # Get planned total epochs from args.yaml
    total_epochs = 0
    if os.path.exists(args_file):
        with open(args_file, 'r') as f:
            args = yaml.safe_load(f)
            total_epochs = args.get('epochs', 0)

    # Check if training was completed by looking at results.csv
    is_completed = False
    last_epoch = 0
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:  # If there's data beyond the header
                    last_epoch = len(lines) - 1  # Subtract 1 for header row
                    # If last_epoch matches or exceeds total_epochs, training was completed
                    is_completed = last_epoch >= total_epochs
        except:
            pass

    return last_epoch, total_epochs, is_completed


def get_last_epoch(directory):
    """Get the last completed epoch from training."""
    last_weights = os.path.join('runs/detect', directory, 'weights', 'last.pt')
    if os.path.exists(last_weights):
        model = YOLO(last_weights)
        # Try to get the last epoch from the model's training history
        try:
            return model.ckpt.get('epoch', 0)
        except:
            return 0
    return 0


def get_numeric_input(prompt, default=None, min_val=None, max_val=None):
    """Get numeric input with validation."""
    while True:
        value = input(f"{prompt} [{default}]: ") if default else input(prompt)
        if not value and default is not None:
            return default
        try:
            value = int(value)
            if min_val is not None and value < min_val:
                print(f"Value must be at least {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"Value must be at most {max_val}")
                continue
            return value
        except ValueError:
            print("Please enter a valid number")


def continue_training_menu():
    """Interactive menu for continuing training."""
    print("\n=== Continue Training Configuration ===")

    directories = get_available_directories()
    if not directories:
        console.print("[red]No existing training directories found in runs/detect![/red]")
        return None

    print("\nAvailable training directories:")
    for i, dir_ in enumerate(directories, 1):
        last_epoch, total_epochs, is_completed = get_training_status(dir_)
        status = "Completed" if is_completed else f"In progress ({last_epoch}/{total_epochs} epochs)"
        print(f"{i}. {dir_} - {status}")

    while True:
        try:
            choice = get_numeric_input("Select directory number: ", min_val=1, max_val=len(directories))
            selected_dir = directories[choice - 1]
            break
        except (ValueError, IndexError):
            print("Invalid choice. Please try again.")

    last_epoch, total_epochs, is_completed = get_training_status(selected_dir)
    training_info = get_training_info(selected_dir)

    print(f"\nTraining Status for {selected_dir}:")
    print(f"Completed epochs: {last_epoch}")
    print(f"Total planned epochs: {total_epochs}")

    if is_completed:
        print("Status: Training completed")
        print("\nSelect continuation mode:")
        print("1. Add additional epochs to completed training")

        mode_choice = "1"
        epochs = get_numeric_input("\nEnter number of additional epochs", 100, min_val=1)
        total_epochs = last_epoch + epochs
    else:
        remaining_epochs = total_epochs - last_epoch
        print(f"Remaining epochs: {remaining_epochs}")
        print("Status: Training incomplete")

        print("\nSelect continuation mode:")
        print("1. Complete remaining epochs")
        print("2. Add new epochs to training")

        mode_choice = input("\nInput Choice: ")

        if mode_choice == "1" and remaining_epochs > 0:
            epochs = remaining_epochs
            total_epochs = total_epochs
            print(f"\nWill continue training for remaining {remaining_epochs} epochs")
        else:
            epochs = get_numeric_input("\nEnter number of additional epochs", 100, min_val=1)
            total_epochs = last_epoch + epochs

    checkpoints = get_checkpoints(selected_dir)
    if not checkpoints:
        console.print("[red]No checkpoint files found in selected directory![/red]")
        return None

    print("\nAvailable checkpoints:")
    for i, checkpoint in enumerate(checkpoints, 1):
        print(f"{i}. {checkpoint}")

    while True:
        try:
            choice = get_numeric_input("Select checkpoint number: ", min_val=1, max_val=len(checkpoints))
            selected_checkpoint = checkpoints[choice - 1]
            break
        except (ValueError, IndexError):
            print("Invalid choice. Please try again.")

    batch = training_info['batch'] if training_info else get_numeric_input("Enter batch size", 8, min_val=1)

    return {
        'directory': selected_dir,
        'checkpoint': selected_checkpoint,
        'batch': batch,
        'epochs': total_epochs,
        'resume_epoch': last_epoch,
        'is_completed': is_completed
    }


def train_model(config, resume=False):
    """Execute model training with given configuration."""
    try:
        if resume:
            checkpoint_path = os.path.join('runs/detect', config['directory'], 'weights', config['checkpoint'])
            model = YOLO(checkpoint_path)

            results = model.train(
                epochs=config['epochs'],
                batch=config['batch'],
                resume=True,
                exist_ok=True
            )
        else:
            model_path = f"yolov8{config['model_type']}.pt"
            model = YOLO(model_path)
            results = model.train(
                data=config['data_yaml'],
                epochs=config['epochs'],
                imgsz=config['imgsz'],
                batch=config['batch'],
                  name=config['model_name'],
                patience=config['patience'],
                exist_ok=False
            )
        return results

    except RuntimeError as e:
        if "out of memory" in str(e):
            console.print("\n[red]CUDA out of memory error occurred. Try:[/red]")
            console.print("1. Reducing batch size (currently: {})".format(config.get('batch')))
            console.print("2. Reducing image size (currently: {})".format(config.get('imgsz', 'N/A')))
            console.print("3. Using a smaller model type")
        raise

def template_hyperparameter_tuning():
    """Setup hyperparameter tuning with a predefined template configuration."""
    print("\n=== Template Hyperparameter Tuning Configuration ===")

    # Get list of available training directories
    directories = get_available_directories()
    if not directories:
        console.print("[red]No existing training directories found in runs/detect![/red]")
        return None

    print("\nAvailable training directories:")
    for i, dir_ in enumerate(directories, 1):
        last_epoch, total_epochs, is_completed = get_training_status(dir_)
        status = "Completed" if is_completed else f"In progress ({last_epoch}/{total_epochs} epochs)"
        print(f"{i}. {dir_} - {status}")

    dir_choice = get_numeric_input("Select directory number: ", min_val=1, max_val=len(directories))
    selected_dir = directories[dir_choice - 1]

    # Get available checkpoints
    checkpoints = get_checkpoints(selected_dir)
    if not checkpoints:
        console.print("[red]No checkpoint files found in selected directory![/red]")
        return None

    print("\nAvailable checkpoints:")
    for i, checkpoint in enumerate(checkpoints, 1):
        print(f"{i}. {checkpoint}")

    checkpoint_choice = get_numeric_input("Select checkpoint number: ", min_val=1, max_val=len(checkpoints))
    selected_checkpoint = checkpoints[checkpoint_choice - 1]
    base_model = os.path.join('runs/detect', selected_dir, 'weights', selected_checkpoint)

    # Basic configuration
    model_name = input("\nEnter new model name for tuning results: ")
    
    # Get data YAML path from original training if available
    args_file = os.path.join('runs/detect', selected_dir, 'args.yaml')
    data_yaml = "dataset.yaml"  # Default
    if os.path.exists(args_file):
        try:
            with open(args_file, 'r') as f:
                args = yaml.safe_load(f)
                if 'data' in args:
                    data_yaml = args['data']
                    print(f"Using data path from original training: {data_yaml}")
        except Exception as e:
            console.print(f"[yellow]Error reading args.yaml: {str(e)}[/yellow]")
    
    data_yaml = input(f"Enter path to dataset.yaml [{data_yaml}]: ") or data_yaml
    
    # Pre-configured search space from template
    custom_space = {
        "lr0": (1e-5, 1e-2),      # Learning rate
        "scale": (0.0, 0.9),       # Scaling augmentation
        "copy_paste": (0.0, 1.0),  # Copy-paste augmentation
        "box": (0.02, 0.2),        # Bounding box loss weight
        "mosaic": (0.0, 1.0)       # Mosaic augmentation
    }
    
    # Display the template search space
    console.print("\n[bold blue]Template Search Space:[/bold blue]")
    for param, (min_val, max_val) in custom_space.items():
        console.print(f"[blue]{param}: {min_val} to {max_val}[/blue]")
    
    # Allow customization of iterations and epochs
    print("\nTuning Configuration:")
    epochs = get_numeric_input("Enter epochs for each iteration", 30, min_val=1)
    iterations = get_numeric_input("Enter number of tuning iterations", 300, min_val=10)
    
    # Optimizer selection
    print("\nSelect optimizer:")
    optimizers = ['SGD', 'Adam', 'AdamW']
    for i, opt in enumerate(optimizers, 1):
        print(f"{i}. {opt}")
    
    opt_choice = get_numeric_input("Input Choice: ", min_val=1, max_val=len(optimizers))
    optimizer = optimizers[opt_choice - 1]
    
    # Advanced options
    use_ray = input("\nUse Ray Tune for distributed tuning? (y/n) [n]: ").lower() == 'y'
    
    # Visualization options
    print("\nVisualization Options:")
    plot_results = input("Generate plots of tuning results? (y/n) [y]: ").lower() != 'n'
    save_results = input("Save best model weights? (y/n) [y]: ").lower() != 'n'
    val_during_tune = input("Run validation during tuning? (y/n) [n]: ").lower() == 'y'
    
    return {
        'model': base_model,
        'model_name': model_name,
        'data_yaml': data_yaml,
        'epochs': epochs,
        'iterations': iterations,
        'optimizer': optimizer,
        'use_ray': use_ray,
        'custom_space': custom_space,
        'plot_results': plot_results,
        'save_results': save_results,
        'val_during_tune': val_during_tune
    }

def main():
    """Main CLI interface."""
    while True:
        console.print(Panel.fit("YOLO Training Interface", style="bold blue"))
        print("\nSelect Training Mode:")
        print("1. New Training")
        print("2. Continue Training")
        print("3. Fine-tune Model")
        print("4. Hyperparameter Tuning")
        print("5. Template Hyperparameter Tuning")
        print("6. Exit")

        choice = input("\nInput Choice: ")

        if choice == "1":
            config = new_training_menu()
            if config:
                print("\nTraining Configuration Summary:")
                for key, value in config.items():
                    print(f"{key}: {value}")

                if input("\nProceed with training? (y/n): ").lower() == 'y':
                    results = train_model(config, resume=False)
                    console.print("[green]Training complete![/green]")

        elif choice == "2":
            config = continue_training_menu()
            if config:
                print("\nTraining Configuration Summary:")
                for key, value in config.items():
                    if key not in ['resume_epoch', 'is_completed']:
                        print(f"{key}: {value}")

                if input("\nProceed with training? (y/n): ").lower() == 'y':
                    results = train_model(config, resume=True)
                    console.print("[green]Training complete![/green]")

        elif choice == "3":
            config = fine_tune_menu()
            if config:
                print("\nFine-tuning Configuration Summary:")
                for key, value in config.items():
                    print(f"{key}: {value}")

                if input("\nProceed with fine-tuning? (y/n): ").lower() == 'y':
                    results = fine_tune_model(config)
                    console.print("[green]Fine-tuning complete![/green]")
                    
        elif choice == "4":
            config = hyperparameter_tuning_menu()
            if config:
                print("\nHyperparameter Tuning Configuration Summary:")
                for key, value in config.items():
                    if key == 'custom_space' and value:
                        print("custom_space:")
                        for param, range_val in value.items():
                            print(f"  {param}: {range_val}")
                    else:
                        print(f"{key}: {value}")
                
                if input("\nProceed with hyperparameter tuning? (y/n): ").lower() == 'y':
                    # Check for resume option
                    tune_dirs = get_tuning_resume_options()
                    if tune_dirs and input("\nResume previous tuning session? (y/n): ").lower() == 'y':
                        print("\nAvailable tuning sessions:")
                        for i, dir_ in enumerate(tune_dirs, 1):
                            print(f"{i}. {dir_}")
                        
                        dir_choice = get_numeric_input("Select directory number to resume: ", min_val=1, max_val=len(tune_dirs))
                        selected_dir = tune_dirs[dir_choice - 1]
                        
                        # Update the config to include resume path
                        config['resume'] = os.path.join('runs/detect', selected_dir)
                    
                    results = tune_model(config)
                    console.print("[green]Hyperparameter tuning complete![/green]")
                    
        elif choice == "5":
            config = template_hyperparameter_tuning()
            if config:
                print("\nTemplate Hyperparameter Tuning Configuration Summary:")
                for key, value in config.items():
                    if key == 'custom_space' and value:
                        print("custom_space:")
                        for param, range_val in value.items():
                            print(f"  {param}: {range_val}")
                    else:
                        print(f"{key}: {value}")
                
                if input("\nProceed with template hyperparameter tuning? (y/n): ").lower() == 'y':
                    # Check for resume option
                    tune_dirs = get_tuning_resume_options()
                    if tune_dirs and input("\nResume previous tuning session? (y/n): ").lower() == 'y':
                        print("\nAvailable tuning sessions:")
                        for i, dir_ in enumerate(tune_dirs, 1):
                            print(f"{i}. {dir_}")
                        
                        dir_choice = get_numeric_input("Select directory number to resume: ", min_val=1, max_val=len(tune_dirs))
                        selected_dir = tune_dirs[dir_choice - 1]
                        
                        # Update the config to include resume path
                        config['resume'] = os.path.join('runs/detect', selected_dir)
                    
                    results = tune_model(config)
                    console.print("[green]Template hyperparameter tuning complete![/green]")

        elif choice == "6":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error occurred: {str(e)}[/red]")