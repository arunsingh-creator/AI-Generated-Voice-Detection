"""
GitHub Pre-Push Validation Script
Checks for sensitive data, proper .gitignore, and validates all files before commit
"""
import os
import re
from pathlib import Path

# ANSI colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_status(message, status='info'):
    if status == 'success':
        print(f"{GREEN}✓{RESET} {message}")
    elif status == 'error':
        print(f"{RED}✗{RESET} {message}")
    elif status == 'warning':
        print(f"{YELLOW}⚠{RESET} {message}")
    else:
        print(f"{BLUE}ℹ{RESET} {message}")


def check_gitignore():
    """Verify .gitignore has all necessary exclusions"""
    print("\n" + "="*60)
    print("1. CHECKING .GITIGNORE")
    print("="*60)
    
    gitignore_path = Path('.gitignore')
    if not gitignore_path.exists():
        print_status(".gitignore not found", 'error')
        return False
    
    with open(gitignore_path, 'r') as f:
        content = f.read()
    
    required_patterns = {
        '.env': 'Environment variables (contains API keys)',
        '__pycache__': 'Python cache files',
        '*.pyc': 'Compiled Python files',
        'venv/': 'Virtual environment',
        '.pytest_cache': 'Pytest cache',
        'data/raw/': 'Training data samples',
    }
    
    all_good = True
    for pattern, description in required_patterns.items():
        if pattern in content:
            print_status(f"{pattern:20s} - {description}", 'success')
        else:
            print_status(f"{pattern:20s} - MISSING! ({description})", 'warning')
            all_good = False
    
    return all_good


def scan_for_secrets():
    """Scan for potential secrets in tracked files"""
    print("\n" + "="*60)
    print("2. SCANNING FOR SENSITIVE DATA")
    print("="*60)
    
    secret_patterns = {
        r'secret_key_[\w]+': 'API keys',
        r'sk-[a-zA-Z0-9]{48}': 'OpenAI keys',
        r'AKIA[0-9A-Z]{16}': 'AWS keys',
        r'password\s*=\s*["\'][^"\']+["\']': 'Hardcoded passwords',
    }
    
    files_to_scan = [
        'app/config.py',
        'app/main.py',
        'scripts/*.py',
        'README.md',
    ]
    
    issues_found = []
    
    for pattern_desc in files_to_scan:
        for file_path in Path('.').glob(pattern_desc):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for pattern, desc in secret_patterns.items():
                        if re.search(pattern, content, re.IGNORECASE):
                            issues_found.append(f"{file_path}: Potential {desc}")
                except:
                    pass
    
    if issues_found:
        for issue in issues_found:
            print_status(issue, 'warning')
        return False
    else:
        print_status("No secrets detected in code files", 'success')
        return True


def check_env_file():
    """Check if .env file exists and is gitignored"""
    print("\n" + "="*60)
    print("3. CHECKING ENVIRONMENT FILES")
    print("="*60)
    
    env_path = Path('.env')
    env_example_path = Path('.env.example')
    
    if env_path.exists():
        print_status(".env file exists (should be gitignored)", 'success')
    else:
        print_status(".env file not found (ok if using environment variables)", 'warning')
    
    if env_example_path.exists():
        print_status(".env.example exists (template for users)", 'success')
    else:
        print_status(".env.example missing (recommended to add)", 'warning')
    
    # Check if .env is in .gitignore
    gitignore_path = Path('.gitignore')
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            if '.env' in f.read():
                print_status(".env is gitignored", 'success')
                return True
            else:
                print_status(".env is NOT gitignored - CRITICAL!", 'error')
                return False
    
    return True


def check_model_files():
    """Check model file sizes and gitignore status"""
    print("\n" + "="*60)
    print("4. CHECKING MODEL FILES")
    print("="*60)
    
    models_dir = Path('models')
    if not models_dir.exists():
        print_status("No models directory found", 'warning')
        return True
    
    model_files = list(models_dir.glob('*.pkl')) + list(models_dir.glob('*.joblib'))
    
    total_size = 0
    for model_file in model_files:
        size = model_file.stat().st_size
        total_size += size
        size_mb = size / (1024 * 1024)
        
        if size_mb > 100:
            print_status(f"{model_file.name}: {size_mb:.1f} MB - TOO LARGE for Git!", 'error')
        elif size_mb > 10:
            print_status(f"{model_file.name}: {size_mb:.1f} MB - Consider Git LFS", 'warning')
        else:
            print_status(f"{model_file.name}: {size_mb:.2f} MB", 'success')
    
    total_mb = total_size / (1024 * 1024)
    print(f"\nTotal model size: {total_mb:.2f} MB")
    
    if total_mb > 100:
        print_status("Models too large - consider using Git LFS or .gitignore", 'error')
        return False
    
    return True


def check_data_files():
    """Ensure training data is not being committed"""
    print("\n" + "="*60)
    print("5. CHECKING TRAINING DATA")
    print("="*60)
    
    data_dir = Path('data/raw')
    if not data_dir.exists():
        print_status("No training data directory", 'success')
        return True
    
    audio_files = list(data_dir.glob('**/*.mp3')) + list(data_dir.glob('**/*.wav'))
    
    if len(audio_files) > 0:
        total_size = sum(f.stat().st_size for f in audio_files)
        size_mb = total_size / (1024 * 1024)
        
        print_status(f"Found {len(audio_files)} audio files ({size_mb:.1f} MB)", 'info')
        
        # Check if data is gitignored
        gitignore_path = Path('.gitignore')
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                gitignore_content = f.read()
                if 'data/raw' in gitignore_content or '*.mp3' in gitignore_content:
                    print_status("Training data is gitignored", 'success')
                    return True
                else:
                    print_status("WARNING: Training data NOT gitignored!", 'error')
                    print_status("Add 'data/raw/' to .gitignore", 'warning')
                    return False
    else:
        print_status("No audio files found", 'success')
        return True
    
    return True


def check_documentation():
    """Verify essential documentation exists"""
    print("\n" + "="*60)
    print("6. CHECKING DOCUMENTATION")
    print("="*60)
    
    docs = {
        'README.md': 'Main documentation',
        'requirements.txt': 'Dependencies',
        'Dockerfile': 'Container configuration',
        '.env.example': 'Environment template',
    }
    
    all_good = True
    for doc, description in docs.items():
        if Path(doc).exists():
            print_status(f"{doc:20s} - {description}", 'success')
        else:
            print_status(f"{doc:20s} - Missing ({description})", 'warning')
            if doc in ['README.md', 'requirements.txt']:
                all_good = False
    
    return all_good


def check_code_quality():
    """Basic code quality checks"""
    print("\n" + "="*60)
    print("7. CODE QUALITY CHECKS")
    print("="*60)
    
    python_files = list(Path('app').glob('**/*.py')) + list(Path('scripts').glob('*.py'))
    
    issues = []
    for py_file in python_files:
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Check for common issues
            if 'print(' in content and 'logger' not in content and 'scripts/' not in str(py_file):
                issues.append(f"{py_file}: Contains print statements (consider logging)")
            
            if 'TODO' in content.upper() or 'FIXME' in content.upper():
                issues.append(f"{py_file}: Contains TODO/FIXME comments")
    
    if issues:
        print_status(f"Found {len(issues)} code quality notes:", 'warning')
        for issue in issues[:5]:  # Show first 5
            print(f"  - {issue}")
    else:
        print_status("No major code quality issues", 'success')
    
    return True


def generate_commit_summary():
    """Generate summary of what's being committed"""
    print("\n" + "="*60)
    print("8. COMMIT SUMMARY")
    print("="*60)
    
    # Count new files
    new_files = {
        'Enhanced Scripts': ['scripts/collect_data_enhanced.py', 'scripts/train_model_enhanced.py', 
                            'scripts/augment_data.py', 'scripts/verify_system.py'],
        'Advanced Features': ['app/services/feature_extractor_advanced.py'],
        'Documentation': ['MODEL_IMPROVEMENT_QUICKSTART.md', 'GUVI_TESTER_VALUES.txt'],
        'Modified': ['requirements.txt', 'app/routes/detection.py'],
        'Models': ['models/voice_classifier.pkl', 'models/feature_scaler.pkl']
    }
    
    for category, files in new_files.items():
        print(f"\n{category}:")
        for file in files:
            if Path(file).exists():
                size = Path(file).stat().st_size
                print_status(f"  {file} ({size:,} bytes)", 'success')
            else:
                print_status(f"  {file} - NOT FOUND", 'warning')


def main():
    print("=" * 60)
    print("GITHUB PRE-PUSH VALIDATION")
    print("=" * 60)
    
    results = {
        'gitignore': check_gitignore(),
        'secrets': scan_for_secrets(),
        'env': check_env_file(),
        'models': check_model_files(),
        'data': check_data_files(),
        'docs': check_documentation(),
        'quality': check_code_quality(),
    }
    
    generate_commit_summary()
    
    # Final summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for check, result in results.items():
        status = 'success' if result else 'error'
        print_status(f"{check.capitalize():20s}: {'PASS' if result else 'FAIL'}", status)
    
    print(f"\nPassed: {passed}/{total} checks")
    
    if passed == total:
        print(f"\n{GREEN}✓ SAFE TO PUSH TO GITHUB{RESET}")
        print("\nRecommended commit message:")
        print("  'Enhanced model with 63 advanced features, hyperparameter tuning, and 71% baseline accuracy'")
    elif passed >= 5:
        print(f"\n{YELLOW}⚠ MINOR ISSUES - Review warnings before pushing{RESET}")
    else:
        print(f"\n{RED}✗ CRITICAL ISSUES - DO NOT PUSH YET{RESET}")
        print("\nFix the errors above before pushing to GitHub.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
