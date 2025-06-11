#!/usr/bin/env python3
"""
Simple fix for the tensor formatting error.
This patches your existing train_hybrid.py file with minimal changes.
"""

import re
from pathlib import Path

def apply_tensor_fix():
    """Apply a simple fix to convert tensors to scalars before logging."""
    
    print("🔧 Applying tensor format fix to train_hybrid.py...")
    
    # Find the training file
    train_file = Path("train_hybrid.py")
    if not train_file.exists():
        print("❌ train_hybrid.py not found")
        return False
    
    # Read current content
    content = train_file.read_text()
    
    # Add the helper function at the top after imports
    helper_function = '''
def convert_tensors_to_scalars(metrics_dict):
    """Convert tensor values to scalars for logging."""
    if not isinstance(metrics_dict, dict):
        return metrics_dict
    
    converted = {}
    for key, value in metrics_dict.items():
        if hasattr(value, 'item') and callable(getattr(value, 'item')):
            try:
                converted[key] = value.item()
            except:
                converted[key] = float(value) if hasattr(value, '__float__') else value
        else:
            converted[key] = value
    return converted

'''
    
    # Find where to insert the helper function (after the last import)
    import_pattern = r'(torch\.set_float32_matmul_precision\(\'high\'\))'
    if import_pattern in content:
        content = re.sub(import_pattern, r'\1\n' + helper_function, content)
        print("✅ Added helper function")
    else:
        print("⚠️  Could not find import section, adding at start")
        content = helper_function + content
    
    # Fix 1: Before logging train_metrics
    old_train_log = r'self\.logger\.log\(train_metrics, \'train\'\)'
    new_train_log = 'self.logger.log(convert_tensors_to_scalars(train_metrics), \'train\')'
    
    if old_train_log in content:
        content = re.sub(old_train_log, new_train_log, content)
        print("✅ Fixed train metrics logging")
    
    # Fix 2: Before logging eval_metrics
    old_eval_log = r'self\.logger\.log\(eval_metrics, \'eval\'\)'
    new_eval_log = 'self.logger.log(convert_tensors_to_scalars(eval_metrics), \'eval\')'
    
    if old_eval_log in content:
        content = re.sub(old_eval_log, new_eval_log, content)
        print("✅ Fixed eval metrics logging")
    
    # Write back the fixed content
    train_file.write_text(content)
    print(f"✅ Successfully patched {train_file.name}")
    
    return True

def main():
    """Apply the fix and provide instructions."""
    print("🚀 TENSOR FORMAT FIX FOR HYBRID TD-MPC2")
    print("=" * 50)
    
    print("Error details:")
    print("  The TD-MPC2 logger expects scalar values but receives tensors")
    print("  This happens when hybrid metrics contain tensor values")
    print()
    
    success = apply_tensor_fix()
    
    if success:
        print("\n🎉 Fix applied successfully!")
        print()
        print("Now restart your training:")
        print("  python train_hybrid.py task=dog-run hybrid_mpc=true")
        print()
        print("Expected output:")
        print("  Step 0: Classical horizon = 12, Transition progress = 0.000")
        print("  train   E: 1    I: 5000    R: X.X    [continues normally]")
        print()
        print("🔍 What to monitor:")
        print("  - Classical horizon should decrease 12→0 over 1M steps")
        print("  - Transition progress should increase 0.0→1.0") 
        print("  - Episode rewards should improve over time")
        print("  - No more tensor format errors!")
        
    else:
        print("\n💥 Automatic fix failed")
        print("Manual fix:")
        print("1. Open train_hybrid.py")
        print("2. Find: self.logger.log(train_metrics, 'train')")
        print("3. Replace with: self.logger.log(convert_tensors_to_scalars(train_metrics), 'train')")
        print("4. Do the same for eval_metrics")
        print("5. Add the convert_tensors_to_scalars function at the top")

if __name__ == "__main__":
    main()