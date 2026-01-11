"""
ML Models Bridge - Auto-detect filename
"""
import sys
import os
import glob

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("=" * 80)
print("ML MODELS BRIDGE - INITIALIZING")
print("=" * 80)

try:
    # Search for Project file
    project_files = glob.glob(os.path.join(current_dir, "Project*.py"))

    print(f"\n🔍 Searching in: {current_dir}")
    print(f"📁 Found files: {project_files}")

    if not project_files:
        raise FileNotFoundError("No 'Project*.py' file found")

    project_file = project_files[0]
    print(f"✅ Using: {os.path.basename(project_file)}")

    # Load the module
    import importlib.util

    spec = importlib.util.spec_from_file_location("project_3", project_file)
    project_3 = importlib.util.module_from_spec(spec)

    sys.modules['project_3'] = project_3
    sys.modules['__mp_main__'] = project_3

    spec.loader.exec_module(project_3)

    # Export classes
    SmartAllocationSystem = project_3.SmartAllocationSystem
    DataEngine = project_3.DataEngine
    InvestorProfiler = project_3.InvestorProfiler
    RegimeDetector = project_3.RegimeDetector
    ReturnForecaster = project_3.ReturnForecaster
    PortfolioOptimizer = project_3.PortfolioOptimizer
    Backtester = project_3.Backtester

    ML_AVAILABLE = True
    print("✅ ML Models loaded successfully")
    print("=" * 80)

except Exception as e:
    ML_AVAILABLE = False
    print(f"\n❌ ERROR: {e}")
    print("=" * 80)

    import traceback

    traceback.print_exc()


    # Dummy classes
    class SmartAllocationSystem:
        pass


    class DataEngine:
        pass


    class InvestorProfiler:
        pass


    class RegimeDetector:
        pass


    class ReturnForecaster:
        pass


    class PortfolioOptimizer:
        pass


    class Backtester:
        pass

__all__ = [
    'SmartAllocationSystem', 'DataEngine', 'InvestorProfiler',
    'RegimeDetector', 'ReturnForecaster', 'PortfolioOptimizer',
    'Backtester', 'ML_AVAILABLE'
]