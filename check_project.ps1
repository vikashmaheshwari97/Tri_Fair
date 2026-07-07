$ErrorActionPreference = "Stop"

Write-Host "Checking Python interpreter..."
python -c "import sys; print(sys.executable); print(sys.version)"
if ($LASTEXITCODE -ne 0) { throw "Interpreter check failed" }

Write-Host "`nCompiling project..."
python -m compileall -q src scripts analysis tests
if ($LASTEXITCODE -ne 0) { throw "Compilation failed" }

Write-Host "`nChecking active script imports..."
python -c "import importlib; modules=['scripts._common','scripts.experiment','scripts.evaluate_prompts','scripts.evaluate_initial_prompts','scripts.prepare_manifests','scripts.run_cost_ablation','scripts.run_hp_sens']; [importlib.import_module(m) for m in modules]; print('Script imports OK')"
if ($LASTEXITCODE -ne 0) { throw "Script import check failed" }

Write-Host "`nChecking source imports..."
python -c "from src.config.dataset_configs import ALL_DATASETS; from src.config.initial_prompts import INITIAL_PROMPTS; from src.helpers.task_creation import create_dev_tasks, create_test_task; from src.tasks.fairness_task import FairnessTask; from src.mo_capo import MoCAPO; from src.tri_fair import TriFair; from src.nsgaii_po_fair import NSGAiiPOFair; print('Source imports OK')"
if ($LASTEXITCODE -ne 0) { throw "Source import check failed" }

$modules = @(
    "scripts.experiment",
    "scripts.evaluate_prompts",
    "scripts.evaluate_initial_prompts",
    "scripts.prepare_manifests",
    "scripts.run_cost_ablation",
    "scripts.run_hp_sens"
)

foreach ($module in $modules) {
    Write-Host "`nChecking CLI: $module"
    python -m $module --help *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "CLI check failed: $module"
    }
}

Write-Host "`nRunning Ruff..."
python -m ruff check src scripts analysis tests `
    --exclude scripts/legacy `
    --exclude analysis/legacy
if ($LASTEXITCODE -ne 0) { throw "Ruff linting failed" }

python -m ruff format --check src scripts analysis tests `
    --exclude scripts/legacy `
    --exclude analysis/legacy
if ($LASTEXITCODE -ne 0) { throw "Ruff formatting check failed" }

Write-Host "`nRunning tests..."
python -m pytest tests analysis/tests -v
if ($LASTEXITCODE -ne 0) { throw "Tests failed" }

Write-Host "`nAll Tri_Fair checks passed."
