Param(
    [switch]$BuildKB = $false
)

Set-Location $PSScriptRoot\..

if ($BuildKB) {
    python scripts/build_kb.py
}

streamlit run app/ui/streamlit_app.py
