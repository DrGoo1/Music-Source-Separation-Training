@echo off
REM Set environment variables to disable SVML before launching Python
set NUMBA_DISABLE_INTEL_SVML=1
set DISABLE_INTEL_SVML=1

REM Launch the Streamlit test for groove analysis
streamlit run test_groove_analysis.py
pause
