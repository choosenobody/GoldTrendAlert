name: Valuation Grid Backtest v2

on:
  workflow_dispatch:

permissions:
  contents: write

jobs:
  backtest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          python -m pip install pandas requests yfinance numpy python-dateutil

      - name: Run Valuation Grid Backtest v2
        env:
          FRED_API_KEY: ${{ secrets.FRED_API_KEY }}
          WGC_CSV_URL: ${{ vars.WGC_CSV_URL }}
          REG_WINDOW_M: ${{ vars.REG_WINDOW_M }}          # 默认36
          CB_YOY_ROLL_M: ${{ vars.CB_YOY_ROLL_M }}        # 默认12
          GRID_BETA: ${{ vars.GRID_BETA }}                # "0.2,0.3,0.4"
          GRID_ALPHA: ${{ vars.GRID_ALPHA }}              # "0.01,0.02,0.03"
          GRID_GAP_PCT: ${{ vars.GRID_GAP_PCT }}          # "8,10,12"
        run: |
          python tools/valuation_backtest_grid_v2.py

      - name: Upload artifact (optional)
        uses: actions/upload-artifact@v4
        with:
          name: valuation_grid_top20
          path: .bot_state/valuation_grid_top20.csv
