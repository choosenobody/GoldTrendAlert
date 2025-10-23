name: Test SoSoValue ETF API

on:
  workflow_dispatch:   # 手动运行按钮

permissions:
  contents: write      # 如需把API结果保存为CSV，需要写权限

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Show inputs & env
        env:
          BTC_ETF_API_URL:       ${{ vars.BTC_ETF_API_URL }}
          BTC_ETF_API_METHOD:    ${{ vars.BTC_ETF_API_METHOD }}
          BTC_ETF_API_HEADERS:   ${{ vars.BTC_ETF_API_HEADERS }}
          BTC_ETF_API_BODY:      ${{ vars.BTC_ETF_API_BODY }}
          BTC_ETF_FLOWS_CSV_URL: ${{ vars.BTC_ETF_FLOWS_CSV_URL }}
        run: |
          echo "API url     : ${BTC_ETF_API_URL:-<default>}"
          echo "API method  : ${BTC_ETF_API_METHOD:-POST}"
          echo "API headers : ${BTC_ETF_API_HEADERS:-{x-soso-api-key: <secret>, Content-Type: application/json}}"
          echo "API body    : ${BTC_ETF_API_BODY:-{\"type\":\"us-btc-spot\"}}"
          echo "CSV url     : ${BTC_ETF_FLOWS_CSV_URL:-<unset>}"

      - name: Call SoSoValue ETF API (diagnostic)
        id: call
        env:
          API_URL:       ${{ vars.BTC_ETF_API_URL }}
          API_METHOD:    ${{ vars.BTC_ETF_API_METHOD }}
          API_HEADERS:   ${{ vars.BTC_ETF_API_HEADERS }}
          API_BODY:      ${{ vars.BTC_ETF_API_BODY }}
          SOSO_API_KEY:  ${{ secrets.SOSOVALUE_API_KEY }}
        run: |
          set -euo pipefail

          API_URL="${API_URL:-https://api.sosovalue.xyz/openapi/v2/etf/historicalInflowChart}"
          API_METHOD="${API_METHOD:-POST}"
          API_BODY="${API_BODY:-{\"type\":\"us-btc-spot\"}}"

          # 构造 Headers（若未在 Variables 里写 headers，则自动补上 API KEY）
          HDR_JSON="${API_HEADERS:-}"
          if [ -z "$HDR_JSON" ]; then
            HDR_JSON="{\"x-soso-api-key\":\"$SOSO_API_KEY\",\"Content-Type\":\"application/json\"}"
          fi

          python - <<'PY' >/tmp/hdr.sh
          import json, os
          hdr = json.loads(os.environ["HDR_JSON"])
          out = " ".join(["-H "+json.dumps(str(k)+": "+str(v)) for k,v in hdr.items()])
          print(out)
          PY
          . /tmp/hdr.sh
          echo "[hdr]" $(/bin/cat /tmp/hdr.sh)

          if [ "${API_METHOD^^}" = "GET" ]; then
            echo "[GET] $API_URL"
            eval "curl -sS $(/bin/cat /tmp/hdr.sh) \"$API_URL\" -o /tmp/resp.json"
          else
            echo "[POST] $API_URL"
            eval "curl -sS $(/bin/cat /tmp/hdr.sh) -d '$API_BODY' \"$API_URL\" -o /tmp/resp.json"
          fi

          # 统计行数并写入 GITHUB_OUTPUT
          python - <<'PY'
          import json, os, sys
          with open("/tmp/resp.json","r") as f:
              j=json.load(f)
          rows=(j.get("data") or {}).get("list") or []
          print("rows =", len(rows))
          if rows:
              print("first =", rows[0])
              print("last  =", rows[-1])
          with open(os.environ["GITHUB_OUTPUT"], "a") as g:
              g.write(f"rows={len(rows)}\n")
          PY

      - name: Fail if no rows
        if: steps.call.outputs.rows == '0'
        run: |
          echo "::error::SoSoValue 返回 0 行。请检查 SOSOVALUE_API_KEY / API_URL / METHOD / HEADERS / BODY。"
          exit 2

      # 可选：把 API 数据落盘为 CSV，供 Bot 的 CSV 回退使用
      - name: Save API result to CSV (.bot_state/btc_spot_etf_flows.csv)
        if: steps.call.outputs.rows != '0'
        run: |
          set -euo pipefail
          mkdir -p .bot_state
          python - <<'PY'
          import json, pandas as pd
          with open("/tmp/resp.json","r") as f:
              j=json.load(f)
          rows=(j.get("data") or {}).get("list") or []
          df=pd.DataFrame(rows)
          dcol=None
          for k in ["date","time","Date","day"]:
              if k in df.columns: dcol=k; break
          vcol=None
          for k in ["totalNetInflow","netInflow","TotalNetInflow","net_flow","netflow"]:
              if k in df.columns: vcol=k; break
          assert dcol and vcol, f"cannot find date/flow columns: {df.columns.tolist()}"
          out=df[[dcol,vcol]].copy()
          out.columns=["Date","NetFlowUSD"]
          out["Date"]=pd.to_datetime(out["Date"], errors="coerce")
          out=out.dropna().sort_values("Date")
          out.to_csv(".bot_state/btc_spot_etf_flows.csv", index=False)
          print("Wrote .bot_state/btc_spot_etf_flows.csv rows=", len(out))
          PY
          git config user.email "etl-bot@example.com"
          git config user.name  "etl-bot"
          git add .bot_state/btc_spot_etf_flows.csv
          git commit -m "auto: update btc_spot_etf_flows.csv from SoSoValue" || echo "no changes"
          git push || true
