# v5 Paper-Trading Bot (NSE)

Runs the v5 swing strategy on a paper account **every weekday at about 6:45 PM IST, on GitHub's servers**. It doesn't need your laptop or Colab.

Each evening it:
1. downloads end-of-day prices for the stocks in `stocks.txt` from Yahoo Finance;
2. fills yesterday's orders at today's open, checks stops and exits, and logs new signals with a fixed quantity (`state/paper_ledger.xlsx`);
3. rebuilds your **dashboard** (`docs/index.html`): account value, cash, open and realized P&L, tomorrow's orders, each open position with its stop, fill and current price, the market switch, strategy health, monthly returns, closed trades, and the equity curve against Nifty. It also writes `docs/REPORT.md`, a text version that GitHub shows directly;
4. optionally sends a **Telegram message** to your phone;
5. saves everything back to the repository (one commit per trading day).

On holidays it finds no new data and changes nothing.

## One-time setup (about 10 minutes)

1. **Create a GitHub account** at github.com if you don't have one.
2. **Create a new repository:** the **+** button → *New repository* → name it `swing-paper-bot` → choose **Private** → *Create repository*.
3. **Upload the files:** on the new repo page, click *uploading an existing file* and drag in **everything from this folder**, including the hidden `.github` folder. On a Mac, press `Cmd+Shift+.` in Finder to see hidden folders. Or use GitHub Desktop.
   *If the `.github` folder won't upload:* in the repo, click *Add file → Create new file*, type the name `.github/workflows/paper.yml`, paste the contents of that file, then commit.
4. **Allow the bot to save its results:** *Settings → Actions → General → Workflow permissions →* **Read and write permissions** → *Save*.
5. **Run it once now:** *Actions* tab → *v5 paper trading* → **Run workflow**. It takes 3–5 minutes. Afterwards, open `docs/REPORT.md` in the repo.

After that it runs automatically every Monday–Friday.

## Open your dashboard

Pick one:

**A. Public repository + GitHub Pages (free, simplest).** Make the repo public (*Settings → General → Change visibility*), then *Settings → Pages → Build and deployment → Deploy from a branch → `main` / `/docs` → Save*. Your dashboard is at `https://<your-username>.github.io/swing-paper-bot/` and updates every evening. Only paper-trading numbers are visible, nothing personal.

**B. Keep the repo private: Cloudflare Pages (free).** At dash.cloudflare.com go to *Workers & Pages → Create → Pages → Connect to Git*, pick the repo, leave **Build command empty**, set **Build output directory** to `docs`, then *Save and Deploy*. You get `https://swing-paper-bot.pages.dev`, which redeploys automatically after each run. To lock it to your email, add *Cloudflare Access* (free for personal use).

**C. No hosting.** Open `docs/index.html` in the repo, click **Download raw file**, and open it in your browser.

Put the address in `config.json` → `"dashboard_url"` and the evening Telegram message will include the link.

## Optional: Telegram alerts on your phone

1. In Telegram, message **@BotFather** → `/newbot` → follow the steps → copy the **token**.
2. Send any message to your new bot, then open `https://api.telegram.org/bot<TOKEN>/getUpdates` in a browser and copy the number after `"chat":{"id":`.
3. In the repo: *Settings → Secrets and variables → Actions → New repository secret*:
   - `TELEGRAM_TOKEN` = the token
   - `TELEGRAM_CHAT_ID` = the chat id

Each evening you'll get something like:
```
v5 paper · 12 Nov 2026
Equity ₹102,361 (+2.36%) · Nifty +1.56%
Breadth 60% · trend ON
Filled: CHOLAFIN @2197.89
Closed: MCX -1.0%
Tomorrow buy at open: SAIL x14
Open: MAZDOCK +15.8%, CHOLAFIN +11.5%
```

## Settings: `config.json`

| Key | Meaning |
|---|---|
| `capital` | Paper capital (used on the first run only) |
| `version` | `"v5 core"` (6 positions, 1.5% risk) or `"v5 more trades"` (10 positions, 1% risk, wider Oversold leader) |
| `cost_pct` | Round-trip costs, % of trade value |
| `start_from` | e.g. `"2026-10-01"` to start the account on a past date (first run only). `null` = start today |
| `dashboard_url` | Your dashboard address (optional), added to the Telegram message |

To restart from scratch, delete `state/paper_ledger.xlsx`.

## Good to know
- **Cost:** private repos get 2,000 free GitHub Actions minutes a month. This bot uses about 100.
- **Timing:** GitHub sometimes starts scheduled jobs 5–30 minutes late. That's fine; it processes whatever trading days are new.
- **Missed runs** (a GitHub outage, say) are caught up automatically on the next run, using only the data each day would have had.
- **No margin:** a position never uses more than 1/6 of the account (1/10 for "more trades") or more than the cash available. On the real 2022–26 trades this costs about 1 point of CAGR compared with the backtests, which allowed up to ~140% invested.
- **Don't edit** `state/paper_ledger.xlsx` by hand. The trades it logs are locked decisions.
- Data comes from Yahoo Finance's free end-of-day feed. If Yahoo hasn't updated by 6:45 PM, that day is processed the next evening.

*Paper trading only. Not investment advice.*
