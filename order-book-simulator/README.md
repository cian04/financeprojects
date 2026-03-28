# Order Book Microstructure Simulator

Event-driven simulation of a limit order book with informed/uninformed traders and an adaptive market maker agent.

## What it models

- **Poisson order arrivals** — uninformed (noise) and informed traders arrive at independent rates
- **Glosten-Milgrom adverse selection** — informed traders know the fundamental value V(t) and trade when mid deviates from it
- **Avellaneda-Stoikov market maker** — continuously quotes bid/ask, adjusting spread for inventory risk and detected informed flow
- **GBM fundamental value** — true price follows geometric Brownian motion, invisible to uninformed traders

## Key concepts demonstrated

| Concept | Implementation |
|---|---|
| Price-time priority | FIFO matching engine with lazy cancellation |
| Adverse selection | MM widens spread when informed flow EWM rises |
| Inventory risk | MM spread grows linearly with net position |
| Price impact | Market orders walk the book, moving mid-price |
| Information asymmetry | Informed traders use market orders when deviation > 5 ticks |

## Usage
```bash
pip install -r requirements.txt
python3 src/main.py

# Custom parameters
python3 src/main.py --T 7200 --informed 0.2 --spread 0.03 --seed 123
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--T` | 3600 | Simulation horizon (seconds) |
| `--informed` | 0.10 | Informed trader fraction |
| `--lambda_u` | 2.0 | Uninformed arrival rate (orders/s) |
| `--spread` | 0.02 | MM base half-spread |
| `--seed` | 42 | Random seed |

## Outputs (in `results/`)

- `mid_price_vs_fundamental.png` — mid tracks fundamental with lag
- `spread_dynamics.png` — realised vs quoted spread over time
- `market_maker_pnl.png` — MM cumulative P&L and inventory
- `depth_profile.png` — final order book depth snapshot
- `trade_distribution.png` — informed vs uninformed trade sizes
- `performance_summary.csv` — all metrics

## Architecture
```
order-book-simulator/
├── src/
│   ├── order.py         # Order and Trade dataclasses
│   ├── orderbook.py     # Matching engine (price-time priority)
│   ├── market_maker.py  # Adaptive MM agent (A-S + G-M models)
│   ├── simulator.py     # Poisson arrival process + event loop
│   ├── analytics.py     # Metrics + chart generation
│   └── main.py          # Entry point
├── results/
├── requirements.txt
└── README.md
```

## References

- Glosten & Milgrom (1985) — Bid, ask and transaction prices in a specialist market with heterogeneously informed traders
- Avellaneda & Stoikov (2008) — High-frequency trading in a limit order book
- Kyle (1985) — Continuous auctions and insider trading
