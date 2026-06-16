# I spent weeks looking for a trading signal. I found a bug.

There's a specific kind of disappointment in quant research: the moment you realize your "interesting result" was a mistake. This is that story. I think publishing the mistake is more useful than the result would have been, and I'll explain why.

## The hypothesis

India's stock market has two dominant institutional players: Foreign Institutional Investors (FII) and Domestic Institutional Investors (DII). Every trading day, NSE publishes how much each group bought and sold. Financial Twitter treats divergence between the two, FII dumping while DII buys or vice versa, as a meaningful signal. The story goes: when foreign and domestic money disagree sharply, somebody is wrong about fair value, and that mispricing should show up in subsequent returns.

I wanted to test this properly. Not "does the market move the next day" — that's too noisy and too easy to fit by accident — but something sharper: does FII/DII divergence predict the direction of CAPM pricing error in sector indices? If flows are mispricing a sector, a rolling-beta CAPM model should be systematically wrong in a particular direction on the days following an extreme divergence.

I built a five-stage pipeline: pull market data, compute three signals (flow divergence, PCR deviation, implied-vs-realized volatility spread), flag days where multiple signals are simultaneously extreme, fit rolling CAPM betas per sector, and run a sign test on the forward pricing error. Clean, modular, defensible.

## The part where it almost worked

Early results looked promising in the way that should make any quant nervous: a few sector/horizon combinations cleared the significance bar, the robustness sweep showed a non-trivial fraction of parameter combinations significant, and there was a tidy story for why a commonly-used proxy (F&O futures positioning) appeared to get the sign wrong. The proposed mechanism: institutions hedge equity sells with futures longs, so naive proxies invert the direction.

I wrote it all up: null hypothesis rejected in a couple of cells, a structural explanation for a popular proxy's failure mode, clean plots. It read like a finished project.

## Then I went back and checked my own work

A few days later, auditing the same repo with fresh eyes (and, to be honest, with someone pushing back hard on every number), three things fell apart in sequence:

**1. The holdout leaked into training.** I'd set aside a separate block of recent data specifically to validate any finding out-of-sample. The entire point of a holdout is that you never touch it during model-building. At some point, while re-running the pipeline with updated data, that holdout silently got merged back into the main training file. Every "out-of-sample" claim I could have made was actually in-sample. This is one of the most basic mistakes in applied ML, and I made it anyway, because the merge happened during a routine data refresh, not during anything I was paying attention to.

**2. A headline number didn't match my own data file.** I'd written that 8.1% of robustness-sweep parameter combinations were significant, "consistent with chance." When I recomputed it directly from the CSV sitting in the same folder as the write-up, the real number was 21.25%. The write-up had been generated from an earlier run; the data had changed two days later; nobody regenerated the document. The lesson here isn't really statistical: a finding and the file that produced it can quietly drift apart, and the only defense is treating "reproduce it right now, from what's actually in the repo" as non-negotiable before trusting any number in a write-up.

**3. The interesting ancillary finding didn't survive a fair test.** The "F&O futures proxy is backwards" claim, the one that felt most publishable, turned out to be built on 100 days of data and a construction error: I'd compared a cumulative position (a stock) to a daily transaction (a flow), which is a bit like comparing someone's bank balance to their grocery bill and being surprised they don't move in sync. When I redid it with the full 1,750-day overlap and the correct flow-equivalent construction, the result was clear: no relationship at all, in either direction. Not backwards, just irrelevant. The original mechanism I'd proposed (institutions hedge equity sells with futures longs, inverting the sign) was a good story. It just wasn't true, or at least wasn't detectable in the data I had.

## What survived

After fixing the leakage, correcting the statistics, and properly testing the ancillary claim, here's what's actually left:

16 sector/forward-horizon combinations tested. Zero survive correction for multiple comparisons. One cell looked momentarily interesting (NIFTY_BANK at a 3-day horizon), survived a non-parametric permutation test, and then died the moment I accounted for the fact that I'd run 16 tests, not one. It also flipped direction across market regimes and carried an economically negligible effect size. Every honest signal of "this isn't real" was present; I just hadn't checked for all of them the first time.

A robustness sweep across 160 parameter combinations comes in at 5.6% significant, exactly what you'd expect from noise.

The F&O proxy finding is now: it's just noise, not a backwards signal. Weaker and less exciting than the original claim, and the correct one.

## Why I'm publishing a null result

The honest version of this project is less flattering than the first draft. There's no trading signal here. But it's more useful to other people than a flashy positive result would have been, for a specific reason: most quant blog posts show you a signal that worked. Almost none show you the actual mechanics of how a wrong signal gets manufactured, not through fraud or obvious carelessness, but through ordinary process gaps: a data refresh that silently breaks an isolation boundary, a write-up that outlives the data it described, a clever mechanistic story that nobody stress-tested with enough data.

If you're building anything similar, the checklist that would have caught all three problems earlier is short:

1. Physically separate your holdout data from your training pipeline. A different folder isn't enough if your code can still reach it; make it structurally impossible to re-merge.
2. Never trust a number in a write-up without regenerating it from the script and data sitting in the repo right now.
3. Before accepting a clean structural explanation for a finding, ask whether you tested it on enough data, and whether you're comparing the right kind of quantity (a level isn't a flow).
4. Always ask what your result looks like after correcting for every test you ran, not just the one you're excited about.

The market didn't give up an edge this time. But the process held up the second time I checked it, and that's the part actually worth writing about.

---

*Full technical write-up, all code, and the complete correction history (including the original, wrong version of the analysis, kept for transparency) are in the repository: [finding.md](finding.md).*
