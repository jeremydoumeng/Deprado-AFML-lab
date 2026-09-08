# Why the data is simulated

The book's methods are defined on **ticks**: the tick rule, imbalance bars, Roll's
spread and Kyle's lambda are all statements about individual trades. Free daily OHLC
would make most of the pipeline decorative.

So the input is a trade tape from a structural model whose parameters are known
(`afml/data/synthetic.py`): a persistent latent information process drives both order
flow and the drift of an efficient price, Kyle-style permanent impact enters through
signed root-volume, a bid-ask bounce is added on top, and trades arrive at
information-dependent Poisson times. The defaults produce ~1000 days of tape at
~24% annualised volatility, a 5bp spread, and an 80% tick-rule hit rate.

This buys something real data cannot: **the estimators can be graded**. The test suite
asserts that Roll's estimator recovers the true spread to within 2% when its assumptions
hold, that it converges to `s·(1−ρ)` under order-flow autocorrelation, and that it is
inflated by `√(1 + 2λ·E[√V]/s)` when permanent impact is present. Those are the
estimators' known biases, derived and then measured rather than hoped away.

The Sharpe ratio reported in [results](results.md) is therefore a property of the
generator, not a claim about markets. What transfers is the machinery.
