import numpy as np

class TemporalAggregator:
    def __init__(self, n_blocks: int, agg_func: str = "mean"):
        """
        Parameters
        ----------
        n_blocks : int
            Number of time blocks to reduce to
        agg_func : str
            Aggregation function: "mean", "sum", "max"
        """
        self.n_blocks = n_blocks
        self.agg_func = agg_func

    def transform(self, values: np.ndarray) -> np.ndarray:
        if values.ndim != 1:
            raise ValueError("TemporalAggregator only supports 1D arrays")
        n = len(values)
        if n < self.n_blocks:
            return values  # If there are fewer values than blocks, just return the original values
        else:
            block_size = n // self.n_blocks
            reduced = []


            for i in range(min(self.n_blocks, n)):  # in case n < n_blocks  
                start = i * block_size
                end = (i + 1) * block_size if i < self.n_blocks - 1 else n
                block = values[start:end]

                if self.agg_func == "mean":
                    reduced.append(np.mean(block))
                elif self.agg_func == "sum":
                    reduced.append(np.sum(block))
                elif self.agg_func == "max":
                    reduced.append(np.max(block))
                else:
                    raise ValueError(f"Unknown aggregation: {self.agg_func}")

            return np.array(reduced)
    

class Discretizer:
    def __init__(
        self,
        vmin: float,
        vmax: float,
        n_bins: int,
        temporal_aggregator: TemporalAggregator | None = None,
    ):
        self.vmin = vmin
        self.vmax = vmax
        self.n_bins = n_bins
        self.temporal_aggregator = temporal_aggregator
        self.bin_edges = np.linspace(vmin, vmax, n_bins + 1)

    def _digitize(self, values: np.ndarray) -> np.ndarray:
        values = np.clip(values, self.vmin, self.vmax)
        bin_idx = np.digitize(values, self.bin_edges) - 1
        return np.minimum(bin_idx, self.n_bins - 1)

    def discretize(self, value) -> np.ndarray:
        if isinstance(value, np.ndarray):
            if self.temporal_aggregator:  # If there is a temporal aggregator, we use it
                values = self.temporal_aggregator.transform(value)
            else:
                values = value
        else: # In case it's a float or int
            values = np.array([value])

        return self._digitize(values)

    
class StateDiscretizer:
    def __init__(self, config: dict):
        self.discretizers = {}

        for var, cfg in config.items():
            aggregator = None

            if "temporal" in cfg:
                aggregator = TemporalAggregator(
                    n_blocks=cfg["temporal"]["n_blocks"],
                    agg_func=cfg["temporal"].get("agg", "mean"),
                )

            self.discretizers[var] = Discretizer(
                vmin=cfg["min"],
                vmax=cfg["max"],
                n_bins=cfg["bins"],
                temporal_aggregator=aggregator,
            )

    def transform(self, obs: dict = {}, predictions: dict = {}) -> tuple:
        state = []
        for var, value in obs.items():
            if var in self.discretizers.keys():
                values = self.discretizers[var].discretize(value)
                state.extend(values.tolist())
            else:
                if isinstance(value, int):
                    state.extend([value])
                else:
                    raise ValueError(f'Observed variable {var} is not an integer and no discretizer was provided')
        for var, value in predictions.items():
            if var in self.discretizers.keys():
                values = self.discretizers[var].discretize(value)
                state.extend(values.tolist())
            else:
                raise ValueError(f'Prediction variable {var} is not an integer and no discretizer was provided')
        return tuple(state)